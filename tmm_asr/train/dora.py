"""
tmm_asr.train.dora — DoRA fine-tuning of Whisper-medium on FLEURS (mix6).

Adaptation target: decoder only (all 24 layers, self_attn + encoder_attn,
q/k/v/out projections). Encoder is frozen throughout — unchanged from
openai/whisper-medium baseline, so encoder hidden states and token merging
behaviour can be compared directly before and after fine-tuning.

Cohort (mix6, balanced tonal/non-tonal split):
  Languages trained on: vi_vn ha_ng ln_cd ta_in mt_mt jv_id + 10% English
    (3 tonal: Vietnamese, Hausa, Lingala; 3 non-tonal: Tamil, Maltese, Javanese)
    6 families: Austroasiatic, Chadic, Bantu, Dravidian, Semitic, Austronesian
  Language balance:        temperature sampling T=0.5 (flattens toward uniform)
  Catastrophic forgetting: DoRA frozen base + joint multilingual + English anchor

Usage:
    # Single GPU
    python -m tmm_asr.train.dora
    python -m tmm_asr.train.dora --gpu 1
    python -m tmm_asr.train.dora --steps 1000   # quick pilot
    # Two GPUs (DDP via torchrun) — auto-detected via LOCAL_RANK
    torchrun --standalone --nproc_per_node=2 -m tmm_asr.train.dora --steps 2000
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate as evaluate_lib
import numpy as np
import torch
from datasets import Audio, concatenate_datasets, interleave_datasets, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise SystemExit(
        "peft not installed.\n"
        "Run: pip install 'peft>=0.13.0'\n"
        "Then re-run this script."
    )


# Constants
RANDOM_SEED     = 42
MODEL_ID        = "openai/whisper-medium"
MODEL_REVISION  = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
FLEURS_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"

FINETUNE_LANGS = ["vi_vn", "ha_ng", "ln_cd", "ta_in", "mt_mt", "jv_id"]
# Whisper code overrides: jv_id -> "jw" (Whisper uses legacy ISO 639-1 "jw"
# for Javanese, not the modern "jv"; see ``tmm_asr.data.languages``).
# Tamil and Maltese use their natural two-letter ISO codes.
WHISPER_CODES  = {
    "vi_vn": "vi", "ha_ng": "ha", "ln_cd": "ln",
    "ta_in": "ta", "mt_mt": "mt", "jv_id": "jw",
}

LORA_R       = 32
LORA_ALPHA   = 64
LR           = 1e-5
WARMUP_STEPS = 200
MAX_STEPS    = 2000
BATCH_SIZE   = 8
GRAD_ACCUM   = 2       # effective batch = 32 under 2-GPU DDP (8 per-device * 2 GPUs * 2 grad-accum); set to 4 for single-GPU to preserve effective batch 32
TEMP         = 0.5    # language sampling temperature — T=0.5 gives sqrt(n) weighting
ENGLISH_FRAC = 0.10   # fraction of training mix that is English

def _default_checkpoint_dir() -> str:
    """
    Resolve where DoRA training writes its checkpoints.

    Precedence:
      1. TMM_CHECKPOINT_DIR env var (if set) — used verbatim.
      2. ./checkpoints/whisper-medium-dora-mix6 relative to the CWD when
         `python -m tmm_asr.train.dora` is launched (fine in the working
         repo; wheel users must override).

    NEVER defaults to a path inside the installed package, so a wheel install
    cannot try to write ~1.9 GB of checkpoints into site-packages.
    """
    override = os.environ.get("TMM_CHECKPOINT_DIR")
    if override:
        return override
    return os.path.join(os.getcwd(), "checkpoints", "whisper-medium-dora-mix6")


CHECKPOINT_DIR = _default_checkpoint_dir()

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# Data loading
def load_fleurs_train(lang_id: str, processor: Any) -> Any:
    """
    Load FLEURS train split for one language, preprocess to log-mel + token ids.
    Uses raw_transcription to preserve tonal diacritics (critical for yo, ln, vi).
    """
    whisper_code = WHISPER_CODES[lang_id]
    print(f"  Loading FLEURS train: {lang_id} ({whisper_code}) ...")

    ds = load_dataset(
        "google/fleurs", lang_id,
        split="train",
        revision=FLEURS_REVISION,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess(batch):
        audio   = batch["audio"]
        inputs  = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        )
        batch["input_features"] = inputs.input_features[0]
        labels = processor.tokenizer(
            batch["raw_transcription"],
            language=whisper_code,
            task="transcribe",
        )
        batch["labels"] = labels.input_ids
        return batch

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=1)
    print(f"    {len(ds)} clips")
    return ds


def load_english_train(processor: Any, n_clips: int) -> Any:
    """Load English FLEURS train split, subsampled to n_clips."""
    print(f"  Loading FLEURS train: en_us (English anchor, {n_clips} clips) ...")
    ds = load_dataset(
        "google/fleurs", "en_us",
        split="train",
        revision=FLEURS_REVISION,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n_clips, len(ds))))

    def preprocess(batch):
        audio  = batch["audio"]
        inputs = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        )
        batch["input_features"] = inputs.input_features[0]
        labels = processor.tokenizer(
            batch["raw_transcription"],
            language="en",
            task="transcribe",
        )
        batch["labels"] = labels.input_ids
        return batch

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=1)
    print(f"    {len(ds)} clips (English anchor)")
    return ds


def build_interleaved_dataset(processor: Any):
    """
    Build joint multilingual dataset with temperature sampling.
    p_i ∝ n_i^TEMP  (T=0.5 → sqrt weighting, flattens large-language dominance)
    + 10% English added separately.
    """
    lang_datasets = []
    lang_sizes    = []

    for lang_id in FINETUNE_LANGS:
        ds = load_fleurs_train(lang_id, processor)
        lang_datasets.append(ds)
        lang_sizes.append(len(ds))

    total_target = sum(lang_sizes)
    n_english    = max(1, int(total_target * ENGLISH_FRAC))
    en_ds        = load_english_train(processor, n_english)
    lang_datasets.append(en_ds)
    lang_sizes.append(len(en_ds))

    # Temperature sampling: p_i ∝ n_i^TEMP
    weights = np.array([n ** TEMP for n in lang_sizes], dtype=np.float64)
    probs   = (weights / weights.sum()).tolist()

    lang_names = FINETUNE_LANGS + ["en_us"]
    print("\n  Sampling probabilities:")
    for name, n, p in zip(lang_names, lang_sizes, probs):
        print(f"    {name:8s}: {n:5d} clips  →  p={p:.3f}")

    combined = interleave_datasets(
        lang_datasets,
        probabilities=probs,
        seed=RANDOM_SEED,
        stopping_strategy="all_exhausted",
    )
    return combined


def build_validation_dataset(processor: Any, n_clips: int = 50) -> Any:
    """
    Load FLEURS validation splits for all 6 fine-tune languages, concatenated.
    50 clips per language = 300 total. Used for WER monitoring during training.
    Each transform binds its language code independently.
    """
    val_datasets = []
    for lang_id in FINETUNE_LANGS:
        whisper_code = WHISPER_CODES[lang_id]
        ds = load_dataset(
            "google/fleurs", lang_id,
            split="validation",
            revision=FLEURS_REVISION,
            trust_remote_code=True,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n_clips, len(ds))))

        def make_preprocess(wcode):
            def preprocess(batch):
                audio  = batch["audio"]
                inputs = processor.feature_extractor(
                    audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
                )
                batch["input_features"] = inputs.input_features[0]
                labels = processor.tokenizer(
                    batch["raw_transcription"], language=wcode, task="transcribe"
                )
                batch["labels"] = labels.input_ids
                return batch
            return preprocess

        ds = ds.map(make_preprocess(whisper_code), remove_columns=ds.column_names, num_proc=1)
        print(f"    val {lang_id}: {len(ds)} clips")
        val_datasets.append(ds)

    return concatenate_datasets(val_datasets)


# Data collator
@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad log-mel inputs
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad label sequences; replace padding token id with -100
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token per-sample — Whisper adds it internally during generation.
        # Per-sample check avoids silent skip when one sample in the batch lacks BOS.
        bos_id = self.processor.tokenizer.bos_token_id
        has_bos = labels[:, 0] == bos_id
        if has_bos.any():
            labels = torch.where(
                has_bos.unsqueeze(1),
                torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100)], dim=1),
                labels,
            )

        # Truncate to Whisper's hard decoder limit of 448 tokens
        labels = labels[:, :448]

        batch["labels"] = labels
        return batch


# Persist training logs at every evaluation step.
class LogCheckpointCallback(TrainerCallback):
    def __init__(self, log_json: str, log_csv: str, eval_steps: int):
        self.log_json  = log_json
        self.log_csv   = log_csv
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step % self.eval_steps != 0:
            return control
        import csv as _csv
        import json
        with open(self.log_json, "w") as f:
            json.dump(state.log_history, f, indent=2)
        all_keys = sorted({k for e in state.log_history for k in e})
        with open(self.log_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(state.log_history)
        return control


# WER callback
class WERCallback(TrainerCallback):
    """
    Computes WER on the validation set at each eval step by running generation
    directly. Bypasses Seq2SeqTrainer's predict_with_generate path, which
    crashes on transformers 5.x (labels passed to model.generate()).
    Logs eval_wer to state.log_history so it appears in training_log.json/csv.
    """
    def __init__(self, processor: Any, val_dataset: Any, collator: Any,
                 device: torch.device, eval_steps: int, batch_size: int = 16):
        self.processor   = processor
        self.val_dataset = val_dataset
        self.collator    = collator
        self.device      = device
        self.eval_steps  = eval_steps
        self.batch_size  = batch_size
        self._metric     = evaluate_lib.load("wer")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step % self.eval_steps != 0:
            return control

        # Unwrap DDP if running multi-GPU via torchrun
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.eval()
        unwrapped.config.use_cache = True
        try:
            all_preds, all_refs = [], []
            n = len(self.val_dataset)
            for i in range(0, n, self.batch_size):
                samples = [self.val_dataset[j] for j in range(i, min(i + self.batch_size, n))]
                batch   = self.collator(samples)
                feats   = batch["input_features"].to(self.device)
                labels  = batch["labels"].numpy()

                with torch.no_grad():
                    gen_ids = unwrapped.generate(
                        input_features=feats,
                        max_new_tokens=225,
                        forced_decoder_ids=None,
                    )

                pred_str  = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
                label_ids = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
                ref_str   = self.processor.batch_decode(label_ids, skip_special_tokens=True)

                all_preds.extend(pred_str)
                all_refs.extend(ref_str)

            wer = round(self._metric.compute(predictions=all_preds, references=all_refs), 4)
            print(f"\n  Step {state.global_step} — eval WER: {wer:.4f}")
            state.log_history.append({"eval_wer": wer, "step": state.global_step,
                                      "epoch": state.epoch})
        except Exception as e:
            print(f"\n  WARNING: WER eval at step {state.global_step} failed: {e}")
        finally:
            unwrapped.config.use_cache = False
            unwrapped.train()
        return control


# Model and DoRA setup
def build_model(device: torch.device) -> tuple:
    print(f"\nLoading {MODEL_ID} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model     = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, revision=MODEL_REVISION
    )

    # Disable Whisper's built-in forced tokens — labels carry them instead
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.config.use_cache          = False  # required for gradient checkpointing

    # DoRA config: all attention projections, all layers (encoder + decoder)
    dora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        use_dora=True,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, dora_config)

    # Freeze ALL encoder adapter weights — decoder-only adaptation.
    # Use "model.encoder." (trailing dot) to avoid matching "model.encoder_attn"
    # which is the decoder's cross-attention and must remain trainable.
    for name, param in model.named_parameters():
        if "model.encoder." in name:
            param.requires_grad = False

    # Required for gradient checkpointing to work with PEFT
    model.enable_input_require_grads()

    model.to(device)
    model.print_trainable_parameters()

    # PEFT's seq2seq wrapper forwards text-only inputs through Whisper to its
    # decoder, where they collide with explicit decoder arguments. Whisper does
    # not consume these inputs, so remove them at the generation-model boundary.
    _PEFT_TEXT_KWARGS = frozenset(["input_ids", "inputs_embeds",
                                   "decoder_inputs_embeds"])
    _wcg = model.base_model.model
    _orig_wcg_forward = _wcg.forward
    def _wcg_forward_filtered(*args, **kwargs):
        for k in _PEFT_TEXT_KWARGS:
            kwargs.pop(k, None)
        return _orig_wcg_forward(*args, **kwargs)
    _wcg.forward = _wcg_forward_filtered

    # Verify encoder is actually frozen before any training begins
    enc_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "model.encoder." in n]
    dec_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "decoder" in n]
    print(f"\n  Encoder trainable params: {len(enc_trainable)}  (expect 0)")
    print(f"  Decoder trainable params: {len(dec_trainable)}  (expect >0)")
    if enc_trainable:
        print("  ERROR: encoder freeze failed — do not proceed with this run.")
        for n in enc_trainable[:5]:
            print(f"    {n}")
        raise RuntimeError("Encoder adapters are trainable. Paper comparison invalid.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M  ({100*trainable/total:.2f}%)")
    print(f"  Device: {device}")
    return processor, model


# Training
def train(args):
    # GPU setup: single GPU (default) or multi-GPU via torchrun (DDP).
    # DataParallel crashes with PEFT; DDP launched by torchrun is safe because
    # each process owns exactly one GPU and gradients sync via NCCL.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        # torchrun sets LOCAL_RANK — use DDP, device assigned per process
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        # Single GPU: hide others to prevent DataParallel activation
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    processor, model = build_model(device)

    print("\nBuilding training dataset ...")
    train_dataset = build_interleaved_dataset(processor)
    print(f"  Total interleaved size: {len(train_dataset)}")

    print("\nBuilding validation dataset ...")
    val_dataset = build_validation_dataset(processor)
    print(f"  Total validation size: {len(val_dataset)}")

    collator = WhisperDataCollator(processor=processor)

    log_json = os.path.join(CHECKPOINT_DIR, "training_log.json")
    log_csv  = os.path.join(CHECKPOINT_DIR, "training_log.csv")

    wer_callback = WERCallback(
        processor=processor,
        val_dataset=val_dataset,
        collator=collator,
        device=device,
        eval_steps=args.eval_steps,
    )
    log_callback = LogCheckpointCallback(
        log_json=log_json,
        log_csv=log_csv,
        eval_steps=args.eval_steps,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        max_steps=args.steps,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        predict_with_generate=False,
        max_grad_norm=1.0,
        load_best_model_at_end=False,  # manual selection safer with PEFT adapters
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=RANDOM_SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
        callbacks=[wer_callback, log_callback],
    )

    # Auto-resume from latest checkpoint if one exists and --restart not set
    resume_from = None
    if not args.restart:
        resume_from = get_last_checkpoint(CHECKPOINT_DIR)
        if resume_from:
            print(f"\n  Resuming from checkpoint: {resume_from}")
        else:
            print("\n  No checkpoint found — starting fresh.")
    else:
        print("\n  --restart flag set — ignoring existing checkpoints.")

    print(f"\nStarting DoRA training: {args.steps} steps, lr={LR}, r={LORA_R}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}\n")

    trainer.train(resume_from_checkpoint=resume_from)

    # Only rank 0 saves files and prints summary
    if trainer.is_world_process_zero():
        model.save_pretrained(CHECKPOINT_DIR)
        processor.save_pretrained(CHECKPOINT_DIR)

        import csv as _csv
        import json
        log_history = trainer.state.log_history
        with open(log_json, "w") as f:
            json.dump(log_history, f, indent=2)
        all_keys = sorted({k for entry in log_history for k in entry})
        with open(log_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(log_history)

        val_loss_entries = {int(e["step"]): e for e in log_history if "eval_loss" in e}
        val_wer_entries  = {int(e["step"]): e for e in log_history if "eval_wer"  in e}
        all_steps = sorted(set(val_loss_entries) | set(val_wer_entries))
        if all_steps:
            print("\n  Validation metrics by checkpoint:")
            print(f"  {'step':>6}  {'eval_loss':>10}  {'eval_wer':>10}")
            for s in all_steps:
                loss_str = f"{val_loss_entries[s]['eval_loss']:.4f}" if s in val_loss_entries else "      —"
                wer_str  = f"{val_wer_entries[s]['eval_wer']:.4f}"  if s in val_wer_entries  else "      —"
                print(f"  {s:>6}  {loss_str:>10}  {wer_str:>10}")
            if val_wer_entries:
                best_s = min(val_wer_entries, key=lambda s: val_wer_entries[s]["eval_wer"])
                print(f"\n  Best checkpoint (lowest WER): step {best_s}  eval_wer={val_wer_entries[best_s]['eval_wer']:.4f}")
            elif val_loss_entries:
                best = min(val_loss_entries.values(), key=lambda e: e["eval_loss"])
                best_s = int(best["step"])
                print(f"\n  Best checkpoint (lowest loss): step {best_s}  eval_loss={best['eval_loss']:.4f}")
            print(f"  Use: --model_path {CHECKPOINT_DIR}/checkpoint-{best_s}")

        print(f"\nDoRA adapter saved to:  {CHECKPOINT_DIR}")
        print(f"Training log (JSON):    {log_json}")
        print(f"Training log (CSV):     {log_csv}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoRA fine-tuning of Whisper-medium")
    parser.add_argument("--gpu",        type=int,  default=0,         help="CUDA device index (default: 0)")
    parser.add_argument("--steps",      type=int,  default=MAX_STEPS, help=f"Training steps (default: {MAX_STEPS})")
    parser.add_argument("--eval-steps", type=int,  default=500,       help="Eval/save interval (default: 500)")
    parser.add_argument("--restart",    action="store_true",           help="Ignore existing checkpoints and start fresh")
    args = parser.parse_args()
    train(args)
