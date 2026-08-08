"""Evaluate Config A token merging with the DoRA Whisper-medium adapter.

The adapter is merged into the pinned base model before decoding. Each
fine-tuning language is evaluated once without merging and at TRR 0.05 through
0.40 on the deterministic cached FLEURS cohort. See ``docs/REPRODUCE.md`` for
the paper command and expected outputs.
"""

import argparse
import copy
import csv
import os
import random
import sys
import time
import warnings

import jiwer
import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from tmm_asr.data.languages import LANGUAGES
from tmm_asr.eval._dora_ckpt import resolve_dora_checkpoint
from tmm_asr.eval.pipeline import (
    COMPRESSION_RATIO_THRESHOLD,
    MODEL_NAME,
    MODEL_REVISION,
    _compression_ratio,
    _prepare_for_wer,
    _transcribe_with_fallback,
    preload_samples,
)
from tmm_asr.merging import attach_merging, detach_merging, read_seq_lens

warnings.filterwarnings("ignore")

# Experiment configuration
CONFIG_NAME    = "A"
CONFIG_LAYERS  = [2, 5, 8, 11, 14, 17, 20, 23]   # A-ToMe-style cascade
TARGET_TRRS    = [0.05, 0.10, 0.20, 0.30, 0.40]
N_DEFAULT      = 264
RANDOM_SEED    = 42

# Default: pull the mix6 adapter (checkpoint-2000) directly from HuggingFace.
# Override with --checkpoint to point at a local training directory.
DEFAULT_CKPT   = "dylan01163104/whisper-medium-dora-mix6"
# Pinned HuggingFace revision for the released DoRA-mix6 adapter.
# Kept in-sync with the frozen checkpoint used to produce the paper's
# §5.2 numbers. Ignored when --checkpoint is a local directory.
DORA_REVISION  = "ad9144916cf661ea2ef462ad273077343c3d803d"

# Six-language adapter cohort, balanced across tonal and non-tonal languages.
LANGS_FT = ["vi_vn", "ha_ng", "ln_cd", "ta_in", "mt_mt", "jv_id"]

# Output root — override with TMM_OUT_DIR env var or --out-dir CLI arg.
RESULTS_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(os.getcwd(), "outputs", "eval"),
)
SEP         = "=" * 84


def per_layer_r(trr: float, n_layers: int) -> float:
    if trr <= 0:
        return 0.0
    return 1.0 - (1.0 - trr) ** (1.0 / n_layers)


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_hallucination_stats(refs_raw, hyps_raw,
                                cr_threshold: float = COMPRESSION_RATIO_THRESHOLD):
    """
    Per-condition hallucination stats based on Whisper's own compression-ratio
    proxy (Radford et al. 2022 §3.3). Computed on the final post-fallback
    hypothesis. CR > 2.4 is the same threshold _transcribe_with_fallback uses
    for temperature retries — values above threshold survived all retries,
    indicating residual hallucination/repetition.

    Returns:
      halluc_rate     : fraction of hypotheses with CR > threshold (hallucinated)
      halluc_n_high_cr: count of such hypotheses
      halluc_n_empty  : count of empty/whitespace hypotheses (decoder gave up)
      halluc_n_total  : total samples
      mean_hyp_words  : mean word count of hypotheses
      mean_ref_words  : mean word count of references
    """
    n = len(hyps_raw)
    if n == 0:
        return {"halluc_rate": float("nan"),
                "halluc_n_high_cr": 0, "halluc_n_empty": 0,
                "halluc_n_total": 0,
                "mean_hyp_words": 0.0, "mean_ref_words": 0.0}

    n_empty, n_high_cr = 0, 0
    hyp_w_sum, ref_w_sum = 0, 0
    for r, h in zip(refs_raw, hyps_raw):
        if not h.strip():
            n_empty += 1
        else:
            if _compression_ratio(h) > cr_threshold:
                n_high_cr += 1
        hyp_w_sum += len(h.split())
        ref_w_sum += len(r.split())

    return {
        "halluc_rate":      n_high_cr / n,
        "halluc_n_high_cr": n_high_cr,
        "halluc_n_empty":   n_empty,
        "halluc_n_total":   n,
        "mean_hyp_words":   hyp_w_sum / n,
        "mean_ref_words":   ref_w_sum / n,
    }


def run_condition(model, processor, gen_cfg, samples, lang_id, normalizer,
                  merge_layers, trr):
    """
    One decode pass.
    Returns (wer, hyps_raw, refs_norm, hyps_norm, halluc_stats, seq_lens, elapsed_s).
    """
    if merge_layers and trr > 0:
        r = per_layer_r(trr, len(merge_layers))
        attach_merging(model, {layer: r for layer in merge_layers})
    else:
        r = 0.0

    t0 = time.time()
    try:
        refs_norm, hyps_norm, hyps_raw, refs_raw = [], [], [], []
        last_sls = None
        for s in samples:
            hyp_raw = _transcribe_with_fallback(
                model, processor, gen_cfg,
                input_features=s["feats"], lang_id=lang_id,
                encoder_outputs=None,
            )
            if r > 0:
                last_sls = read_seq_lens(model)
            hyp_norm = _prepare_for_wer(hyp_raw, lang_id, normalizer)
            refs_norm.append(s["r_norm"])
            hyps_norm.append(hyp_norm)
            hyps_raw.append(hyp_raw)
            refs_raw.append(s.get("ref_raw", ""))
        wer = (jiwer.wer(refs_norm, hyps_norm)
               if any(r_.strip() for r_ in refs_norm) else float("nan"))
        # Hallucination is computed on the RAW hypothesis (pre-normaliser) so
        # the compression-ratio measures what the decoder actually produced,
        # not what survived text-normalisation.
        halluc = compute_hallucination_stats(refs_raw, hyps_raw)
    finally:
        if r > 0:
            detach_merging(model)

    return wer, hyps_raw, refs_norm, hyps_norm, halluc, last_sls, time.time() - t0


def load_finetuned_model(checkpoint_dir: str, device: torch.device):
    """
    Load Whisper-medium + DoRA adapter, merge into base weights.

    The base model is moved to `device` before the adapter is attached, so
    `merge_and_unload()` and adapter loading use the same device.

    Both base and adapter are fp32 (`torch_dtype=torch.float32`; adapter
    safetensors verified fp32 at checkpoint-2000), so the merge is fp32+fp32
    and incurs no mixed-precision drift regardless of device.
    """
    # Resolve BEFORE loading the base model so a bad --checkpoint fails
    # instantly instead of wasting the several-second Whisper base load.
    try:
        resolved = resolve_dora_checkpoint(checkpoint_dir, DORA_REVISION)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"DoRA checkpoint resolution failed: {e}")

    print(f"  Loading base Whisper from {MODEL_NAME} (revision={MODEL_REVISION[:8]}) ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, revision=MODEL_REVISION,
    ).to(device)

    if resolved.kind == "local":
        print(f"  Loading DoRA adapter from local dir {resolved.ident} (onto {device}) ...")
        model = PeftModel.from_pretrained(model, resolved.ident)
    else:
        print(f"  Loading DoRA adapter from HF Hub {resolved.ident}"
              f" (revision={resolved.revision[:8]}, onto {device}) ...")
        model = PeftModel.from_pretrained(model, resolved.ident,
                                          revision=resolved.revision)
    print(f"  merge_and_unload() on {device} — baking DoRA deltas into base weights ...")
    model = model.merge_and_unload()
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS_FT,
                    help="FT languages (default: paper mix6 — vi_vn ha_ng ln_cd ta_in mt_mt jv_id)")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--trrs", nargs="+", type=float, default=TARGET_TRRS)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                    help=f"DoRA checkpoint dir (default: {DEFAULT_CKPT})")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output CSV filename.")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                    help="Directory for output CSVs. Default: $TMM_OUT_DIR or <cwd>/outputs/eval/")
    args = ap.parse_args()
    # Honour --out-dir (default is env var / repo-root outputs/eval).
    RESULTS_DIR = args.out_dir

    langs = args.langs
    trrs  = sorted(set(args.trrs))
    seed_everything()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix  = f"_{args.tag}" if args.tag else ""
    ckpt_short = os.path.basename(os.path.normpath(args.checkpoint))
    out_csv = os.path.join(
        RESULTS_DIR,
        f"finetune_merge_rerun_cfg{CONFIG_NAME}_n{args.n}_{ckpt_short}{suffix}.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print("  FT + MERGE RERUN — new K-cosine intra-block merging")
    print(f"  Config {CONFIG_NAME}: layers={CONFIG_LAYERS}")
    print("  Greedy only · no SP · no random")
    print(f"  Device     : {device}  ·  Model: {MODEL_NAME}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  FT langs   : {langs}")
    print(f"  TRRs       : {trrs}")
    for t in trrs:
        r = per_layer_r(t, len(CONFIG_LAYERS))
        print(f"    TRR={t:.2f}  →  per_layer_r={r:.4f}")
    print(f"  Output     : {out_csv}")
    print(f"{SEP}\n")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model     = load_finetuned_model(args.checkpoint, device)
    normalizer = BasicTextNormalizer()
    gen_cfg    = copy.deepcopy(model.generation_config)

    rows  = []
    t_run = time.time()

    for li, lang_id in enumerate(langs, start=1):
        info = LANGUAGES.get(lang_id, {})
        lang_name = info.get("name", lang_id)
        tonal     = info.get("tonal", "")
        resource  = info.get("resource", "")
        family    = info.get("family", "")

        print(f"\n{SEP}")
        print(f"  [{li}/{len(langs)}] LANG: {lang_id}  ({lang_name})  "
              f"tonal={tonal}  resource={resource}  family={family}")
        print(SEP)

        print(f"  Loading {args.n} samples ...", flush=True)
        samples = preload_samples(lang_id, args.n, processor, device)

        # --- FT-baseline (no merging) once per language ---
        print(f"  [{lang_id}] FT-baseline (no merging) ...", flush=True)
        (b_wer, _, _, _, b_halluc, _, b_elapsed) = run_condition(
            model, processor, gen_cfg, samples, lang_id, normalizer,
            merge_layers=[], trr=0.0)
        print(f"    FT-baseline WER = {b_wer*100:6.2f}%   "
              f"halluc_rate = {b_halluc['halluc_rate']*100:5.2f}%   "
              f"(n_high_cr={b_halluc['halluc_n_high_cr']}/"
              f"{b_halluc['halluc_n_total']}, "
              f"n_empty={b_halluc['halluc_n_empty']})   "
              f"({b_elapsed:5.1f}s)",
              flush=True)
        rows.append({
            "lang_id": lang_id, "lang_name": lang_name,
            "tonal": tonal, "resource": resource, "family": family,
            "checkpoint": ckpt_short,
            "config": "baseline_ft", "layers": "[]",
            "trr": 0.0, "per_layer_r": 0.0,
            "wer": round(b_wer, 6), "wer_delta": 0.0,
            "halluc_rate":      round(b_halluc["halluc_rate"], 6),
            "halluc_n_high_cr": b_halluc["halluc_n_high_cr"],
            "halluc_n_empty":   b_halluc["halluc_n_empty"],
            "mean_hyp_words":   round(b_halluc["mean_hyp_words"], 2),
            "mean_ref_words":   round(b_halluc["mean_ref_words"], 2),
            "seq_len_final": 1500, "n_samples": args.n,
            "elapsed_s": round(b_elapsed, 1),
        })
        b_halluc_rate = b_halluc["halluc_rate"]

        # --- TRR sweep under Config A ---
        for trr in trrs:
            r_layer = per_layer_r(trr, len(CONFIG_LAYERS))
            print(f"  [{lang_id}] Config {CONFIG_NAME} TRR={trr:.2f}  "
                  f"per_r={r_layer:.4f} ...", flush=True)
            (m_wer, _, _, _, m_halluc, sls, elapsed) = run_condition(
                model, processor, gen_cfg, samples, lang_id, normalizer,
                merge_layers=CONFIG_LAYERS, trr=trr)
            delta        = m_wer - b_wer
            halluc_delta = m_halluc["halluc_rate"] - b_halluc_rate
            sl_final     = sls.get(CONFIG_LAYERS[-1], None) if sls else None
            print(f"    TRR={trr:.2f}  WER={m_wer*100:6.2f}%  "
                  f"Δ={delta*100:+6.2f}pp  "
                  f"halluc={m_halluc['halluc_rate']*100:5.2f}% "
                  f"(Δ={halluc_delta*100:+5.2f}pp)  "
                  f"seq@L{CONFIG_LAYERS[-1]}={sl_final}  ({elapsed:5.1f}s)",
                  flush=True)
            rows.append({
                "lang_id": lang_id, "lang_name": lang_name,
                "tonal": tonal, "resource": resource, "family": family,
                "checkpoint": ckpt_short,
                "config": CONFIG_NAME, "layers": str(CONFIG_LAYERS),
                "trr": trr, "per_layer_r": round(r_layer, 6),
                "wer": round(m_wer, 6), "wer_delta": round(delta, 6),
                "halluc_rate":      round(m_halluc["halluc_rate"], 6),
                "halluc_n_high_cr": m_halluc["halluc_n_high_cr"],
                "halluc_n_empty":   m_halluc["halluc_n_empty"],
                "mean_hyp_words":   round(m_halluc["mean_hyp_words"], 2),
                "mean_ref_words":   round(m_halluc["mean_ref_words"], 2),
                "seq_len_final": sl_final, "n_samples": args.n,
                "elapsed_s": round(elapsed, 1),
            })

        # Incremental CSV save
        fieldnames = ["lang_id", "lang_name", "tonal", "resource", "family",
                      "checkpoint", "config", "layers", "trr", "per_layer_r",
                      "wer", "wer_delta",
                      "halluc_rate", "halluc_n_high_cr", "halluc_n_empty",
                      "mean_hyp_words", "mean_ref_words",
                      "seq_len_final", "n_samples", "elapsed_s"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  [{lang_id}] CSV updated → {out_csv}")

    # --- Final summary ---
    print(f"\n{SEP}\n  SUMMARY — FT-{ckpt_short} + Config {CONFIG_NAME} "
          f"{CONFIG_LAYERS}\n  WER deltas (pp) vs FT-baseline, n={args.n}\n{SEP}")
    hdr = "  ".join(f"TRR={t:.2f}".rjust(9) for t in trrs)
    print(f"  {'Lang':>8}  {'FT-base%':>10}  " + hdr)
    print(f"  {'-'*8:>8}  {'-'*10:>10}  " + "  ".join("-" * 9 for _ in trrs))

    by_lct = {(r["lang_id"], r["config"], r["trr"]): r for r in rows}
    baseline = {r["lang_id"]: r["wer"] for r in rows if r["config"] == "baseline_ft"}

    for lang in langs:
        b = baseline.get(lang, float("nan"))
        cells = []
        for t in trrs:
            r = by_lct.get((lang, CONFIG_NAME, t))
            d = r["wer_delta"] if r else float("nan")
            cells.append(f"{d*100:+6.2f}pp".rjust(9))
        print(f"  {lang:>8}  {b*100:>9.2f}%  " + "  ".join(cells))

    mean_cells = []
    for t in trrs:
        ds = [by_lct[(lang_id, CONFIG_NAME, t)]["wer_delta"]
              for lang_id in langs if (lang_id, CONFIG_NAME, t) in by_lct]
        m = sum(ds) / len(ds) if ds else float("nan")
        mean_cells.append(f"{m*100:+6.2f}pp".rjust(9))
    print(f"  {'MEAN Δ':>8}  {'':>10}  " + "  ".join(mean_cells))

    total_min = (time.time() - t_run) / 60
    print(f"\n  Total runtime: {total_min:.1f} min")
    print(f"  Saved CSV : {out_csv}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
