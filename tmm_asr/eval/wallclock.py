"""
tmm_asr.eval.wallclock — encoder + end-to-end wall-clock benchmark.

Measures whisper-medium wall-clock time using the paper merging method:
head-mean K-cosine similarity, intra-block placement (between the attention
residual and the FFN), Config A cascade [2, 5, 8, 11, 14, 17, 20, 23] — via
`tmm_asr.merging.attach_merging`, matching the method used to produce every
WER number in the paper.

Cohort defaults to the paper's mix6 (Hausa, Javanese, Lingala, Maltese,
Tamil, Vietnamese) with 8 samples/lang, matching the Limitations §
methodology block.

Conditions per sample:
  - baseline (no merge, TRR=0.0)
  - merged CTM at TRR ∈ {0.20, 0.30, 0.40}

Methodology:
  1. CUDA events for GPU-native timing rather than host timers.
  2. torch.cuda.synchronize() before and after every measured region.
  3. N_WARMUP iterations per condition discarded.
  4. Two-phase measurement per sample: encoder-only phase runs first
     (no intervening decoder passes), then E2E phase. Within each phase
     the condition order is Latin-square rotated by sample_idx, so each
     condition sits in each ordinal position an equal number of times
     across the run — position-in-sequence bias averages out.
  5. torch.backends.cudnn.benchmark = False, deterministic = True.
  6. Default attn_implementation (deployment config).
  7. float32 dtype matched across all conditions.
  8. Decoder generation is capped at max_new_tokens but stops on EOS;
     different conditions can emit different token counts. Per-condition
     token counts are recorded in wallclock.csv so E2E speedup is honestly
     reported as end-to-end latency, not as isolated encoder cost. For the
     paper's encoder claim, run --encoder-only.
  9. gc.disable() wraps every timed region — Python GC pauses can add 50ms+
     of random noise if a collection fires mid-measurement.
 10. Hardware + software versions logged at startup and recorded in the CSV.

Outputs (default: outputs/wallclock/ under the invocation CWD):
  wallclock.csv          — long-form, one row per (lang, method, trr, sample, iter)
  wallclock_summary.csv  — aggregate mean/std per (lang, method, trr) cell
                            with speedup ratios vs baseline

Run:
  CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.wallclock

Smoke-test (2 langs, 3 samples, ~2 min):
  CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.wallclock \\
      --langs vi_vn ha_ng --n-samples 3
"""

import argparse
import csv
import gc
import logging
import os
import platform
import subprocess
import time
import warnings

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from tmm_asr.data.fleurs import load_or_cache_fleurs
from tmm_asr.data.languages import get_whisper_code
from tmm_asr.eval.pipeline import compute_silence_info, per_layer_r_for_trr
from tmm_asr.merging import (
    attach_merging,
    detach_merging,
    read_patched_layers,
    read_seq_lens,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# Configuration
MODEL_NAME       = "openai/whisper-medium"
MODEL_REVISION   = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
DATASET_NAME     = "google/fleurs"
DATASET_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"

# Paper method: Config A cascade on whisper-medium — 8 merge layers, cascade
# step 3, matching every WER number in the paper (via tmm_asr.merging).
CONFIG_A_LAYERS  = [2, 5, 8, 11, 14, 17, 20, 23]

# Paper mix6 wall-clock cohort (Limitations §): 6 low/mid-resource langs.
DEFAULT_LANGS    = ["ha_ng", "jv_id", "ln_cd", "mt_mt", "ta_in", "vi_vn"]

DEFAULT_TRRS     = [0.20, 0.30, 0.40]
BATCH_SIZE          = 1    # locked: latency benchmark, not throughput
DEFAULT_N_SAMPLES   = 8    # paper Limitations § reports 8 utterances/lang
N_GLOBAL_WARMUP     = 10   # full forward passes at script start to bring GPU to max clock
DEFAULT_N_WARMUP    = 5    # per-condition warmup (discarded) inside measure()
DEFAULT_N_ENC_ITERS = 10
DEFAULT_N_E2E_ITERS = 5
MAX_NEW_TOKENS      = 200
SEED                = 1234

# Outputs land under <cwd>/outputs/wallclock/ by default; overridable via --out-dir.
_DEFAULT_OUT_DIR = os.path.join("outputs", "wallclock")

SEP  = "=" * 84
DSEP = "─" * 84


# Setup
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for wall-clock benchmarking.")
    return torch.device("cuda")


def set_deterministic():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gpu_temperature_c() -> int:
    """
    Query current GPU temperature via nvidia-smi. Returns degrees Celsius.
    Uses CUDA_VISIBLE_DEVICES to find the physical GPU index so the reading
    matches the device actually running the benchmark.
    Recorded per sample so thermal stability can be checked across the run.
    Returns -1 if query fails (non-fatal).
    """
    try:
        # CUDA_VISIBLE_DEVICES may remap device 0 to a physical GPU other than 0
        phys_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        result = subprocess.run(
            ["nvidia-smi",
             f"--id={phys_idx}",
             "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def global_gpu_warmup(model, device):
    """
    Drive the GPU through N_GLOBAL_WARMUP full encoder + short decoder passes
    on zero-filled dummy data before any measurement begins.

    Why this is necessary:
      GPUs idle in a low-power state (reduced clock frequency). The first real
      compute after idling triggers clock ramp-up, which takes several ms.
      Without this warmup, whichever condition happened to be measured first
      for a sample would appear artificially slow. Condition order is now
      Latin-square counterbalanced (see condition_order), but the global
      warmup is still needed so the very first sample of the run does not
      pay the cold-clock penalty.

    Dummy data shape matches real Whisper input: [1, 80, 3000] float32.
    We use zero-filled features (valid mel shape, all silence) so no actual
    transcription content is processed.
    """
    print(f"  Global GPU warmup: {N_GLOBAL_WARMUP} full encoder+decoder passes on dummy data ...",
          flush=True)
    dummy_features = torch.zeros(1, 80, 3000, dtype=torch.float32, device=device)
    for _ in range(N_GLOBAL_WARMUP):
        with torch.no_grad():
            enc_hidden = model.model.encoder(dummy_features).last_hidden_state
            enc_out    = BaseModelOutput(last_hidden_state=enc_hidden)
            model.generate(
                encoder_outputs=enc_out,
                max_new_tokens=20, do_sample=False, num_beams=1,
            )
        torch.cuda.synchronize()
    print("  GPU warmup complete. GPU is now at full clock frequency.\n", flush=True)


def log_environment(device):
    gpu_name = torch.cuda.get_device_name(device)
    gpu_cap  = torch.cuda.get_device_capability(device)
    vram_gb  = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(SEP)
    print("  WALL-CLOCK BENCHMARK — Whisper-medium")
    print(SEP)
    print(f"  GPU              : {gpu_name}  (cap={gpu_cap}, {vram_gb:.1f} GB)")
    print(f"  CUDA             : {torch.version.cuda}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  Transformers     : {transformers.__version__}")
    print(f"  Python           : {platform.python_version()}")
    print(f"  Platform         : {platform.platform()}")
    print(f"  Batch size       : {BATCH_SIZE}  (latency benchmark, not throughput)")
    start_temp = gpu_temperature_c()
    print(f"  GPU temp at start: {start_temp}°C")
    print(SEP)
    return {
        "gpu_name":          gpu_name,
        "cuda_version":      torch.version.cuda,
        "torch_version":     torch.__version__,
        "transformers_ver":  transformers.__version__,
        "batch_size":        BATCH_SIZE,
        "start_temp_c":      start_temp,
    }


# CUDA-event timing primitives
def _time_one(fn, *args, **kwargs):
    """Time a single GPU operation with proper sync. Returns elapsed ms."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def measure(fn, args=(), kwargs=None, n_warmup=DEFAULT_N_WARMUP, n_iter=10):
    """
    Run warmups (discarded) then n_iter measured iterations. Returns list[ms].

    GC is disabled for the entire duration — warmup + measurement — and
    re-enabled via finally so it always restores even on exception.
    A GC collection mid-measurement adds 50ms+ of random noise that would
    corrupt individual iteration times and widen the std unpredictably.
    """
    kwargs = kwargs or {}
    gc.disable()
    try:
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iter):
            ms, _ = _time_one(fn, *args, **kwargs)
            times.append(ms)
    finally:
        gc.enable()
    return times


# Timed condition runners
def enc_forward(model, input_features):
    """
    Timed function: one encoder forward pass. Merging state is set OUTSIDE
    this call via attach_merging(model, ..., record_stats=False), so this
    function times only the forward-pass cost of the paper method.
    """
    with torch.no_grad():
        return model.model.encoder(input_features).last_hidden_state


def e2e_forward(model, input_features, lang_code, return_ids=False):
    """
    Timed function: full model.generate() end-to-end.

    Note: generation stops on EOS regardless of max_new_tokens, so different
    encoder conditions may emit different token counts. This function returns
    the generated ids when return_ids=True so the caller can log per-condition
    decoder length for honest reporting.
    """
    with torch.no_grad():
        kwargs = dict(
            input_features=input_features,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        )
        if lang_code is not None:
            kwargs["language"] = lang_code
            kwargs["task"]     = "transcribe"
        ids = model.generate(**kwargs)
    return ids if return_ids else None


def count_decoder_tokens(model, input_features, lang_code) -> int:
    """
    Run one untimed generate() pass and return the non-special token count.
    Also serves as a real-data warmup before the timed measurement.
    """
    ids = e2e_forward(model, input_features, lang_code, return_ids=True)
    return max(0, int(ids.shape[-1]) - 1)


# Merge attachment validation
def assert_merging_attached(model, expected_layers, expected_seq_len_range=None):
    """
    Sanity: patched layers match expectation, and (optionally) after one dummy
    forward the final layer's post-merge seq_len falls in the expected range.

    Raises RuntimeError with actionable message on any mismatch.
    """
    patched = read_patched_layers(model)
    if patched != sorted(expected_layers):
        raise RuntimeError(
            f"attach_merging assertion FAILED: patched={patched}, expected={sorted(expected_layers)}"
        )
    if expected_seq_len_range is not None:
        lo, hi = expected_seq_len_range
        dummy = torch.zeros(1, 80, 3000, dtype=torch.float32,
                            device=next(model.parameters()).device)
        with torch.no_grad():
            model.model.encoder(dummy)
        seq_lens = read_seq_lens(model)
        final = seq_lens.get(max(patched))
        if final is None or not (lo <= final <= hi):
            raise RuntimeError(
                f"attach_merging seq_len assertion FAILED: final seq_len={final}, "
                f"expected in [{lo}, {hi}]. Full trajectory: {seq_lens}"
            )


# Latin-square counterbalancing
def condition_order(conditions, sample_idx: int) -> list:
    """
    Cyclic Latin-square permutation: sample i uses the conditions rotated by i.
    Over k = len(conditions) consecutive samples, each condition appears in
    each ordinal position exactly once. Removes first-position (cold-GPU) bias.
    """
    k = len(conditions)
    shift = sample_idx % k
    return list(conditions[shift:]) + list(conditions[:shift])


# Per-sample measurement
def measure_sample_encoder(model, input_features, sample_idx, trrs,
                            n_warmup, n_enc_iters):
    """
    Encoder-only per-sample measurement, counterbalanced condition order.

    Returns dict keyed by (method, trr) -> {"enc_ms": [...]}.
    method in {"baseline", "merged"}; baseline uses trr=0.0.

    Conditions are the baseline plus one merged condition per TRR. Their run
    order rotates by sample_idx (see condition_order), so any position-in-
    sequence bias averages out across samples rather than being pinned to the
    baseline. No decoder passes intervene in this phase.
    """
    conditions = [("baseline", 0.0)] + [("merged", trr) for trr in trrs]

    # Precompute merge dicts once per TRR
    merge_dicts = {trr: {layer: per_layer_r_for_trr(trr, len(CONFIG_A_LAYERS))
                          for layer in CONFIG_A_LAYERS}
                   for trr in trrs}

    out = {}
    for method, trr in condition_order(conditions, sample_idx):
        detach_merging(model)  # always start clean
        if method == "merged":
            attach_merging(model, merge_dicts[trr], record_stats=False)
        try:
            enc_t = measure(enc_forward, args=(model, input_features),
                            n_warmup=n_warmup, n_iter=n_enc_iters)
        finally:
            detach_merging(model)
        out[(method, trr)] = {"enc_ms": enc_t, "e2e_ms": []}
    return out


def measure_sample_e2e(model, input_features, lang_code, sample_idx, trrs,
                        n_warmup, n_e2e_iters):
    """
    End-to-end per-sample measurement, counterbalanced condition order.

    Records the generated-token count per condition into out[(m,trr)]["n_tok"]
    since generation stops on EOS and different encoder conditions can emit
    different lengths. Downstream e2e speedup comparisons must be interpreted
    with these counts in mind — they are not an isolated encoder measurement.
    """
    conditions = [("baseline", 0.0)] + [("merged", trr) for trr in trrs]
    merge_dicts = {trr: {layer: per_layer_r_for_trr(trr, len(CONFIG_A_LAYERS))
                          for layer in CONFIG_A_LAYERS}
                   for trr in trrs}

    out = {}
    for method, trr in condition_order(conditions, sample_idx):
        detach_merging(model)
        if method == "merged":
            attach_merging(model, merge_dicts[trr], record_stats=False)
        try:
            # Record per-condition decoder length (untimed)
            n_tok = count_decoder_tokens(model, input_features, lang_code)
            e2e_t = measure(e2e_forward, args=(model, input_features, lang_code),
                            n_warmup=n_warmup, n_iter=n_e2e_iters)
        finally:
            detach_merging(model)
        out[(method, trr)] = {"enc_ms": [], "e2e_ms": e2e_t, "n_tok": n_tok}
    return out


# Aggregation and CSV output
def write_long_row(writer, env, lang_id, method, trr, sample_idx, iter_idx,
                   metric, ms, temp_c, dur_s, n_sil, n_sph, n_decoder_tokens):
    writer.writerow({
        "gpu":               env["gpu_name"],
        "batch_size":        env["batch_size"],
        "torch":             env["torch_version"],
        "transformers":      env["transformers_ver"],
        "attn_impl":         env.get("attn_implementation", ""),
        "lang_id":           lang_id,
        "method":            method,
        "trr":               f"{trr:.2f}",
        "sample_idx":        sample_idx,
        "temp_c":            temp_c,
        "dur_s":             f"{dur_s:.2f}",
        "n_sil":             n_sil,
        "n_sph":             n_sph,
        "n_decoder_tokens":  n_decoder_tokens,
        "iter_idx":          iter_idx,
        "metric":            metric,    # "enc" or "e2e"
        "ms":                f"{ms:.4f}",
    })


def aggregate_and_write_summary(all_results, env, summary_csv):
    """
    Compute per-sample speedup ratios then average them.

    Why NOT ratio-of-pooled-means:
      Without clock locking, the GPU drifts thermally across the run. Pooling
      all baseline iterations and all treatment iterations conflates samples
      measured at different temperatures. Per-sample ratios cancel that
      drift because each ratio is computed from measurements taken within the
      same ~30-60s window on the same audio, with the same thermal state.

    Why counterbalanced ordering also matters:
      Even within one sample, the FIRST condition measured runs against a
      slightly cooler GPU than the LAST. If baseline were pinned to position
      zero every time, per-sample ratios would still be biased. The condition
      order is Latin-square rotated by sample_idx (see condition_order), so
      each condition sits in each ordinal position an equal number of times
      across the run and position bias averages out.
    """
    # Group by (lang, method, trr, sample_idx) — keep samples separate
    per_sample = {}
    for r in all_results:
        k = (r["lang"], r["method"], r["trr"], r["sample"])
        per_sample[k] = {
            "enc_mean": float(np.mean(r["enc_ms"])) if r["enc_ms"] else float("nan"),
            "e2e_mean": float(np.mean(r["e2e_ms"])) if r["e2e_ms"] else float("nan"),
        }

    # Collect per-sample baseline means keyed by (lang, sample_idx)
    base_per_sample = {}
    for (lang, method, trr, sidx), v in per_sample.items():
        if method == "baseline":
            base_per_sample[(lang, sidx)] = {
                "enc": v["enc_mean"],
                "e2e": v["e2e_mean"],
            }

    # For each (lang, method, trr) compute per-sample speedup then aggregate
    cells = {}
    for (lang, method, trr, sidx), v in per_sample.items():
        k = (lang, method, trr)
        cells.setdefault(k, {"enc_ms_all": [], "e2e_ms_all": [],
                             "enc_speedups": [], "e2e_speedups": []})
        cells[k]["enc_ms_all"].append(v["enc_mean"])
        cells[k]["e2e_ms_all"].append(v["e2e_mean"])
        b = base_per_sample.get((lang, sidx))
        if b:
            if b["enc"] > 0 and not np.isnan(v["enc_mean"]):
                cells[k]["enc_speedups"].append(b["enc"] / v["enc_mean"])
            if b["e2e"] > 0 and not np.isnan(v["e2e_mean"]):
                cells[k]["e2e_speedups"].append(b["e2e"] / v["e2e_mean"])

    with open(summary_csv, "w", newline="") as f:
        cols = ["gpu", "batch_size", "torch", "transformers", "attn_impl",
                "lang_id", "method", "trr", "n_samples",
                "enc_mean_ms", "enc_std_ms",
                "enc_speedup_mean", "enc_speedup_std",
                "e2e_mean_ms", "e2e_std_ms",
                "e2e_speedup_mean", "e2e_speedup_std"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (lang, method, trr), v in sorted(cells.items()):
            # Filter NaN sample means so std over an all-NaN list is NaN, not 0.
            enc = np.array([x for x in v["enc_ms_all"] if not np.isnan(x)])
            e2e = np.array([x for x in v["e2e_ms_all"] if not np.isnan(x)])
            es  = np.array(v["enc_speedups"])
            es2 = np.array(v["e2e_speedups"])
            def _mean(a): return float(a.mean()) if len(a) else float("nan")
            # Sample standard deviation is undefined for fewer than two values.
            def _std(a):  return float(a.std(ddof=1)) if len(a) > 1 else float("nan")
            w.writerow({
                "gpu":               env["gpu_name"],
                "batch_size":        env["batch_size"],
                "torch":             env["torch_version"],
                "transformers":      env["transformers_ver"],
                "attn_impl":         env.get("attn_implementation", ""),
                "lang_id":           lang,
                "method":            method,
                "trr":               f"{trr:.2f}",
                "n_samples":         len(enc),
                "enc_mean_ms":       f"{_mean(enc):.4f}",
                "enc_std_ms":        f"{_std(enc):.4f}",
                "enc_speedup_mean":  f"{_mean(es):.4f}",
                "enc_speedup_std":   f"{_std(es):.4f}",
                "e2e_mean_ms":       f"{_mean(e2e):.4f}",
                "e2e_std_ms":        f"{_std(e2e):.4f}",
                "e2e_speedup_mean":  f"{_mean(es2):.4f}",
                "e2e_speedup_std":   f"{_std(es2):.4f}",
            })


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=None,
        help=f"Space-separated lang ids. Default: paper mix6 ({' '.join(DEFAULT_LANGS)}).")
    parser.add_argument("--trrs", nargs="+", type=float, default=None,
        help=f"Space-separated TRRs. Default: {' '.join(map(str, DEFAULT_TRRS))}")
    parser.add_argument("--n-samples",    type=int,   default=DEFAULT_N_SAMPLES,
        help=f"Utterances per language (default {DEFAULT_N_SAMPLES}, paper Limitations §).")
    parser.add_argument("--n-warmup",     type=int,   default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-enc-iters",  type=int,   default=DEFAULT_N_ENC_ITERS)
    parser.add_argument("--n-e2e-iters",  type=int,   default=DEFAULT_N_E2E_ITERS)
    parser.add_argument("--min-duration", type=float, default=0.0,
        help="Skip samples shorter than this many seconds (e.g. 15.0). "
             "Increase --n-samples to ensure enough long clips are loaded.")
    parser.add_argument("--encoder-only", action="store_true",
        help="Skip end-to-end timing (faster smoke tests).")
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_OUT_DIR,
        help=f"Directory for wallclock.csv + wallclock_summary.csv "
             f"(default: {_DEFAULT_OUT_DIR}).")
    args = parser.parse_args()

    langs   = args.langs if args.langs else DEFAULT_LANGS
    # Canonical ordering keeps aggregation and progress output deterministic.
    raw_trrs = args.trrs if args.trrs else DEFAULT_TRRS
    trrs     = sorted({round(float(t), 6) for t in raw_trrs})
    for t in trrs:
        if not (0.0 < t < 0.5):
            raise ValueError(f"--trrs value {t} outside valid range (0, 0.5); merging module rejects it.")
    if not trrs:
        raise ValueError("--trrs must supply at least one non-zero TRR.")

    out_dir     = args.out_dir
    long_csv    = os.path.join(out_dir, "wallclock.csv")
    summary_csv = os.path.join(out_dir, "wallclock_summary.csv")

    set_deterministic()
    device = get_device()
    env    = log_environment(device)

    print(f"  Languages        : {langs}")
    print(f"  TRRs             : {trrs}")
    print(f"  Samples per lang : {args.n_samples}")
    print(f"  Warmup iters     : {args.n_warmup}")
    print(f"  Enc iters        : {args.n_enc_iters}")
    print(f"  E2E iters        : {args.n_e2e_iters} {'(SKIPPED)' if args.encoder_only else ''}")
    print(f"  max_new_tokens   : {MAX_NEW_TOKENS}")
    print(SEP)

    print(f"\nLoading {MODEL_NAME} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model     = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, revision=MODEL_REVISION,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)
    attn_impl = model.config._attn_implementation
    print(f"  device: {device}")
    print(f"  attn_implementation: {attn_impl} (as loaded — no override)")
    env["attn_implementation"] = attn_impl
    print()

    # Verify attachment and the expected final sequence length at the largest TRR.
    max_trr = trrs[-1]
    max_per_r = per_layer_r_for_trr(max_trr, len(CONFIG_A_LAYERS))
    attach_merging(model,
                   {layer: max_per_r for layer in CONFIG_A_LAYERS},
                   record_stats=False)
    try:
        # Expected final seq_len at TRR=0.40 with 8 merge layers ≈ 900 ± ~30
        expected_lo = int(1500 * (1 - max_trr)) - 30
        expected_hi = int(1500 * (1 - max_trr)) + 30
        assert_merging_attached(model, CONFIG_A_LAYERS,
                                expected_seq_len_range=(expected_lo, expected_hi))
    finally:
        detach_merging(model)
    print(f"  attach/detach + final seq_len assertion PASSED "
          f"(cascade={CONFIG_A_LAYERS}, TRR={max_trr}, "
          f"expected seq_len ∈ [{expected_lo}, {expected_hi}])\n")

    global_gpu_warmup(model, device)

    os.makedirs(out_dir, exist_ok=True)
    long_cols = ["gpu", "batch_size", "torch", "transformers", "attn_impl",
                 "lang_id", "method", "trr",
                 "sample_idx", "temp_c", "dur_s", "n_sil", "n_sph",
                 "n_decoder_tokens", "iter_idx", "metric", "ms"]

    all_results = []
    t_start = time.time()

    # The context manager flushes buffered rows if evaluation raises.
    with open(long_csv, "w", newline="") as long_f:
        long_writer = csv.DictWriter(long_f, fieldnames=long_cols)
        long_writer.writeheader()

        for li, lang_id in enumerate(langs):
            whisper_code = get_whisper_code(lang_id)  # may be None for ig_ng
            print(DSEP)
            print(f"  [{li+1}/{len(langs)}]  {lang_id}  (whisper_code={whisper_code})")
            print(DSEP)

            raw = load_or_cache_fleurs(lang_id, "test", args.n_samples,
                                       DATASET_NAME, DATASET_REVISION)

            for si, sample in enumerate(raw):
                arr = sample["audio_array"]
                sr  = sample["sampling_rate"]
                dur = len(arr) / sr

                if args.min_duration > 0.0 and dur < args.min_duration:
                    print(f"  s{si:02d}  SKIP  dur={dur:.1f}s  (< min {args.min_duration:.1f}s)")
                    continue

                input_features = (processor(arr, sampling_rate=sr, return_tensors="pt")
                                  .input_features.to(device))
                assert input_features.shape[0] == BATCH_SIZE, (
                    f"Batch size violation: expected {BATCH_SIZE}, "
                    f"got {input_features.shape[0]}. Benchmark must run batch_size=1."
                )
                sample_temp  = gpu_temperature_c()
                sil_info     = compute_silence_info(input_features, dur)
                silence_mask = sil_info["authoritative_mask"]
                n_sil        = int(silence_mask.sum())
                n_sph        = int((~silence_mask).sum())

                # Run encoder and end-to-end phases separately and counterbalance each.
                # PHASE 1 — encoder-only, condition order rotated by sample_idx.
                enc_results = measure_sample_encoder(
                    model, input_features, sample_idx=si, trrs=trrs,
                    n_warmup=args.n_warmup, n_enc_iters=args.n_enc_iters,
                )
                # PHASE 2 — e2e (optional), separate condition ordering.
                if not args.encoder_only:
                    e2e_results = measure_sample_e2e(
                        model, input_features, whisper_code, sample_idx=si,
                        trrs=trrs,
                        n_warmup=args.n_warmup, n_e2e_iters=args.n_e2e_iters,
                    )
                else:
                    e2e_results = {}

                # Merge the two phases keyed by (method, trr).
                conditions = [("baseline", 0.0)] + [("merged", t) for t in trrs]
                results = {}
                for key in conditions:
                    entry = {
                        "enc_ms": enc_results.get(key, {}).get("enc_ms", []),
                        "e2e_ms": e2e_results.get(key, {}).get("e2e_ms", []),
                        "n_tok":  e2e_results.get(key, {}).get("n_tok", -1),
                    }
                    results[key] = entry

                # write long-form rows + collect for summary
                for (method, trr), m in results.items():
                    n_tok = m["n_tok"]
                    for k, ms in enumerate(m["enc_ms"]):
                        write_long_row(long_writer, env, lang_id, method, trr,
                                       si, k, "enc", ms, sample_temp,
                                       dur, n_sil, n_sph, n_tok)
                    for k, ms in enumerate(m["e2e_ms"]):
                        write_long_row(long_writer, env, lang_id, method, trr,
                                       si, k, "e2e", ms, sample_temp,
                                       dur, n_sil, n_sph, n_tok)
                    all_results.append({
                        "lang":   lang_id,
                        "method": method,
                        "trr":    trr,
                        "sample": si,
                        "enc_ms": m["enc_ms"],
                        "e2e_ms": m["e2e_ms"],
                    })

                base_enc = float(np.mean(results[("baseline", 0.0)]["enc_ms"]))
                top_enc  = float(np.mean(results[("merged", trrs[-1])]["enc_ms"]))
                enc_msg  = (f"enc base={base_enc:6.2f}ms  "
                            f"merged@{trrs[-1]:.2f}={top_enc:6.2f}ms  "
                            f"(speedup={base_enc/top_enc:.3f}x)")
                if args.encoder_only:
                    e2e_msg = "e2e SKIPPED"
                else:
                    base_e2e = float(np.mean(results[("baseline", 0.0)]["e2e_ms"]))
                    top_e2e  = float(np.mean(results[("merged", trrs[-1])]["e2e_ms"]))
                    base_tok = results[("baseline", 0.0)]["n_tok"]
                    top_tok  = results[("merged", trrs[-1])]["n_tok"]
                    e2e_msg  = (f"e2e base={base_e2e:7.1f}ms(tok={base_tok:3d})  "
                                f"merged@{trrs[-1]:.2f}={top_e2e:7.1f}ms(tok={top_tok:3d})  "
                                f"(speedup={base_e2e/top_e2e:.3f}x)")
                print(f"  s{si:02d}  {sample_temp:2d}°C  dur={dur:5.1f}s  "
                      f"n_sil={n_sil:4d}  n_sph={n_sph:4d}  "
                      f"{enc_msg}  {e2e_msg}", flush=True)

                del input_features
                torch.cuda.empty_cache()

            long_f.flush()

    elapsed_min = (time.time() - t_start) / 60
    print(f"\n  Total elapsed: {elapsed_min:.1f} min")

    aggregate_and_write_summary(all_results, env, summary_csv)
    print(f"  Wrote: {long_csv}")
    print(f"  Wrote: {summary_csv}")
    print(SEP)


if __name__ == "__main__":
    main()
