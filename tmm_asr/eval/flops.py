"""Theoretical encoder-FLOPs post-processor for result CSVs.

Rationale:
  The merging code reports `seq_len_final` per (lang, config, trr) but does not
  compute FLOPs. Wall-clock latency on our hardware does not reflect the
  theoretical compute saving (kernel-launch overhead dominates at batch=1
  with no fused merge kernel), so we report theoretical encoder FLOPs as the
  efficiency axis. This is the same metric A-ToMe reports.

What it computes:
  For every row of every active CSV in `results/`:
    1. Reconstruct the per-layer sequence-length trajectory from the columns
       `layers` (which encoder layers merge) and `per_layer_r` (the per-layer
       reduction rate), simulating the actual integer arithmetic in
       ``tmm_asr.merging``:
           n_merge = int((n_pre - 1) * per_layer_r)
           n_post  = n_pre - n_merge
    2. Sanity-check that the reconstructed final length matches the recorded
       `seq_len_final` (tolerance: 0 — must be exact, since the simulation
       mirrors the production arithmetic).
    3. Sum theoretical FLOPs across all encoder layers under the convention
       that merging happens post-MHA-residual / pre-FFN:
           attn uses n_pre, FFN uses n_post
       FLOPs per layer:
           attn = 8 * n_pre * d^2 + 4 * n_pre^2 * d
           ffn  = 4 * n_post * d * d_ff
    4. Reports merged_gflops, baseline_gflops (TRR=0 at same n0=1500), and
       enc_flops_reduction = 1 - merged/baseline.

  Output: sibling files `<name>_with_flops.csv` for every input CSV, plus a
  cross-model summary `theoretical_flops_summary.csv`.

Run:
  python -m tmm_asr.eval.flops
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Frozen paper CSVs ship inside the package (`tmm_asr/paper_results/`) so
# wheel installs work without needing the source tree. This is also the
# default input dir for --results-dir.
_PKG_ROOT = Path(__file__).resolve().parent.parent   # tmm_asr/
RESULTS_DIR = _PKG_ROOT / "paper_results"

# Whisper encoder architecture constants (HuggingFace WhisperConfig defaults).
# Verified against transformers WhisperConfig for each pinned revision.
# n_mels: needed for the conv stem (conv1: n_mels -> d_model, k=3, stride=1;
#         conv2: d_model -> d_model, k=3, stride=2). The stem is invariant
#         under merging but is part of the encoder, so we include it in both
#         numerator and denominator of enc_flops_reduction for honesty
#         (omitting it from both would overstate the reduction by ~0.8-1%).
ARCH = {
    "whisper-small": {
        "d_model": 768, "d_ff": 3072, "n_heads": 12, "n_enc_layers": 12,
        "n_mels": 80,
    },
    "whisper-medium": {
        "d_model": 1024, "d_ff": 4096, "n_heads": 16, "n_enc_layers": 24,
        "n_mels": 80,
    },
    "whisper-large-v3": {
        "d_model": 1280, "d_ff": 5120, "n_heads": 20, "n_enc_layers": 32,
        "n_mels": 128,
    },
}

# All CSVs not in this list are produced by scripts that target whisper-medium
# (the default for fixed_rate_main_rerun, finetune_merge_rerun, layer_ablation).
_FILENAME_HINT_TO_MODEL = {
    "whisper-small":    "whisper-small",
    "whisper-medium":   "whisper-medium",
    "whisper-large-v3": "whisper-large-v3",
}

# Explicit allow-list of medium-only CSVs that have no `model` column and no
# filename hint. Anything outside this set will raise rather than silently
# defaulting to whisper-medium, so a future copy of an old/non-medium CSV
# into results/ can't quietly get the wrong architecture.
_KNOWN_MEDIUM_DEFAULTS = {
    "fixed_rate_main_rerun_cfgA_n264_halfall_single.csv",
    "finetune_merge_rerun_cfgA_n264_checkpoint-2000_gpu1.csv",
    "finetune_merge_rerun_cfgA_n264_checkpoint-2000_mix6.csv",  # released paper file
    "finetune_holdout_cfgA_n264_checkpoint-2000_mix6.csv",       # released paper file
    "layer_ablation_new_merging_cfgA_n264_gpu1.csv",
    "layer_ablation_new_merging_cfgB_n264_gpu0.csv",
}

# Whisper encoder always operates on 1500 mel-frame positions after the two
# stride-1, stride-2 conv stems (independent of n_mels).
N0 = 1500


def detect_model(csv_path: Path, row: pd.Series) -> str:
    """Identify the Whisper variant for a given row."""
    if "model" in row and isinstance(row["model"], str):
        if row["model"] in ARCH:
            return row["model"]
    name = csv_path.name
    for hint, model in _FILENAME_HINT_TO_MODEL.items():
        if hint in name:
            return model
    if name in _KNOWN_MEDIUM_DEFAULTS:
        return "whisper-medium"
    raise ValueError(
        f"Cannot determine Whisper variant for {name}: no `model` column, "
        f"no filename hint, and not in the medium-default allow-list. "
        f"Add an entry to _KNOWN_MEDIUM_DEFAULTS or _FILENAME_HINT_TO_MODEL."
    )



def parse_layers(layers_field) -> list[int]:
    """Parse the stringified-list `layers` column. Baseline rows have []."""
    if pd.isna(layers_field):
        return []
    if isinstance(layers_field, (list, tuple)):
        return list(layers_field)
    s = str(layers_field).strip()
    if s in ("", "[]", "nan"):
        return []
    return list(ast.literal_eval(s))


def simulate_trajectory(
    n0: int,
    n_enc_layers: int,
    merge_layers: Iterable[int],
    per_layer_r: float,
) -> list[tuple[int, int]]:
    """
    Return a list of (n_pre, n_post) per encoder layer, mirroring the integer
    arithmetic in ``tmm_asr.merging``:
        n_pairs = n_pre - 1
        n_merge = int(n_pairs * per_layer_r)
        n_post  = n_pre - n_merge

    Indexing: the CSV `layers` column uses Whisper's 1-based convention (see
    ``tmm_asr.merging.attach_merging``, which uses `layers[layer_1 - 1]`
    when slicing the 0-based encoder ModuleList). We convert to 0-based here
    so the loop variable L matches the merge_set comparison.
    """
    merge_set = {int(L) - 1 for L in merge_layers}
    traj: list[tuple[int, int]] = []
    n = n0
    for L in range(n_enc_layers):
        n_pre = n
        if L in merge_set and per_layer_r > 0 and n_pre >= 2:
            n_merge = int((n_pre - 1) * per_layer_r)
            n_post = n_pre - n_merge
        else:
            n_post = n_pre
        traj.append((n_pre, n_post))
        n = n_post
    return traj


def layer_flops(n_pre: int, n_post: int, d: int, d_ff: int) -> float:
    """
    Theoretical FLOPs for one encoder layer, intra-block merge convention:
      attention runs at n_pre, FFN runs at n_post.

    Components (multiply-add counted as 2 FLOPs):
      Q,K,V projections:      6 * n_pre * d^2
      Attention output proj:  2 * n_pre * d^2
      QK^T:                   2 * n_pre^2 * d
      softmax(.) * V:         2 * n_pre^2 * d
      FFN up   (d -> d_ff):   2 * n_post * d * d_ff
      FFN down (d_ff -> d):   2 * n_post * d_ff * d

    Omitted (standard convention, e.g. ToMe / A-ToMe / fvcore):
      LayerNorms (4 * n * d total per layer) — < 0.1% of layer FLOPs at
      our d / d_ff. Documented as an explicit convention.
    """
    attn = 8.0 * n_pre * (d ** 2) + 4.0 * (n_pre ** 2) * d
    ffn  = 4.0 * n_post * d * d_ff
    return attn + ffn


def conv_stem_flops(arch: dict) -> float:
    """
    Whisper conv stem FLOPs (fixed cost, invariant under merging).
      conv1: n_mels -> d_model, kernel=3, stride=1, L_out = 3000
      conv2: d_model -> d_model, kernel=3, stride=2, L_out = 1500
    Each conv: 2 * L_out * C_out * (kernel * C_in)
    Included in both numerator and denominator of enc_flops_reduction so the
    ratio is honest (the conv stem is part of the encoder).
    """
    n_mels = arch["n_mels"]
    d = arch["d_model"]
    conv1 = 2.0 * 3000 * d * (3 * n_mels)
    conv2 = 2.0 * 1500 * d * (3 * d)
    return conv1 + conv2


def encoder_flops(traj: list[tuple[int, int]], arch: dict) -> float:
    d, d_ff = arch["d_model"], arch["d_ff"]
    transformer = sum(layer_flops(np_, npo, d, d_ff) for np_, npo in traj)
    return conv_stem_flops(arch) + transformer


def baseline_flops(arch: dict) -> float:
    """FLOPs at TRR=0 (no merging) — every layer at n0 throughout."""
    traj = [(N0, N0)] * arch["n_enc_layers"]
    return encoder_flops(traj, arch)


def process_csv(csv_path: Path, out_dir: Path) -> dict | None:
    df = pd.read_csv(csv_path)
    if not {"layers", "per_layer_r", "seq_len_final"}.issubset(df.columns):
        return None

    enriched_rows = []
    mismatches = 0
    for _, row in df.iterrows():
        model = detect_model(csv_path, row)
        arch = ARCH[model]
        merge_layers = parse_layers(row["layers"])
        per_layer_r = float(row["per_layer_r"]) if not pd.isna(row["per_layer_r"]) else 0.0

        traj = simulate_trajectory(N0, arch["n_enc_layers"], merge_layers, per_layer_r)
        reconstructed_final = traj[-1][1]
        # Empirically no NaN in any existing CSV (all dtype=int64), but guard
        # anyway so the script doesn't crash on future runs that hit edge cases.
        if pd.isna(row["seq_len_final"]):
            recorded_final = reconstructed_final  # no recorded value to compare
        else:
            recorded_final = int(row["seq_len_final"])
        if reconstructed_final != recorded_final:
            mismatches += 1

        merged = encoder_flops(traj, arch)
        baseline = baseline_flops(arch)
        new_row = row.to_dict()
        new_row["enc_flops_baseline_gflops"] = baseline / 1e9
        new_row["enc_flops_merged_gflops"] = merged / 1e9
        new_row["enc_flops_reduction"] = 1.0 - merged / baseline
        new_row["seq_len_final_reconstructed"] = reconstructed_final
        enriched_rows.append(new_row)

    out = pd.DataFrame(enriched_rows)
    out_path = out_dir / (csv_path.stem + "_with_flops.csv")
    out.to_csv(out_path, index=False)
    return {
        "path": csv_path.name,
        "rows": len(df),
        "mismatches": mismatches,
        "out": out_path.name,
    }


def summarise(out_paths: list[Path]) -> pd.DataFrame:
    """
    Build the cross-model summary table at TRR=0.40 (functional-5 mean
    encoder-FLOP reduction).
    """
    rows = []
    for p in out_paths:
        df = pd.read_csv(p)
        if "trr" not in df.columns:
            continue
        sub = df[df["trr"].round(2) == 0.40]
        if sub.empty:
            continue
        # If the CSV has a `lang_id` column, restrict to the paper's mix6
        # trained cohort where present so the cross-model summary matches
        # the trained-6 numbers cited in §5.2.
        trained_6 = {"vi_vn", "ta_in", "jv_id", "mt_mt", "ln_cd", "ha_ng"}
        if "lang_id" in sub.columns:
            sub_f = sub[sub["lang_id"].isin(trained_6)]
            if not sub_f.empty:
                sub = sub_f
        rows.append({
            "source_csv": p.name,
            "n_rows_at_trr_0.40": len(sub),
            "mean_enc_flops_reduction": sub["enc_flops_reduction"].mean(),
            "mean_baseline_gflops": sub["enc_flops_baseline_gflops"].mean(),
            "mean_merged_gflops": sub["enc_flops_merged_gflops"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help=f"Directory of input WER CSVs (default: {RESULTS_DIR})")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs") / "flops",
                    help="Directory for the *_with_flops.csv sidecars and, "
                         "unless --summary-out is given, the cross-model "
                         "summary (default: outputs/flops/). Does not "
                         "modify --results-dir. Avoid pointing this at the "
                         "packaged frozen-results directory — the wheel "
                         "build will then ship the sidecars as if they were "
                         "paper artifacts.")
    ap.add_argument("--summary-out", type=Path, default=None,
                    help="Explicit path (or directory) for the frozen "
                         "`theoretical_flops_summary.csv`. Use this to "
                         "refresh the packaged summary WITHOUT polluting "
                         "tmm_asr/paper_results/ with the eight sidecars: "
                         "e.g. `--summary-out tmm_asr/paper_results/"
                         "theoretical_flops_summary.csv`.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the summary target: default = inside --out-dir; explicit --summary-out
    # may be a path OR a directory (in which case we append the canonical filename).
    if args.summary_out is None:
        summary_path = args.out_dir / "theoretical_flops_summary.csv"
    else:
        summary_path = (
            args.summary_out / "theoretical_flops_summary.csv"
            if args.summary_out.is_dir() else args.summary_out
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    targets = sorted(
        p for p in args.results_dir.glob("*.csv")
        if "_with_flops" not in p.name
        and "composition_table" not in p.name
        and "atome_descriptive" not in p.name
        and "theoretical_flops_summary" not in p.name
        and "wallclock" not in p.name
        and not p.name.startswith("stale_pre_fix")
    )

    print(f"Processing {len(targets)} CSV(s) from {args.results_dir}")
    print(f"Writing *_with_flops.csv sidecars to {args.out_dir}\n")
    summary_rows = []
    out_paths = []
    for csv_path in targets:
        result = process_csv(csv_path, args.out_dir)
        if result is None:
            print(f"  [skip] {csv_path.name}  (missing required columns)")
            continue
        flag = "  OK" if result["mismatches"] == 0 else f"  WARN: {result['mismatches']} mismatch(es)"
        print(f"  {flag}  {result['path']}  ({result['rows']} rows)  -> {result['out']}")
        summary_rows.append(result)
        out_paths.append(args.out_dir / result["out"])

    print("\nCross-model summary (TRR=0.40, paper mix6 trained cohort where applicable):")
    summary = summarise(out_paths)
    if not summary.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary.to_string(index=False))
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote: {summary_path}")
    else:
        print("  (no rows at TRR=0.40 found)")

    n_mismatches = sum(r["mismatches"] for r in summary_rows)
    if n_mismatches:
        print(f"\nWARNING: {n_mismatches} row(s) had trajectory mismatches "
              "vs recorded seq_len_final — inspect *_with_flops.csv "
              "(seq_len_final vs seq_len_final_reconstructed).")
        sys.exit(1)


if __name__ == "__main__":
    main()
