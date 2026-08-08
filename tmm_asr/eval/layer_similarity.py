"""Measure baseline adjacent-key cosine similarity across Whisper layers.

The command supports the pinned small, medium, and large-v3 checkpoints. It
uses the same key projection and cosine calculation as token merging, records
every encoder layer without merging, and writes one row per language and
layer for the layer-similarity figure.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import warnings

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from tmm_asr.data.fleurs import load_or_cache_fleurs
from tmm_asr.merging import attach_merging, detach_merging, read_cosines

warnings.filterwarnings("ignore")

# Defaults
DEFAULT_MODEL    = "openai/whisper-medium"
DEFAULT_REVISION = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"

DATASET_NAME     = "google/fleurs"
DATASET_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"

RANDOM_SEED = 42

# Shared 11-language stratification for cross-model comparison.
LANG_GROUPS_DEFAULT = {
    "tonal_high":     ["cmn_hans_cn"],
    "tonal_mid":      ["vi_vn", "th_th"],
    "tonal_low":      ["yo_ng", "ha_ng", "ln_cd"],
    "nontonal_high":  ["en_us", "es_419", "de_de"],
    "nontonal_mid":   ["sw_ke"],
    "nontonal_low":   ["am_et"],
}

# Output root — override with TMM_OUT_DIR env var or --out-dir CLI arg.
RESULTS_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(os.getcwd(), "outputs", "eval"),
)
SEP = "=" * 80


def short_name(model: str) -> str:
    """openai/whisper-medium -> whisper-medium"""
    return model.split("/")[-1]


def load_samples(lang_id, n_samples, processor, device):
    raw = load_or_cache_fleurs(
        lang_id, "test", n_samples, DATASET_NAME, DATASET_REVISION
    )
    samples = []
    for s in raw:
        feats = (processor(s["audio_array"], sampling_rate=s["sampling_rate"],
                           return_tensors="pt")
                 .input_features.to(device))
        samples.append({"feats": feats, "id": s["id"]})
    return samples


def build_baseline_spec(n_layers: int) -> dict:
    """Record cosine statistics at every layer without merging."""
    return {layer: ("none", 0.0) for layer in range(1, n_layers + 1)}


def run_baseline(model, samples, n_layers: int):
    """Return {layer_1based -> mean adjacent K-cosine across samples}."""
    spec = build_baseline_spec(n_layers)
    accum = {layer: [] for layer in range(1, n_layers + 1)}
    for s in samples:
        attach_merging(model, spec)
        try:
            with torch.no_grad():
                _ = model.model.encoder(s["feats"])
            stats = read_cosines(model)
        finally:
            detach_merging(model)
        for layer, stats_for_layer in stats.items():
            accum[layer].append(stats_for_layer["cos_mean"])
    return {
        layer: float(np.nanmean(values))
        for layer, values in accum.items()
        if values
    }


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    ap.add_argument("--revision", type=str, default=None,
                    help="HF revision; if --model is the default, uses the "
                         "pinned default revision; otherwise leaves unpinned "
                         "unless explicitly passed.")
    ap.add_argument("--n",        type=int, default=50,
                    help="Samples per language (default 50, matches descriptive run)")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="Override the default 11-lang stratification.")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                    help="Directory for output CSVs. Default: $TMM_OUT_DIR or <cwd>/outputs/eval/")
    args = ap.parse_args()
    # Honour --out-dir (default is env var / repo-root outputs/eval).
    RESULTS_DIR = args.out_dir

    # Resolve --revision to the paper-pinned value for any of the 3 model scales.
    from tmm_asr.eval.pipeline import (
        LARGE_V3_MODEL_REVISION,
        SMALL_MODEL_REVISION,
    )
    from tmm_asr.eval.pipeline import (
        MODEL_REVISION as MEDIUM_MODEL_REVISION,
    )
    _pins = {
        "openai/whisper-small":    SMALL_MODEL_REVISION,
        "openai/whisper-medium":   MEDIUM_MODEL_REVISION,
        "openai/whisper-large-v3": LARGE_V3_MODEL_REVISION,
    }
    if args.revision is None:
        args.revision = _pins.get(args.model, DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(
        RESULTS_DIR, f"layer_similarity_{short_name(args.model)}_n{args.n}.csv"
    )

    langs = args.langs or sorted(
        {lang_id for lang_ids in LANG_GROUPS_DEFAULT.values() for lang_id in lang_ids}
    )
    group_of = {
        lang_id: group
        for group, lang_ids in LANG_GROUPS_DEFAULT.items()
        for lang_id in lang_ids
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print("  Layer-similarity sweep — baseline (no merging)")
    print(f"  Model      : {args.model}")
    print(f"  Revision   : {args.revision or 'main (UNPINNED — non-reproducible)'}")
    print(f"  Device     : {device}")
    print(f"  n_samples  : {args.n}/lang")
    print(f"  Languages  : {langs}")
    print(f"  Output     : {out_csv}")
    print(f"{SEP}\n")

    processor = WhisperProcessor.from_pretrained(
        args.model, revision=args.revision
    )
    # fp32 explicit — required for whisper-large-v3 (its default fp16 mismatches
    # the fp32 mel features the processor produces). No-op on the smaller models.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32
    ).to(device).eval()
    n_layers = model.config.encoder_layers
    print(f"  Encoder layers detected: {n_layers}\n")

    rows = []
    for lang_id in langs:
        print(f"  [{lang_id}] loading {args.n} samples ...", flush=True)
        samples = load_samples(lang_id, args.n, processor, device)
        cos_by_layer = run_baseline(model, samples, n_layers)
        for L in range(1, n_layers + 1):
            rows.append({
                "model":     short_name(args.model),
                "lang_id":   lang_id,
                "group":     group_of.get(lang_id, "?"),
                "layer":     L,
                "n_layers":  n_layers,
                "cos_mean":  round(cos_by_layer.get(L, float("nan")), 6),
                "n_samples": args.n,
            })
        # quick per-lang summary
        peak_layer = max(range(1, n_layers + 1),
                         key=lambda L: cos_by_layer.get(L, float("-inf")))
        print(f"    peak similarity at layer {peak_layer} "
              f"(cos = {cos_by_layer[peak_layer]:.4f})")

    fieldnames = ["model", "lang_id", "group", "layer", "n_layers",
                  "cos_mean", "n_samples"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {out_csv}")


if __name__ == "__main__":
    main()
