"""Evaluate token merging across Whisper model scales.

The default run evaluates Whisper-small on the six adapter-training languages
with greedy adjacent K-cosine merging. It records an unmerged baseline and
Config A at each requested TRR in a common WER and hallucination schema.

  • no merging (baseline)
  • Config A (A-ToMe-style every-3-starting-at-L2 cascade) at each TRR

Why the layer set is auto-derived:
  Config A on whisper-medium (24 layers) = [2, 5, 8, 11, 14, 17, 20, 23].
  Whisper-small has 12 encoder layers, so the same "every 3rd from L2" rule
  yields [2, 5, 8, 11] (4 layers). Per-layer ratios scale automatically:
    TRR=0.40 on 8 layers → per_layer_r ≈ 0.062
    TRR=0.40 on 4 layers → per_layer_r ≈ 0.120
  but the *final* encoder length after the cascade is identical (1500 → 900).

The layer set is derived from encoder depth, and the ``model`` column makes
the output directly comparable with the medium, DoRA, and large-v3 results.

Run
---
    # whisper-small + Config A on the 6 FT langs
    CUDA_VISIBLE_DEVICES=1 python -m tmm_asr.eval.cross_scale --tag small

    # Other model variants (auto-derives Config A from layer count)
    python -m tmm_asr.eval.cross_scale \\
        --model openai/whisper-base --tag base
"""

import argparse
import copy
import csv
import os
import random
import time
import warnings

import jiwer
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from tmm_asr.data.languages import LANGUAGES
from tmm_asr.eval.pipeline import (
    COMPRESSION_RATIO_THRESHOLD,
    _compression_ratio,
    _prepare_for_wer,
    _transcribe_with_fallback,
    preload_samples,
)
from tmm_asr.merging import attach_merging, detach_merging, read_seq_lens

warnings.filterwarnings("ignore")

# Defaults
DEFAULT_MODEL = "openai/whisper-small"
# Paper mix6 cohort — the six languages the DoRA adapter was trained on and
# that all §5.3 cross-scale results in the paper are reported over.
LANGS_FT      = ["vi_vn", "ta_in", "jv_id", "mt_mt", "ln_cd", "ha_ng"]
TARGET_TRRS   = [0.05, 0.10, 0.20, 0.30, 0.40]
N_DEFAULT     = 264
RANDOM_SEED   = 42

# Output root — override with TMM_OUT_DIR env var or --out-dir CLI arg.
RESULTS_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(os.getcwd(), "outputs", "eval"),
)
SEP         = "=" * 84

FIELDNAMES = [
    "lang_id", "lang_name", "tonal", "resource", "family", "model",
    "config", "layers", "trr", "per_layer_r", "wer", "wer_delta",
    "halluc_rate", "halluc_n_high_cr", "halluc_n_empty",
    "mean_hyp_words", "mean_ref_words", "seq_len_final", "n_samples",
    "elapsed_s",
]


def _expected_condition_keys(trrs):
    return [("baseline", 0.0), *[("A", float(trr)) for trr in trrs]]


def order_rows(rows, langs, trrs):
    by_key = {
        (row["lang_id"], row["config"], float(row["trr"])): row
        for row in rows
    }
    return [
        by_key[(lang_id, config, trr)]
        for lang_id in langs
        for config, trr in _expected_condition_keys(trrs)
        if (lang_id, config, trr) in by_key
    ]


def load_resume_rows(out_csv, langs, n_samples, trrs, model_short):
    """Validate and retain complete language blocks from a partial sweep."""
    if not os.path.exists(out_csv):
        return [], set()

    with open(out_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"lang_id", "model", "config", "trr", "n_samples", "wer"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"cannot resume {out_csv}: missing required columns {missing}"
            )
        existing = list(reader)

    requested_langs = set(langs)
    expected_conditions = set(_expected_condition_keys(trrs))
    rows_by_lang = {}
    seen_keys = set()
    for row_number, row in enumerate(existing, start=2):
        lang_id = row["lang_id"]
        if lang_id not in requested_langs:
            raise ValueError(
                f"cannot resume {out_csv}: row {row_number} has unrequested "
                f"language {lang_id!r}"
            )
        if row["model"] != model_short:
            raise ValueError(
                f"cannot resume {out_csv}: row {row_number} has model "
                f"{row['model']!r}, expected {model_short!r}"
            )
        try:
            row_n = int(float(row["n_samples"]))
            condition = (row["config"], float(row["trr"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot resume {out_csv}: invalid identity value on row {row_number}"
            ) from exc
        if row_n != n_samples:
            raise ValueError(
                f"cannot resume {out_csv}: row {row_number} has n_samples={row_n}, "
                f"expected {n_samples}"
            )
        if condition not in expected_conditions:
            raise ValueError(
                f"cannot resume {out_csv}: row {row_number} has unexpected "
                f"condition {condition}"
            )
        key = (lang_id, *condition)
        if key in seen_keys:
            raise ValueError(f"cannot resume {out_csv}: duplicate condition {key}")
        seen_keys.add(key)
        rows_by_lang.setdefault(lang_id, []).append(
            {field: row.get(field, "") for field in FIELDNAMES}
        )

    complete_langs = {
        lang_id
        for lang_id, lang_rows in rows_by_lang.items()
        if {(row["config"], float(row["trr"])) for row in lang_rows}
        == expected_conditions
    }
    complete_rows = [
        row
        for lang_id in complete_langs
        for row in rows_by_lang[lang_id]
    ]
    return order_rows(complete_rows, langs, trrs), complete_langs


def write_rows(out_csv, rows, langs, trrs):
    ordered = order_rows(rows, langs, trrs)
    tmp_csv = f"{out_csv}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp_csv, out_csv)
    return ordered


# Config A auto-derivation
def get_config_a_layers(n_encoder_layers: int) -> list:
    """
    A-ToMe-style cascade: every 3rd encoder layer starting at L2.

    Excludes the final encoder layer (index == n_encoder_layers in 1-indexed
    counting). Rationale: the last encoder layer's output goes directly to
    decoder cross-attention with no further encoder processing. Merging there
    leaves zero "buffer" layers to smooth out merge-induced perturbations
    before the decoder consumes the sequence — a more sensitive merge point
    than mid-stack merges. For whisper-small (12 enc layers) and whisper-medium
    (24), the every-3-from-L2 pattern already naturally stops at L11 and L23
    respectively (one layer short of the last). For whisper-large-v3 (32),
    the pattern would land on L32 (the last layer) without this exclusion;
    we cap one layer short to preserve consistent behaviour across model sizes.

    Examples:
      12 layers (whisper-small)    → [2, 5, 8, 11]                        (last enc=L12, buffer=1)
      24 layers (whisper-medium)   → [2, 5, 8, 11, 14, 17, 20, 23]        (last enc=L24, buffer=1)
      32 layers (whisper-large-v3) → [2, 5, 8, 11, 14, 17, 20, 23, 26, 29] (last enc=L32, buffer=3)
       6 layers (whisper-tiny)     → [2, 5]                                (last enc=L6,  buffer=1)
    """
    return list(range(2, n_encoder_layers, 3))


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
    Per-condition hallucination stats based on Whisper's compression-ratio
    proxy (Radford et al. 2022 §3.3). Same definition as the merging-eval
    scripts — CR > 2.4 on the raw hypothesis after the temperature-fallback
    loop terminated.
    """
    n = len(hyps_raw)
    if n == 0:
        return {"halluc_rate": float("nan"),
                "halluc_n_high_cr": 0, "halluc_n_empty": 0,
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
        "mean_hyp_words":   hyp_w_sum / n,
        "mean_ref_words":   ref_w_sum / n,
    }


def run_condition(model, processor, gen_cfg, samples, lang_id, normalizer,
                  merge_layers, trr):
    """
    One decode pass.
    Returns (wer, halluc_stats, seq_lens_dict, elapsed_s).
    """
    if merge_layers and trr > 0:
        r = per_layer_r(trr, len(merge_layers))
        attach_merging(model, {layer: r for layer in merge_layers})
    else:
        r = 0.0

    t0 = time.time()
    try:
        refs_norm, hyps_raw, refs_raw = [], [], []
        last_sls = None
        for s in samples:
            hyp_raw = _transcribe_with_fallback(
                model, processor, gen_cfg,
                input_features=s["feats"], lang_id=lang_id,
                encoder_outputs=None,
            )
            if r > 0:
                last_sls = read_seq_lens(model)
            hyps_raw.append(hyp_raw)
            refs_raw.append(s.get("ref_raw", ""))
            refs_norm.append(s["r_norm"])

        hyps_norm = [_prepare_for_wer(h, lang_id, normalizer) for h in hyps_raw]
        wer = (jiwer.wer(refs_norm, hyps_norm)
               if any(r.strip() for r in refs_norm) else float("nan"))
        halluc = compute_hallucination_stats(refs_raw, hyps_raw)
    finally:
        if r > 0:
            detach_merging(model)

    return wer, halluc, last_sls, time.time() - t0


# Main
def main():
    global RESULTS_DIR
    from tmm_asr.eval.pipeline import (
        LARGE_V3_MODEL_REVISION,
        SMALL_MODEL_REVISION,
    )
    from tmm_asr.eval.pipeline import (
        MODEL_REVISION as MEDIUM_MODEL_REVISION,
    )
    _DEFAULT_REVISIONS = {
        "openai/whisper-small":    SMALL_MODEL_REVISION,
        "openai/whisper-medium":   MEDIUM_MODEL_REVISION,
        "openai/whisper-large-v3": LARGE_V3_MODEL_REVISION,
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="HuggingFace Whisper model name "
                         "(default: openai/whisper-small)")
    ap.add_argument("--revision", type=str, default=None,
                    help="Pinned HF revision. If omitted, resolves to the paper-pinned revision for the given --model.")
    ap.add_argument("--langs", nargs="+", default=LANGS_FT,
                    help="Language list (default: 6 FT langs)")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--trrs", nargs="+", type=float, default=TARGET_TRRS)
    ap.add_argument("--no-merge", action="store_true",
                    help="Skip merging conditions; baseline-only run.")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output CSV filename")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                    help="Directory for output CSVs. Default: $TMM_OUT_DIR or <cwd>/outputs/eval/")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore an existing destination CSV and start over. "
                         "By default, compatible complete language blocks are resumed.")
    args = ap.parse_args()
    # Honour --out-dir (default is env var / repo-root outputs/eval).
    RESULTS_DIR = args.out_dir

    # If --revision omitted, use the paper-pinned revision for this model
    if args.revision is None:
        args.revision = _DEFAULT_REVISIONS.get(args.model)

    seed_everything()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_short = args.model.split("/")[-1]
    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(
        RESULTS_DIR,
        f"whisper_size_baseline_{model_short}_n{args.n}{suffix}.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trrs = sorted(set(args.trrs)) if not args.no_merge else []
    rows = []
    already_done = set()
    if not args.no_resume:
        try:
            rows, already_done = load_resume_rows(
                out_csv, args.langs, args.n, trrs, model_short
            )
        except ValueError as exc:
            ap.error(str(exc))

    print(f"\n{SEP}")
    print("  WHISPER SIZE × MERGING — Config A, greedy adjacent K-cosine")
    print(f"  Model      : {args.model}")
    print(f"  Revision   : {args.revision or 'main (UNPINNED — non-reproducible)'}")
    print(f"  Device     : {device}")
    print(f"  Langs      : {args.langs}")
    print(f"  n_samples  : {args.n}")
    if args.no_merge:
        print("  Mode       : baseline ONLY (--no-merge)")
    else:
        print(f"  TRRs       : {trrs}")
    print(f"  Output     : {out_csv}")
    print(f"{SEP}\n")

    print(f"  Loading {args.model} ...")
    processor = WhisperProcessor.from_pretrained(args.model, revision=args.revision)
    # Explicit torch_dtype=torch.float32 — whisper-large-v3 ships as fp16 by
    # default on HuggingFace, but our processor produces fp32 mel features
    # and the merging code (head-mean K-cosine, L2 norm) is fp32. Forcing
    # fp32 model weights eliminates dtype mismatch and avoids precision
    # drift in the cosine computation.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(device).eval()
    normalizer = BasicTextNormalizer()
    gen_cfg    = copy.deepcopy(model.generation_config)

    # ---- Auto-derive Config A from the model's actual encoder depth ----
    n_enc_layers = len(model.model.encoder.layers)
    config_layers = get_config_a_layers(n_enc_layers)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters   : {n_params/1e6:.1f}M")
    print(f"  Encoder layers     : {n_enc_layers}")
    print(f"  d_model            : {model.config.d_model}")
    print(f"  encoder_attn_heads : {model.config.encoder_attention_heads}")
    print(f"  Config A (auto)    : {config_layers}  ({len(config_layers)} merge layers)")
    if not args.no_merge:
        for t in trrs:
            r = per_layer_r(t, len(config_layers))
            print(f"    TRR={t:.2f}  →  per_layer_r={r:.4f}")
    print()

    if already_done:
        print(
            f"  [RESUME] Preserving {len(already_done)} complete language "
            f"block(s): {sorted(already_done)}\n"
        )
    t_run = time.time()

    for li, lang_id in enumerate(args.langs, start=1):
        if lang_id in already_done:
            print(f"\n{SEP}")
            print(f"  [{li}/{len(args.langs)}] LANG: {lang_id}  [RESUME → SKIP, already in CSV]")
            print(SEP)
            continue
        info = LANGUAGES.get(lang_id, {})
        lang_name    = info.get("name", lang_id)
        tonal        = info.get("tonal", "")
        resource     = info.get("resource", "")
        family       = info.get("family", "")
        whisper_code = info.get("whisper_code", None)

        print(f"\n{SEP}")
        print(f"  [{li}/{len(args.langs)}] LANG: {lang_id}  ({lang_name})  "
              f"tonal={tonal}  resource={resource}  family={family}")
        if whisper_code is None:
            print("  whisper_code=None → empty hyps, WER meaningless")
        print(SEP)

        print(f"  Loading {args.n} samples ...", flush=True)
        samples = preload_samples(lang_id, args.n, processor, device)

        # ---- Baseline (no merging) once per language ----
        print(f"  [{lang_id}] baseline (no merging) ...", flush=True)
        b_wer, b_halluc, _, b_elapsed = run_condition(
            model, processor, gen_cfg, samples, lang_id, normalizer,
            merge_layers=[], trr=0.0)
        print(f"    baseline WER = {b_wer*100:6.2f}%   "
              f"halluc = {b_halluc['halluc_rate']*100:5.2f}%   "
              f"(n_high_cr={b_halluc['halluc_n_high_cr']}/{args.n}, "
              f"n_empty={b_halluc['halluc_n_empty']})   "
              f"({b_elapsed:5.1f}s)", flush=True)

        rows.append({
            "lang_id": lang_id, "lang_name": lang_name,
            "tonal": tonal, "resource": resource, "family": family,
            "model": model_short,
            "config": "baseline", "layers": "[]",
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

        # ---- TRR sweep under Config A (skipped if --no-merge) ----
        for trr in trrs:
            r_layer = per_layer_r(trr, len(config_layers))
            print(f"  [{lang_id}] Config A {config_layers} TRR={trr:.2f}  "
                  f"per_r={r_layer:.4f} ...", flush=True)
            m_wer, m_halluc, sls, elapsed = run_condition(
                model, processor, gen_cfg, samples, lang_id, normalizer,
                merge_layers=config_layers, trr=trr)
            delta        = m_wer - b_wer
            halluc_delta = m_halluc["halluc_rate"] - b_halluc_rate
            sl_final     = sls.get(config_layers[-1], None) if sls else None
            print(f"    TRR={trr:.2f}  WER={m_wer*100:6.2f}%  "
                  f"Δ={delta*100:+6.2f}pp  "
                  f"halluc={m_halluc['halluc_rate']*100:5.2f}% "
                  f"(Δ={halluc_delta*100:+5.2f}pp)  "
                  f"seq@L{config_layers[-1]}={sl_final}  ({elapsed:5.1f}s)",
                  flush=True)
            rows.append({
                "lang_id": lang_id, "lang_name": lang_name,
                "tonal": tonal, "resource": resource, "family": family,
                "model": model_short,
                "config": "A", "layers": str(config_layers),
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

        # Atomic incremental save: a killed process leaves the last complete file.
        rows = write_rows(out_csv, rows, args.langs, trrs)
        print(f"  [{lang_id}] CSV updated → {out_csv}")

    # ---- Final summary ----
    print(f"\n{SEP}\n  SUMMARY — {args.model}  ·  Config A {config_layers}\n"
          f"  WER deltas (pp) vs baseline, n={args.n}\n{SEP}")
    if trrs:
        hdr = "  ".join(f"TRR={t:.2f}".rjust(9) for t in trrs)
        print(f"  {'Lang':>8}  {'Base%':>10}  {'Halluc%':>9}  " + hdr)
        print(f"  {'-'*8:>8}  {'-'*10:>10}  {'-'*9:>9}  "
              + "  ".join("-" * 9 for _ in trrs))

        by_lct = {
            (r["lang_id"], r["config"], float(r["trr"])): r
            for r in rows
        }
        baseline = {r["lang_id"]: r for r in rows if r["config"] == "baseline"}

        for lang in args.langs:
            b_row = baseline.get(lang)
            if not b_row:
                continue
            cells = []
            for t in trrs:
                r = by_lct.get((lang, "A", t))
                d = float(r["wer_delta"]) if r else float("nan")
                cells.append(f"{d*100:+6.2f}pp".rjust(9))
            print(f"  {lang:>8}  {float(b_row['wer'])*100:>9.2f}%  "
                  f"{float(b_row['halluc_rate'])*100:>8.2f}%  "
                  + "  ".join(cells))

        mean_cells = []
        for t in trrs:
            ds = [float(by_lct[(lang_id, "A", t)]["wer_delta"])
                  for lang_id in args.langs if (lang_id, "A", t) in by_lct]
            m = sum(ds) / len(ds) if ds else float("nan")
            mean_cells.append(f"{m*100:+6.2f}pp".rjust(9))
        print(f"  {'MEAN Δ':>8}  {'':>10}  {'':>9}  " + "  ".join(mean_cells))
    else:
        print(f"  {'Lang':>8}  {'Base%':>10}  {'Halluc%':>9}")
        for r in rows:
            print(f"  {r['lang_id']:>8}  {float(r['wer'])*100:>9.2f}%  "
                  f"{float(r['halluc_rate'])*100:>8.2f}%")

    total_min = (time.time() - t_run) / 60
    print(f"\n  Total runtime: {total_min:.1f} min")
    print(f"  Saved CSV : {out_csv}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
