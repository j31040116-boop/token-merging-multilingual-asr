"""Evaluate the DoRA Whisper-medium adapter on held-out FLEURS languages.

The evaluation uses Config A at TRR 0 through 0.40 on six untrained
low/mid-resource languages and four high-resource anchors. Model, adapter,
and dataset revisions are pinned, and aggregate and per-sample CSVs are
written incrementally. See ``docs/REPRODUCE.md`` for the exact command,
cohort, expected runtime, and reproducibility limits.
"""

from __future__ import annotations

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

# Held-out evaluation cohort for the mix6 FT run: 6 untrained low/mid-resource
# languages + 4 high-resource anchors = 10 languages total. Order: H first,
# then M, then L, then VL. Mandarin (cmn_hans_cn) is excluded as in the main
# paper sweep because its baseline WER is dominated by a tokenization artifact.
LANGS_HOLDOUT = [
    # --- High-resource anchors (4) ---
    "en_us", "fr_fr", "de_de", "es_419",
    # --- Untrained low/mid-resource (6): Thai, Swahili, Afrikaans, Icelandic,
    # ---   Welsh, Kazakh
    "th_th",        # Medium, tonal
    "sw_ke",        # Medium, non-tonal
    "af_za",        # Low,    non-tonal
    "is_is",        # Low,    non-tonal
    "cy_gb",        # Low,    non-tonal
    "kk_kz",        # VL,     non-tonal
]

# FT-cohort languages (mix6) — sanity-check guard so this script never
# accidentally re-evaluates a trained language. Update this set whenever
# FINETUNE_LANGS in tmm_asr.train.dora is changed.
LANGS_FT = frozenset({"vi_vn", "ha_ng", "ln_cd", "ta_in", "mt_mt", "jv_id"})

# Output root — override with TMM_OUT_DIR env var or --out-dir CLI arg.
RESULTS_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(os.getcwd(), "outputs", "eval"),
)
SEP         = "=" * 84


# Helpers
def per_layer_r(trr: float, n_layers: int) -> float:
    """Per-merge-layer rate that compounds to a global TRR over n_layers steps."""
    if trr <= 0:
        return 0.0
    return 1.0 - (1.0 - trr) ** (1.0 / n_layers)


def seed_everything(seed: int = RANDOM_SEED) -> None:
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
    proxy. CR > 2.4 is the same threshold _transcribe_with_fallback uses for
    temperature retries — values above threshold survived all retries,
    indicating residual hallucination/repetition.
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
    One decode pass over the preloaded samples for a single (lang, TRR) cell.

    Returns
    -------
    wer            : float  – jiwer.wer over normalised refs/hyps for this cell
    refs_raw       : list   – pre-normalisation reference strings
    refs_norm      : list   – post-normalisation references (what jiwer sees)
    hyps_raw       : list   – raw decoder outputs (used for CR and per-sample CSV)
    hyps_norm      : list   – normalised hypotheses
    halluc_stats   : dict   – per-condition aggregate halluc stats
    seq_lens       : dict   – {layer_idx: post-merge sequence length} or {}
    elapsed_s      : float
    """
    if merge_layers and trr > 0:
        r = per_layer_r(trr, len(merge_layers))
        attach_merging(model, {layer: r for layer in merge_layers})
    else:
        r = 0.0

    t0 = time.time()
    refs_norm, hyps_norm, hyps_raw, refs_raw = [], [], [], []
    last_sls = {}
    try:
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
        halluc = compute_hallucination_stats(refs_raw, hyps_raw)
    finally:
        if r > 0:
            detach_merging(model)

    return (wer, refs_raw, refs_norm, hyps_raw, hyps_norm, halluc,
            last_sls, time.time() - t0)


def load_finetuned_model(checkpoint_dir: str, device: torch.device):
    """Load and merge the fp32 DoRA adapter into Whisper-medium on-device."""
    # Resolve BEFORE loading the base model so a bad --checkpoint fails
    # instantly instead of wasting the several-second Whisper base load.
    try:
        resolved = resolve_dora_checkpoint(checkpoint_dir, DORA_REVISION)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"DoRA checkpoint resolution failed: {e}")

    print(f"  Loading base Whisper from {MODEL_NAME} "
          f"(revision={MODEL_REVISION[:8]}) ...")
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
    print(f"  merge_and_unload() on {device} — baking DoRA deltas into base ...")
    model = model.merge_and_unload()
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


# CSV writers
AGG_FIELDS = [
    "lang_id", "lang_name", "tonal", "resource", "family",
    "checkpoint", "config", "layers", "trr", "per_layer_r",
    "wer", "wer_delta",
    "halluc_rate", "halluc_n_high_cr", "halluc_n_empty",
    "mean_hyp_words", "mean_ref_words",
    "seq_len_final", "n_samples", "elapsed_s",
]

PS_FIELDS = [
    "lang_id", "trr", "sample_idx", "sample_id", "audio_duration_s",
    "reference_raw", "reference_norm",
    "hypothesis_raw", "hypothesis_norm",
    "wer_sample", "compression_ratio", "high_cr_flag", "empty_hyp_flag",
]


def write_csv(path: str, fields: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def per_sample_rows(lang_id, trr, samples, refs_raw, refs_norm,
                    hyps_raw, hyps_norm):
    """One row per sample for this (lang, TRR) cell.

    Note on ``wer_sample``: this is per-utterance WER — edit distance for that
    single (ref, hyp) pair divided by the reference word count of that pair.
    It can exceed 1.0 on short utterances and **is not directly comparable to
    the corpus-level ``wer`` in the aggregate CSV**, which is computed as
    ``jiwer.wer(all_refs, all_hyps)`` (Σ edits / Σ ref-words across the whole
    cohort, length-weighted). Averaging ``wer_sample`` across rows will NOT
    reproduce the aggregate ``wer``. The per-sample column exists for
    per-utterance error analysis (outlier inspection, CR-vs-WER correlation,
    etc.), not for aggregation. Column name kept as ``wer_sample`` to match
    the existing convention in
    a compatible per-sample baseline CSV.
    """
    out = []
    for i, (s, r_raw, r_norm, h_raw, h_norm) in enumerate(
            zip(samples, refs_raw, refs_norm, hyps_raw, hyps_norm)):
        try:
            wer_s = jiwer.wer([r_norm], [h_norm]) if r_norm.strip() else float("nan")
        except Exception:
            wer_s = float("nan")
        cr = _compression_ratio(h_raw) if h_raw.strip() else float("nan")
        out.append({
            "lang_id":          lang_id,
            "trr":              trr,
            "sample_idx":       i,
            "sample_id":        s.get("sid", ""),
            "audio_duration_s": round(s.get("dur", 0.0), 4),
            "reference_raw":    r_raw,
            "reference_norm":   r_norm,
            "hypothesis_raw":   h_raw,
            "hypothesis_norm":  h_norm,
            "wer_sample":       (round(wer_s, 6)
                                  if not np.isnan(wer_s) else "nan"),
            "compression_ratio": (round(cr, 4)
                                   if not np.isnan(cr) else "nan"),
            "high_cr_flag":     (1 if (not np.isnan(cr)
                                       and cr > COMPRESSION_RATIO_THRESHOLD)
                                   else 0),
            "empty_hyp_flag":   (0 if h_raw.strip() else 1),
        })
    return out


# Main
def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS_HOLDOUT,
                    help="Held-out languages to evaluate (default: 10 — "
                         "4 high-resource anchors + 6 untrained low/mid-"
                         "resource languages, mix6 cohort)")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--trrs", nargs="+", type=float, default=TARGET_TRRS)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                    help="DoRA checkpoint dir")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix on output filenames.")
    ap.add_argument("--skip-per-sample", action="store_true",
                    help="Skip writing the per-sample CSV (debug only).")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                    help="Directory for output CSVs. Default: $TMM_OUT_DIR or <cwd>/outputs/eval/")
    args = ap.parse_args()
    # Honour --out-dir (default is env var / repo-root outputs/eval).
    RESULTS_DIR = args.out_dir

    # Keep trained languages in the dedicated adapter-cohort evaluation.
    overlap = [lang_id for lang_id in args.langs if lang_id in LANGS_FT]
    if overlap:
        sys.exit(f"Refusing to run: requested languages overlap with FT cohort "
                 f"{sorted(LANGS_FT)} → {overlap}. Use "
                 f"`python -m tmm_asr.eval.ft_merge` for those.")

    langs = args.langs
    trrs  = sorted(set(args.trrs))
    seed_everything()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix     = f"_{args.tag}" if args.tag else ""
    ckpt_short = os.path.basename(os.path.normpath(args.checkpoint))
    agg_csv = os.path.join(
        RESULTS_DIR,
        f"finetune_holdout_cfg{CONFIG_NAME}_n{args.n}_{ckpt_short}{suffix}.csv"
    )
    ps_csv = os.path.join(
        RESULTS_DIR,
        f"finetune_holdout_cfg{CONFIG_NAME}_n{args.n}_{ckpt_short}{suffix}"
        f"_per_sample.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print("  HELD-OUT FT EVAL — DoRA-medium on 16 paper languages NOT in")
    print("                     the fine-tuning cohort")
    print(f"  Config {CONFIG_NAME}: layers={CONFIG_LAYERS}")
    print(f"  Device     : {device}  ·  Base: {MODEL_NAME}")
    print(f"  Base rev   : {MODEL_REVISION}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Held-out   : {langs}  (n={len(langs)})")
    print(f"  TRRs       : {trrs}")
    for t in trrs:
        r = per_layer_r(t, len(CONFIG_LAYERS))
        print(f"    TRR={t:.2f}  →  per_layer_r={r:.4f}")
    print(f"  Agg CSV    : {agg_csv}")
    print(f"  PS  CSV    : {ps_csv}")
    print(f"{SEP}\n")

    processor  = WhisperProcessor.from_pretrained(MODEL_NAME,
                                                  revision=MODEL_REVISION)
    model      = load_finetuned_model(args.checkpoint, device)
    normalizer = BasicTextNormalizer()
    gen_cfg    = copy.deepcopy(model.generation_config)

    agg_rows = []
    ps_rows  = []
    t_run    = time.time()

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

        # --- baseline (no merging) once per language ---
        print(f"  [{lang_id}] FT baseline (no merging) ...", flush=True)
        (b_wer, b_refs_raw, b_refs_norm, b_hyps_raw, b_hyps_norm,
         b_halluc, _, b_elapsed) = run_condition(
            model, processor, gen_cfg, samples, lang_id, normalizer,
            merge_layers=[], trr=0.0,
        )
        print(f"    FT baseline WER = {b_wer*100:6.2f}%   "
              f"halluc_rate = {b_halluc['halluc_rate']*100:5.2f}%   "
              f"(n_high_cr={b_halluc['halluc_n_high_cr']}/"
              f"{b_halluc['halluc_n_total']}, "
              f"n_empty={b_halluc['halluc_n_empty']})   "
              f"({b_elapsed:5.1f}s)", flush=True)

        agg_rows.append({
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
            "seq_len_final":    1500, "n_samples": args.n,
            "elapsed_s":        round(b_elapsed, 1),
        })
        if not args.skip_per_sample:
            ps_rows.extend(per_sample_rows(
                lang_id, 0.0, samples,
                b_refs_raw, b_refs_norm, b_hyps_raw, b_hyps_norm))

        b_halluc_rate = b_halluc["halluc_rate"]

        # --- TRR sweep under Config A ---
        for trr in trrs:
            r_layer = per_layer_r(trr, len(CONFIG_LAYERS))
            print(f"  [{lang_id}] Config {CONFIG_NAME} TRR={trr:.2f}  "
                  f"per_r={r_layer:.4f} ...", flush=True)
            (m_wer, m_refs_raw, m_refs_norm, m_hyps_raw, m_hyps_norm,
             m_halluc, sls, elapsed) = run_condition(
                model, processor, gen_cfg, samples, lang_id, normalizer,
                merge_layers=CONFIG_LAYERS, trr=trr,
            )
            delta        = m_wer - b_wer
            halluc_delta = m_halluc["halluc_rate"] - b_halluc_rate
            sl_final     = sls.get(CONFIG_LAYERS[-1], None) if sls else None
            print(f"    TRR={trr:.2f}  WER={m_wer*100:6.2f}%  "
                  f"Δ={delta*100:+6.2f}pp  "
                  f"halluc={m_halluc['halluc_rate']*100:5.2f}% "
                  f"(Δ={halluc_delta*100:+5.2f}pp)  "
                  f"seq@L{CONFIG_LAYERS[-1]}={sl_final}  ({elapsed:5.1f}s)",
                  flush=True)

            agg_rows.append({
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
                "seq_len_final":    sl_final, "n_samples": args.n,
                "elapsed_s":        round(elapsed, 1),
            })
            if not args.skip_per_sample:
                ps_rows.extend(per_sample_rows(
                    lang_id, trr, samples,
                    m_refs_raw, m_refs_norm, m_hyps_raw, m_hyps_norm))

        # Incremental save after each language so a kill mid-run leaves a
        # usable partial CSV.
        write_csv(agg_csv, AGG_FIELDS, agg_rows)
        if not args.skip_per_sample:
            write_csv(ps_csv, PS_FIELDS, ps_rows)
        print(f"  [{lang_id}] Agg CSV updated → {agg_csv}")
        if not args.skip_per_sample:
            print(f"  [{lang_id}] PS  CSV updated → {ps_csv}")

    # --- Final summary ---
    print(f"\n{SEP}\n  SUMMARY — Held-out FT eval, ckpt={ckpt_short}, "
          f"Config {CONFIG_NAME} {CONFIG_LAYERS}\n  WER deltas (pp) vs "
          f"FT-baseline, n={args.n}\n{SEP}")
    hdr = "  ".join(f"TRR={t:.2f}".rjust(9) for t in trrs)
    print(f"  {'Lang':>8}  {'FT-base%':>10}  " + hdr)
    print(f"  {'-'*8:>8}  {'-'*10:>10}  " + "  ".join("-" * 9 for _ in trrs))

    by_lct   = {(r["lang_id"], r["config"], r["trr"]): r for r in agg_rows}
    baseline = {r["lang_id"]: r["wer"]
                for r in agg_rows if r["config"] == "baseline_ft"}

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
    print(f"  Agg CSV : {agg_csv}")
    if not args.skip_per_sample:
        print(f"  PS  CSV : {ps_csv}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
