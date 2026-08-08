"""Whisper-medium fixed-rate WER sweep for paper §5.1.

Runs an unmerged baseline and Config A greedy adjacent K-cosine merging at
each requested TRR. See ``docs/REPRODUCE.md`` for the frozen 18-language
cohort and the one-command single-GPU launcher.
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
# Output root — override with TMM_OUT_DIR env var or --out-dir CLI arg.
RESULTS_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(os.getcwd(), "outputs", "eval"),
)
SEP            = "=" * 84

# Paper §5.1 12-language main sweep: mid + low + very-low resource langs on
# whisper-medium. Excludes the six languages dropped from the paper for having
# baseline WER >= 100% on whisper-medium (yo_ng, uz_uz, pa_in, so_so, am_et,
# sn_zw). To evaluate those anyway, pass --langs ... explicitly.
LANGS_ALL = [
    "af_za", "cy_gb", "ha_ng",
    "is_is", "jv_id", "kk_kz", "ln_cd", "mt_mt",
    "sw_ke", "ta_in", "th_th", "vi_vn",
]
HALVES = {
    "A": LANGS_ALL[:6],    # af_za .. kk_kz  (6 langs)
    "B": LANGS_ALL[6:],    # ln_cd .. vi_vn  (6 langs)
}

FIELDNAMES = [
    "lang_id", "lang_name", "tonal", "resource", "family",
    "config", "layers", "trr", "per_layer_r", "wer", "wer_delta",
    "halluc_rate", "halluc_n_high_cr", "halluc_n_empty",
    "mean_hyp_words", "mean_ref_words", "seq_len_final", "n_samples",
    "elapsed_s",
]


def resolve_langs_and_label(explicit_langs, lang_half, half_label_override):
    """Pure helper: decide (langs, half_label) from the three CLI inputs.

    - explicit_langs != None  -> ('custom', explicit list)
    - lang_half != None       -> (lang_half, HALVES[lang_half])
    - else                    -> ('plotted', LANGS_ALL)
    A non-None `half_label_override` replaces the auto-derived label after
    the langs are resolved. Extracted so tests can exercise the branching
    directly instead of the argparse+eval-loop monolith.
    """
    if explicit_langs:
        langs = list(explicit_langs)
        half_label = "custom"
    elif lang_half:
        langs = HALVES[lang_half]
        half_label = lang_half
    else:
        langs = LANGS_ALL
        half_label = "plotted"
    if half_label_override is not None:
        half_label = half_label_override
    return langs, half_label


def output_filename(n: int, half_label: str, tag: str) -> str:
    """Pure helper: assemble the paper-canonical filename that
    fig2_lowres.py reads. Isolated so a unit test can assert the exact
    string without touching argparse or eval work."""
    suffix = f"_{tag}" if tag else ""
    return f"fixed_rate_main_rerun_cfg{CONFIG_NAME}_n{n}_half{half_label}{suffix}.csv"


def _expected_condition_keys(trrs):
    return [("baseline", 0.0), *[(CONFIG_NAME, float(trr)) for trr in trrs]]


def order_rows(rows, langs, trrs):
    """Return completed rows in the canonical CLI language/condition order."""
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


def load_resume_rows(out_csv, langs, n_samples, trrs):
    """Load only complete language blocks from a compatible partial CSV.

    A crash can leave a valid prefix of completed languages. We preserve those
    blocks, discard any incomplete block, and reject rows from a different run
    rather than silently combining experiments that happen to share a filename.
    """
    if not os.path.exists(out_csv):
        return [], set()

    with open(out_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"lang_id", "config", "trr", "n_samples", "wer", "wer_delta"}
        missing = sorted(required - fieldnames)
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
        # Project onto the current schema. This permits resuming an older CSV
        # that lacks newly added diagnostic columns without preserving unknown
        # future columns that this version cannot interpret.
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
    """Atomically replace the incremental CSV in canonical row order."""
    ordered = order_rows(rows, langs, trrs)
    tmp_csv = f"{out_csv}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp_csv, out_csv)
    return ordered


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
    proxy (Radford et al. 2022 §3.3). CR > 2.4 indicates repetitive output that
    survived all temperature retries — same threshold _transcribe_with_fallback
    uses to decide whether to escalate temperature.
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
    One (lang, condition) decode pass.
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
        halluc = compute_hallucination_stats(refs_raw, hyps_raw)
    finally:
        if r > 0:
            detach_merging(model)

    return wer, hyps_raw, refs_norm, hyps_norm, halluc, last_sls, time.time() - t0


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="Explicit lang list (overrides --lang-half).")
    ap.add_argument("--lang-half", choices=list(HALVES.keys()), default=None,
                    help="A = first 6 langs (af_za .. kk_kz), "
                         "B = last 6 langs (ln_cd .. vi_vn). "
                         "Default: all 12 paper main-sweep langs.")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--trrs", nargs="+", type=float, default=TARGET_TRRS)
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output CSV filename.")
    ap.add_argument("--half-label", type=str, default=None,
                    help="Override the halfXXX token in the output filename "
                         "(default: derived from --lang-half / --langs). Use "
                         "`--half-label all` together with the documented "
                         "18-language list to produce the frozen artifact's "
                         "canonical filename and row order.")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                    help="Directory for output CSVs. Default: $TMM_OUT_DIR or <cwd>/outputs/eval/")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore an existing destination CSV and start over. "
                         "By default, compatible complete language blocks are resumed.")
    args = ap.parse_args()
    # Honour --out-dir (default is env var / repo-root outputs/eval).
    RESULTS_DIR = args.out_dir

    # Resolve language list + filename label (pure helper — see
    # resolve_langs_and_label / output_filename for the branch table).
    langs, half_label = resolve_langs_and_label(
        args.langs, args.lang_half, args.half_label)

    trrs = sorted(set(args.trrs))
    seed_everything()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_csv = os.path.join(
        RESULTS_DIR, output_filename(args.n, half_label, args.tag))

    rows = []
    already_done = set()
    if not args.no_resume:
        try:
            rows, already_done = load_resume_rows(out_csv, langs, args.n, trrs)
        except ValueError as exc:
            ap.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print("  WHISPER-MEDIUM FIXED-RATE SWEEP — K-cosine intra-block merging")
    print(f"  Config {CONFIG_NAME}: layers={CONFIG_LAYERS}  (A-ToMe canonical, 8 layers)")
    print("  Greedy only  ·  no SP  ·  no random")
    print(f"  Device : {device}  ·  Model: {MODEL_NAME}")
    print(f"  Langs  ({len(langs)}/{len(LANGS_ALL)}, half={half_label}): {langs}")
    print(f"  TRRs   : {trrs}")
    for t in trrs:
        r = per_layer_r(t, len(CONFIG_LAYERS))
        print(f"    TRR={t:.2f}  →  per_layer_r={r:.4f}")
    print(f"  Output : {out_csv}")
    print(f"{SEP}\n")

    if already_done:
        print(
            f"  [RESUME] Preserving {len(already_done)} complete language "
            f"block(s): {sorted(already_done)}\n"
        )

    if len(already_done) < len(langs):
        print("  Loading model + processor ...")
        processor = WhisperProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        ).to(device).eval()
        normalizer = BasicTextNormalizer()
        # Defensive against transformers mutating the passed config. The
        # fallback helper also copies it for each decoding attempt.
        gen_cfg = copy.deepcopy(model.generation_config)
    else:
        print("  [RESUME] All requested language blocks are complete; no model load needed.")
    t_run = time.time()

    for li, lang_id in enumerate(langs, start=1):
        if lang_id in already_done:
            print(f"  [{li}/{len(langs)}] {lang_id}: [RESUME → SKIP complete block]")
            continue
        info = LANGUAGES.get(lang_id, {})
        lang_name = info.get("name", lang_id)
        tonal     = info.get("tonal", "")
        resource  = info.get("resource", "")
        family    = info.get("family", "")
        whisper_code = info.get("whisper_code", None)

        print(f"\n{SEP}")
        print(f"  [{li}/{len(langs)}] LANG: {lang_id}  ({lang_name})  "
              f"tonal={tonal}  resource={resource}  family={family}")
        if whisper_code is None:
            print("  whisper_code=None → decoding will return empty; "
                  "WER reported for completeness only.")
        print(SEP)

        print(f"  Loading {args.n} samples ...", flush=True)
        samples = preload_samples(lang_id, args.n, processor, device)

        # --- baseline (no merging) once per language ---
        print(f"  [{lang_id}] baseline (no merging) ...", flush=True)
        (b_wer, _, _, _, b_halluc, _, b_elapsed) = run_condition(
            model, processor, gen_cfg, samples, lang_id, normalizer,
            merge_layers=[], trr=0.0)
        print(f"    baseline WER = {b_wer*100:6.2f}%   "
              f"halluc_rate = {b_halluc['halluc_rate']*100:5.2f}%   "
              f"(n_high_cr={b_halluc['halluc_n_high_cr']}/"
              f"{b_halluc['halluc_n_total']}, "
              f"n_empty={b_halluc['halluc_n_empty']})   "
              f"({b_elapsed:5.1f}s)",
              flush=True)
        rows.append({
            "lang_id": lang_id, "lang_name": lang_name,
            "tonal": tonal, "resource": resource, "family": family,
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

        # Write atomically so a killed process leaves the last complete file.
        rows = write_rows(out_csv, rows, langs, trrs)
        print(f"  [{lang_id}] CSV updated → {out_csv}")

    # --- Final summary table ---
    print(f"\n{SEP}\n  SUMMARY — Config {CONFIG_NAME} {CONFIG_LAYERS}")
    print(f"  WER deltas (pp) vs baseline, n={args.n}\n{SEP}")
    hdr = "  ".join(f"TRR={t:.2f}".rjust(9) for t in trrs)
    print(f"  {'Lang':>8}  {'Baseline%':>10}  " + hdr)
    print(f"  {'-'*8:>8}  {'-'*10:>10}  " + "  ".join("-" * 9 for _ in trrs))

    by_lct = {
        (r["lang_id"], r["config"], float(r["trr"])): r
        for r in rows
    }
    baseline = {
        r["lang_id"]: float(r["wer"])
        for r in rows
        if r["config"] == "baseline"
    }

    for lang in langs:
        b = baseline.get(lang, float("nan"))
        cells = []
        for t in trrs:
            r = by_lct.get((lang, CONFIG_NAME, t))
            d = float(r["wer_delta"]) if r else float("nan")
            cells.append(f"{d*100:+6.2f}pp".rjust(9))
        print(f"  {lang:>8}  {b*100:>9.2f}%  " + "  ".join(cells))

    mean_cells = []
    for t in trrs:
        ds = [float(by_lct[(lang_id, CONFIG_NAME, t)]["wer_delta"])
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
