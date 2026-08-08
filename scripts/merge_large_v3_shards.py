"""
Merge the two per-GPU whisper-large-v3 mix16 CSVs into the canonical mix16 CSV.

Run AFTER both large_v3_gpu0 and large_v3_gpu1 tmux sessions finish.

Usage (from repo root):
  python scripts/merge_large_v3_shards.py            # uses defaults below
  python scripts/merge_large_v3_shards.py --in-dir tmm_asr/paper_results \\
      --out tmm_asr/paper_results/whisper_size_baseline_whisper-large-v3_n264_mix16.csv

Validates that the union has exactly 16 langs x 6 TRRs (no overlap, no gaps).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Match launcher default. When rebuilding the FROZEN paper CSV, pass
# --in-dir tmm_asr/paper_results explicitly.
DEFAULT_DIR = os.environ.get(
    "TMM_OUT_DIR", os.path.join(_REPO, "outputs", "eval", "large_v3"),
)

EXPECTED_LANGS = {
    "vi_vn", "ta_in", "jv_id", "mt_mt", "ln_cd", "ha_ng",
    "en_us", "fr_fr", "de_de", "es_419",
    "th_th", "sw_ke", "af_za", "is_is", "cy_gb", "kk_kz",
}
EXPECTED_TRRS = {0.0, 0.05, 0.10, 0.20, 0.30, 0.40}


def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default=DEFAULT_DIR,
                    help=f"Directory containing the two GPU shards (default: {DEFAULT_DIR})")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <in-dir>/whisper_size_baseline_whisper-large-v3_n264_mix16.csv)")
    args = ap.parse_args()

    gpu0_csv = os.path.join(args.in_dir, "whisper_size_baseline_whisper-large-v3_n264_mix16_gpu0.csv")
    gpu1_csv = os.path.join(args.in_dir, "whisper_size_baseline_whisper-large-v3_n264_mix16_gpu1.csv")
    out_csv  = args.out or os.path.join(args.in_dir, "whisper_size_baseline_whisper-large-v3_n264_mix16.csv")

    rows0 = load(gpu0_csv)
    rows1 = load(gpu1_csv)

    # Sanity: per-CSV langs
    by0 = defaultdict(set)
    for r in rows0:
        by0[r["lang_id"]].add(float(r["trr"]))
    by1 = defaultdict(set)
    for r in rows1:
        by1[r["lang_id"]].add(float(r["trr"]))

    print(f"GPU0 CSV: {len(by0)} langs, {len(rows0)} rows")
    print(f"GPU1 CSV: {len(by1)} langs, {len(rows1)} rows")

    overlap = set(by0) & set(by1)
    if overlap:
        sys.exit(f"FAIL: langs appear in both GPU CSVs: {overlap}")

    # Per-lang completeness
    bad = []
    for lang, trrs in {**by0, **by1}.items():
        if trrs != EXPECTED_TRRS:
            bad.append((lang, sorted(trrs)))
    if bad:
        print("WARN: incomplete langs:")
        for lang, trrs in bad:
            print(f"  {lang}: TRRs={trrs}")
        sys.exit("FAIL: incomplete data — let the runs finish before merging.")

    union_langs = set(by0) | set(by1)
    missing = EXPECTED_LANGS - union_langs
    if missing:
        sys.exit(f"FAIL: missing langs: {sorted(missing)}")
    extra = union_langs - EXPECTED_LANGS
    if extra:
        sys.exit(f"FAIL: unexpected langs: {sorted(extra)}")

    # Merge — preserve the canonical lang order
    canonical_order = [
        "vi_vn", "ha_ng", "ln_cd", "ta_in", "mt_mt", "jv_id",
        "en_us", "fr_fr", "de_de", "es_419",
        "th_th", "sw_ke", "af_za", "is_is", "cy_gb", "kk_kz",
    ]
    by_lang = defaultdict(list)
    for r in rows0 + rows1:
        by_lang[r["lang_id"]].append(r)

    fieldnames = list(rows0[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for lang in canonical_order:
            for r in by_lang[lang]:
                w.writerow(r)

    print(f"\nOK: merged {len(rows0) + len(rows1)} rows -> {out_csv}")
    print(f"     {len(EXPECTED_LANGS)} langs x {len(EXPECTED_TRRS)} TRRs")


if __name__ == "__main__":
    main()
