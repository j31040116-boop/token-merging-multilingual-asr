#!/usr/bin/env bash
# Smoke test for the layer-similarity pipeline (~30s on GPU).
# Runs whisper-small on 3 langs at n=5 and validates the output CSV shape.
#
# Usage (from repo root):
#   bash scripts/smoke_test_layer_similarity.sh
set -euo pipefail

REPO="${TMM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

SMOKE_LANGS="en_us vi_vn ta_in"
SMOKE_N=5

# Route output into a smoke-only temp dir so this never collides with real
# eval runs or writes into site-packages (see tests/test_output_paths.py).
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tmm-smoke.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
OUT_CSV="$OUT_DIR/layer_similarity_whisper-small_n${SMOKE_N}.csv"

echo "[smoke] whisper-small on $SMOKE_LANGS at n=$SMOKE_N ..."
echo "[smoke] output will land at: $OUT_CSV"
python -m tmm_asr.eval.layer_similarity \
    --model openai/whisper-small \
    --n "$SMOKE_N" \
    --langs $SMOKE_LANGS \
    --out-dir "$OUT_DIR"

echo
echo "[smoke] verifying output ..."
OUT_CSV="$OUT_CSV" python - <<'PY_INNER'
import csv, os, sys
from collections import Counter
path = os.environ["OUT_CSV"]
rows = list(csv.DictReader(open(path)))
langs = Counter(r["lang_id"] for r in rows)
print(f"  CSV rows:    {len(rows)}")
print(f"  langs found: {dict(langs)}")
expected = {"en_us", "vi_vn", "ta_in"}
if set(langs) != expected:
    print(f"  FAIL: got {set(langs)}, expected {expected}"); sys.exit(1)
# whisper-small has 12 encoder layers
for lang, n in langs.items():
    if n != 12:
        print(f"  FAIL: {lang} has {n} rows, expected 12"); sys.exit(1)
cos = [float(r["cos_mean"]) for r in rows if r["cos_mean"] not in ("", "nan")]
print(f"  cos_mean range: {min(cos):.3f} .. {max(cos):.3f}")
if not (0.0 < min(cos) and max(cos) <= 1.001):
    print("  FAIL: cos_mean out of expected range"); sys.exit(1)
print("  PASS")
PY_INNER

echo "[smoke] OK. Next: bash scripts/regenerate_layer_similarity.sh"
