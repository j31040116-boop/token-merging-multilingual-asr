# Reproducing the paper's frozen CSVs and figures

Every number and figure in the ICNLSP 2026 paper is backed by one of the CSVs in [`tmm_asr/paper_results/`](../tmm_asr/paper_results/) — see [`manifest.json`](../tmm_asr/paper_results/manifest.json) for the paper-claim → CSV mapping with sha256s.

This document lists the exact command that produced each CSV. If you want to regenerate one from scratch, run the command **exactly as written** — the `--tag` values and language selections are load-bearing because they become part of the output filename, and the figure scripts look for those specific filenames.

**Reproducibility fine print.** No re-run of a WER script produces a *bit-for-bit* copy of a frozen CSV: every eval CSV records an `elapsed_s` column that reflects wall-clock timing, and numerical differences across GPU hardware, drivers, CUDA/cuBLAS versions, and discrete decoding decisions can change exact WER values. When this document says a recipe "matches" the frozen file, we mean a **compatible schema** (all frozen columns are present; newer diagnostic columns may be added), the same cohort, and the same row order, with results compared against the frozen artifact and any observed numerical differences reported — not a matching sha256 or a universal WER tolerance. Figure regeneration from a frozen CSV is deterministic on the same matplotlib/font stack; PDFs additionally embed a creation timestamp.

Every command below writes to `$TMM_OUT_DIR` if that env var is set, else to `<cwd>/outputs/eval/`. To overwrite a frozen CSV in place, add `--out-dir tmm_asr/paper_results`.

---

## Setup

All commands assume:

```bash
cd token-merging-multilingual-asr
pip install -e ".[dev]"                        # or the split requirements
```

Model and dataset revisions are pinned in `tmm_asr/eval/pipeline.py`; each executable evaluation script sets seed 42 and deterministic cuDNN mode — no per-command flags needed. For stricter cuBLAS determinism, export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching an evaluation.

### One-command core GPU reproduction

Run the complete whisper-medium §5.1 and whisper-small §5.2 sweeps sequentially
on one GPU from one terminal:

```bash
bash scripts/reproduce_core_gpu.sh
```

The script uses `python` from the active environment, runs the exact frozen
cohorts sequentially on one GPU, writes under `outputs/reproduction_full/`,
stops if either job fails, and checks the resulting compatible schema, cohort,
condition order, row order, and stable metadata against the frozen artifacts.
It reports observed WER differences without imposing or silently repairing a
numerical tolerance.

GPU selection follows this order: `TMM_GPU`, an existing
`CUDA_VISIBLE_DEVICES`, then physical GPU 0. Examples:

```bash
TMM_GPU=1 bash scripts/reproduce_core_gpu.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/reproduce_core_gpu.sh
TMM_PYTHON=/path/to/venv/bin/python TMM_GPU=1 \
    bash scripts/reproduce_core_gpu.sh
```

The launcher accepts exactly one visible GPU. `TMM_REPRO_DIR` changes the
output root. Both evaluation programs write after each complete language;
restarting the same command preserves compatible complete blocks and resumes
the rest. An incompatible existing CSV causes a clear error instead of mixing
runs. Use each evaluator's `--no-resume` option only when you intentionally
want to start that destination again. Progress is retained in `main.log`,
`small.log`, and `driver.log` below the output root.

On the measured RTX 3080 Ti setup, budget approximately **16–19 hours** for
both full frozen cohorts. Runtime is hardware-, cache-, and language-dependent.

---

## The 12 frozen CSVs

### 1. Cross-lingual sweep on whisper-medium (§5.1, Table 2 medium-stock column, Figure 2 bottom)

**Frozen file**: `fixed_rate_main_rerun_cfgA_n264_halfall_single.csv` — **contains 18 languages** (108 rows = 18 × 6 conditions).

The paper plots the 12-language `main_sweep_12` subset (af_za, cy_gb, ha_ng, is_is, jv_id, kk_kz, ln_cd, mt_mt, sw_ke, ta_in, th_th, vi_vn). The other 6 rows (am_et, pa_in, sn_zw, so_so, uz_uz, yo_ng) — noted in [`main_sweep.py`](../tmm_asr/eval/main_sweep.py) as baseline WER ≥ 100% on whisper-medium — were collected in the same run but excluded from Table 2 / Figure 2 for cohort-consistency reasons. The figure script filters to the plotted 12 automatically.

**Default command** (12 plotted langs → correct paper numbers, but a 72-row CSV — the frozen file has 108 rows over 18 langs):

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.main_sweep \
    --n 264 \
    --tag single
```

This writes `..._halfplotted_single.csv`, deliberately distinct from the
18-language frozen artifact so the two cohorts cannot silently overwrite one
another.

**To reproduce the frozen 18-language CSV under a compatible schema and matching cohort/row order**, pass all 18 langs explicitly AND override the filename token with `--half-label all` (an explicit `--langs` list would otherwise stamp the file `..._halfcustom_single.csv`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.main_sweep \
    --n 264 \
    --tag single \
    --half-label all \
    --langs af_za am_et cy_gb ha_ng is_is jv_id kk_kz ln_cd mt_mt \
            pa_in sn_zw so_so sw_ke ta_in th_th uz_uz vi_vn yo_ng
```

Rows in the frozen CSV appear in the order given above (alphabetical by ISO code); this recipe preserves that order because the eval loop iterates `--langs` in the CLI-provided order.

Notes:
- `--tag single` is required either way — without it the filename lacks the `_single` suffix that `fig2_lowres.py` reads.
- Cited numbers over the 12 plotted langs at TRR=0.40: mean ΔWER = −0.21 pp; worst degradation = +1.01 pp (sw_ke); largest improvement = −1.41 pp (ln_cd).
- Runtime: approximately 9 h (12-lang projection) / 13 h (full 18-lang) on the measured 3080 Ti run. Resume support avoids losing completed languages.

### 2. High-resource anchor sweep on whisper-medium (Figure 2 top)

**Frozen file**: `whisper_size_baseline_whisper-medium_n264_medium_highres.csv` — **contains 5 languages** (30 rows = 5 × 6 conditions).

The paper plots the 4-language anchor subset (en_us, fr_fr, de_de, es_419); Mandarin (cmn_hans_cn) was collected in the same run but excluded so the anchor cohort aligns with the cross-scale plots in Figure 3.

**Default command** — reproduces the frozen 5-language CSV under matching schema/cohort/row order (30 rows in the order `en_us, es_419, de_de, cmn_hans_cn, fr_fr`):

```bash
bash scripts/regenerate_highres.sh
```

**Paper-plot-only variant** (4 anchors, 24-row CSV — a strict subset of the frozen cohort, so its rows differ in count from the shipped file):

```bash
bash scripts/regenerate_highres.sh --plotted
```

Equivalently by hand — same 5-lang cohort in the frozen row order:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.cross_scale \
    --model openai/whisper-medium \
    --langs en_us es_419 de_de cmn_hans_cn fr_fr \
    --n 264 \
    --tag medium_highres
```

Or the 4-anchor paper-plot variant:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.cross_scale \
    --model openai/whisper-medium \
    --langs en_us fr_fr de_de es_419 \
    --n 264 \
    --tag medium_highres
```

Notes:
- `--tag medium_highres` matches the filename Figure 2 top reads.
- Row order in the frozen CSV = the CLI order above; cross_scale.py iterates `--langs` in list order (no sort), so reordering shuffles the rows even when contents are equivalent.
- Runtime: approximately 4.3 h for the frozen 5-language run; hardware and decoded output lengths materially affect it.

### 3. Cross-scale small (§5.2, Figure 3)

**Frozen file**: `whisper_size_baseline_whisper-small_n264_mix16.csv`

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.cross_scale \
    --model openai/whisper-small \
    --langs vi_vn ha_ng ln_cd ta_in mt_mt jv_id \
            en_us fr_fr de_de es_419 \
            th_th sw_ke af_za is_is cy_gb kk_kz \
    --n 264 \
    --tag mix16
```

Notes:
- 16-language cohort (12 main-sweep + 4 anchors). Default is mix6 — must override with the full `--langs` list.
- `--tag mix16` is required to match the filename.
- Row order above matches the row order in the frozen CSV (`vi_vn ha_ng ln_cd ta_in mt_mt jv_id en_us fr_fr de_de es_419 th_th sw_ke af_za is_is cy_gb kk_kz`). `cross_scale.py` iterates `--langs` in list order (no sort), so reordering here breaks row-level match even though contents are equivalent.
- Runtime: approximately 3–6 h on a single 3080 Ti, depending on cache state and decoded output lengths.

### 4. Cross-scale large-v3 (§5.2, Figure 3, dual-GPU)

**Frozen files**: `..._mix16_gpu0.csv` + `..._mix16_gpu1.csv` (and their merger `..._mix16.csv`)

```bash
bash scripts/launch_large_v3_dual_gpu.sh
# Budget ~9 h; monitor with tmux attach -t large_v3_gpu0 / gpu1
python scripts/merge_large_v3_shards.py
```

Notes:
- Launcher exports `TMM_OUT_DIR=$REPO/outputs/eval/large_v3` and writes both shards there.
- Merger defaults to that same directory.
- To rebuild the frozen merged CSV in place: `python scripts/merge_large_v3_shards.py --in-dir tmm_asr/paper_results`.

### 5. DoRA + merging composition (§5.3, Figure 3)

**Frozen file**: `finetune_merge_rerun_cfgA_n264_checkpoint-2000_mix6.csv`

The output filename is derived from `os.path.basename(args.checkpoint)`. To reproduce the frozen filename you must either:

**Option A** — checkpoint at a local dir whose basename is `checkpoint-2000`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.ft_merge \
    --checkpoint /path/to/whisper-medium-dora-mix6/checkpoint-2000 \
    --n 264 \
    --tag mix6
```

**Option B** — checkpoint from HuggingFace (the released adapter), rename output:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.ft_merge \
    --checkpoint dylan01163104/whisper-medium-dora-mix6 \
    --n 264
# Produces: finetune_merge_rerun_cfgA_n264_whisper-medium-dora-mix6.csv
# Rename to the frozen name:
mv outputs/eval/finetune_merge_rerun_cfgA_n264_whisper-medium-dora-mix6.csv \
   outputs/eval/finetune_merge_rerun_cfgA_n264_checkpoint-2000_mix6.csv
```

Notes:
- The HF adapter is pinned at revision `ad9144916cf661ea2ef462ad273077343c3d803d` and matches the local `checkpoint-2000` snapshot bit-for-bit.
- Runtime: approximately 4.9 h in the frozen run.

### 6. DoRA held-out generalisation (§5.3 ¶3)

**Frozen file**: `finetune_holdout_cfgA_n264_checkpoint-2000_mix6.csv`

Same `--checkpoint`/renaming caveat as §5. See:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.ft_holdout \
    --checkpoint /path/to/whisper-medium-dora-mix6/checkpoint-2000 \
    --n 264 \
    --tag mix6
```

Notes:
- Default 10-lang held-out cohort (4 anchors + 6 untrained mid/low): en_us fr_fr de_de es_419 th_th sw_ke af_za is_is cy_gb kk_kz.
- Runtime: approximately 5.9 h in the frozen run.

### 7–9. Layer-similarity per scale (Figure 1)

**Frozen files**:
- `layer_similarity_whisper-small_n264.csv`
- `layer_similarity_whisper-medium_n264.csv`
- `layer_similarity_whisper-large-v3_n264.csv`

```bash
bash scripts/regenerate_layer_similarity.sh
```

Runs all three sequentially (~28 min), writes to `$TMM_OUT_DIR` (default `outputs/eval/layer_similarity/`), then regenerates the figure from those fresh CSVs.

### 10. Theoretical FLOP summary (§5.4)

**Frozen file**: `theoretical_flops_summary.csv`

Analytic, no GPU. Two variants — do not use `--out-dir tmm_asr/paper_results` (that writes the eight `*_with_flops.csv` sidecars into the packaged frozen-results directory, and the next wheel build would ship them as if they were paper artifacts):

**Standard run** (sidecars + summary land in `outputs/flops/`, nothing touched inside the package):

```bash
python -m tmm_asr.eval.flops
```

**Refresh the packaged summary** without shipping sidecars — use the dedicated `--summary-out` flag:

```bash
python -m tmm_asr.eval.flops \
    --summary-out tmm_asr/paper_results/theoretical_flops_summary.csv
```

This writes only `theoretical_flops_summary.csv` inside the package (which pyproject already packages); the eight `*_with_flops.csv` sidecars stay in `outputs/flops/`.

Note: this re-derives per-model FLOP reductions from the `_with_flops` sidecar CSVs, which are computed on the fly from the six source CSVs (main_sweep, cross_scale small/large-v3, ft_merge, ft_holdout, medium_highres). The frozen `theoretical_flops_summary.csv` was regenerated to match — historical rows referencing files like `medium_stock_6FT` and `large_v3_func5` are no longer present.

---

## The 4 paper figures

Every figure regenerates from the frozen CSVs in `tmm_asr/paper_results/` (no GPU):

```bash
python -m tmm_asr.figures.fig1_layer_similarity   # Figure 1
python -m tmm_asr.figures.fig2_highres            # Figure 2 top
python -m tmm_asr.figures.fig2_lowres             # Figure 2 bottom
python -m tmm_asr.figures.fig3_cross_scale_ft     # Figure 3
```

To regenerate from a fresh set of CSVs (e.g. after re-running an eval), pass `--in-dir`:

```bash
python -m tmm_asr.figures.fig1_layer_similarity --in-dir outputs/eval/layer_similarity
```

PNG outputs are visually identical when regenerated from the frozen CSVs on the same matplotlib/font stack; exact byte equality depends on the matplotlib version. PDFs additionally embed a creation timestamp.

---

## Wall-clock benchmark (Limitations §)

Not a frozen CSV — depends on hardware. On our 3080 Ti it reproduces the paper's 1.21× encoder speedup at TRR=0.40 (measured 1.27× at N=48):

```bash
CUDA_VISIBLE_DEVICES=0 python -m tmm_asr.eval.wallclock \
    --encoder-only \
    --out-dir outputs/wallclock
```

Runtime: ~15–20 min. The methodology (Latin-square counterbalancing, N=48 = 6 langs × 8 samples, encoder-only isolation) is documented in the script's module docstring.

---

## What each eval script's default gives you

If you run without the exact flags above, you get a differently-named CSV that still contains valid data — just not the frozen filename. Rename or repass `--tag` to match if you want to drop it into `tmm_asr/paper_results/` and have the figure scripts pick it up.

| Script | Default output name | Frozen name |
|---|---|---|
| `main_sweep` (default 12 langs) | `..._halfall_single.csv` (72 rows) | `..._halfall_single.csv` (108 rows — needs explicit `--langs` for all 18, see §1) |
| `cross_scale --model whisper-small` | `..._n264.csv` (6 langs) | `..._n264_mix16.csv` (needs `--tag mix16` + 16 langs) |
| `cross_scale --model whisper-medium` (4 anchors) | `..._medium_highres.csv` (24 rows) | `..._medium_highres.csv` (30 rows — needs cmn_hans_cn in `--langs`, see §2) |
| `ft_merge` (HF checkpoint) | `..._whisper-medium-dora-mix6.csv` | `..._checkpoint-2000_mix6.csv` (rename or use local checkpoint) |
| `ft_holdout` (HF checkpoint) | `..._whisper-medium-dora-mix6.csv` | `..._checkpoint-2000_mix6.csv` (same) |
| `layer_similarity` | `..._n{N}.csv` (name matches frozen) | ✓ |
| `flops` | `theoretical_flops_summary.csv` (name matches frozen) | ✓ |
