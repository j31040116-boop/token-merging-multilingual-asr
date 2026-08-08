"""Tests for main-sweep cohort resolution, filenames, and resumability."""
from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from tmm_asr.eval.main_sweep import (
    CONFIG_NAME,
    FIELDNAMES,
    HALVES,
    LANGS_ALL,
    load_resume_rows,
    order_rows,
    output_filename,
    resolve_langs_and_label,
)

# Filename helpers

class TestOutputFilename:

    def test_frozen_override_is_halfall(self):
        assert output_filename(264, "all", "single") == \
            "fixed_rate_main_rerun_cfgA_n264_halfall_single.csv"

    def test_no_tag_omits_trailing_underscore(self):
        assert output_filename(264, "all", "") == \
            "fixed_rate_main_rerun_cfgA_n264_halfall.csv"

    def test_half_label_variants(self):
        assert "halfA" in output_filename(264, "A", "")
        assert "halfB" in output_filename(264, "B", "single")
        assert "halfcustom" in output_filename(264, "custom", "single")

    def test_n_is_embedded(self):
        assert "_n32_" in output_filename(32, "all", "smoke")
        assert "_n264_" in output_filename(264, "all", "single")


class TestResolveLangsAndLabel:

    def test_no_flags_returns_plotted_cohort(self):
        langs, label = resolve_langs_and_label(None, None, None)
        assert langs == LANGS_ALL
        assert label == "plotted"

    def test_lang_half_a(self):
        langs, label = resolve_langs_and_label(None, "A", None)
        assert langs == HALVES["A"]
        assert label == "A"

    def test_explicit_langs_default_label_is_custom(self):
        langs, label = resolve_langs_and_label(["af_za", "vi_vn"], None, None)
        assert langs == ["af_za", "vi_vn"]
        assert label == "custom"

    def test_explicit_langs_plus_half_label_all_yields_halfall(self):
        """Produce the frozen artifact's canonical filename: pass all 18
        languages AND --half-label all. The label override must win, even
        when --langs would otherwise force 'custom'."""
        eighteen = [
            "af_za", "am_et", "cy_gb", "ha_ng", "is_is", "jv_id",
            "kk_kz", "ln_cd", "mt_mt", "pa_in", "sn_zw", "so_so",
            "sw_ke", "ta_in", "th_th", "uz_uz", "vi_vn", "yo_ng",
        ]
        langs, label = resolve_langs_and_label(eighteen, None, "all")
        assert langs == eighteen
        assert label == "all"
        # And the assembled filename matches the frozen paper CSV.
        assert output_filename(264, label, "single") == \
            "fixed_rate_main_rerun_cfgA_n264_halfall_single.csv"

    def test_override_wins_over_lang_half_too(self):
        _, label = resolve_langs_and_label(None, "A", "custom-suffix")
        assert label == "custom-suffix"


# Keep the documented filename override discoverable from the CLI.

def test_half_label_flag_in_help():
    out = subprocess.run(
        [sys.executable, "-m", "tmm_asr.eval.main_sweep", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, (
        f"--help exited {out.returncode}\nstderr: {out.stderr[:500]}"
    )
    assert "--half-label" in out.stdout, (
        "main_sweep --help does not advertise --half-label."
    )


def _row(lang_id, config, trr, n=264):
    row = {field: "" for field in FIELDNAMES}
    row.update({
        "lang_id": lang_id,
        "config": config,
        "trr": trr,
        "n_samples": n,
        "wer": 0.1,
        "wer_delta": 0.0,
    })
    return row


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class TestResumeRows:

    trrs = [0.05, 0.1]

    def complete_block(self, lang_id):
        return [
            _row(lang_id, "baseline", 0.0),
            *[_row(lang_id, CONFIG_NAME, trr) for trr in self.trrs],
        ]

    def test_missing_file_starts_fresh(self, tmp_path):
        rows, done = load_resume_rows(
            tmp_path / "missing.csv", ["af_za"], 264, self.trrs
        )
        assert rows == []
        assert done == set()

    def test_complete_blocks_resume_in_requested_order(self, tmp_path):
        path = tmp_path / "partial.csv"
        rows = self.complete_block("vi_vn") + self.complete_block("af_za")
        _write_csv(path, rows)

        resumed, done = load_resume_rows(
            path, ["af_za", "vi_vn", "cy_gb"], 264, self.trrs
        )

        assert done == {"af_za", "vi_vn"}
        assert [row["lang_id"] for row in resumed] == [
            "af_za", "af_za", "af_za", "vi_vn", "vi_vn", "vi_vn"
        ]

    def test_incomplete_language_block_is_discarded(self, tmp_path):
        path = tmp_path / "partial.csv"
        _write_csv(path, self.complete_block("af_za") + [_row("vi_vn", "baseline", 0)])

        resumed, done = load_resume_rows(
            path, ["af_za", "vi_vn"], 264, self.trrs
        )

        assert done == {"af_za"}
        assert {row["lang_id"] for row in resumed} == {"af_za"}

    def test_rejects_unrequested_language(self, tmp_path):
        path = tmp_path / "wrong-language.csv"
        _write_csv(path, self.complete_block("vi_vn"))

        with pytest.raises(ValueError, match="unrequested language"):
            load_resume_rows(path, ["af_za"], 264, self.trrs)

    def test_rejects_different_sample_count(self, tmp_path):
        path = tmp_path / "wrong-n.csv"
        _write_csv(path, [_row("af_za", "baseline", 0, n=32)])

        with pytest.raises(ValueError, match="n_samples=32"):
            load_resume_rows(path, ["af_za"], 264, self.trrs)

    def test_rejects_different_condition_set(self, tmp_path):
        path = tmp_path / "wrong-trr.csv"
        _write_csv(path, [_row("af_za", CONFIG_NAME, 0.4)])

        with pytest.raises(ValueError, match="unexpected condition"):
            load_resume_rows(path, ["af_za"], 264, self.trrs)

    def test_rejects_duplicate_condition(self, tmp_path):
        path = tmp_path / "duplicate.csv"
        duplicate = _row("af_za", "baseline", 0.0)
        _write_csv(path, [duplicate, duplicate])

        with pytest.raises(ValueError, match="duplicate condition"):
            load_resume_rows(path, ["af_za"], 264, self.trrs)

    def test_order_rows_uses_language_then_condition_order(self):
        shuffled = [
            _row("vi_vn", CONFIG_NAME, 0.1),
            _row("af_za", CONFIG_NAME, 0.05),
            _row("vi_vn", "baseline", 0),
            _row("af_za", "baseline", 0),
            _row("vi_vn", CONFIG_NAME, 0.05),
            _row("af_za", CONFIG_NAME, 0.1),
        ]
        ordered = order_rows(shuffled, ["af_za", "vi_vn"], self.trrs)
        assert [(row["lang_id"], row["config"], float(row["trr"])) for row in ordered] == [
            ("af_za", "baseline", 0.0),
            ("af_za", CONFIG_NAME, 0.05),
            ("af_za", CONFIG_NAME, 0.1),
            ("vi_vn", "baseline", 0.0),
            ("vi_vn", CONFIG_NAME, 0.05),
            ("vi_vn", CONFIG_NAME, 0.1),
        ]
