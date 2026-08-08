"""Behavioural tests for cross-scale incremental resume validation."""

import csv

import pytest

from tmm_asr.eval.cross_scale import FIELDNAMES, load_resume_rows


def row(lang_id, config, trr, *, model="whisper-small", n=264):
    result = {field: "" for field in FIELDNAMES}
    result.update(
        lang_id=lang_id,
        model=model,
        config=config,
        trr=trr,
        n_samples=n,
        wer=0.1,
        wer_delta=0.0,
    )
    return result


def block(lang_id, trrs=(0.05, 0.1)):
    return [
        row(lang_id, "baseline", 0.0),
        *[row(lang_id, "A", trr) for trr in trrs],
    ]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_resume_preserves_only_complete_blocks_in_cli_order(tmp_path):
    path = tmp_path / "partial.csv"
    write_csv(path, block("vi_vn") + block("af_za") + [row("cy_gb", "baseline", 0)])

    rows, complete = load_resume_rows(
        path,
        ["af_za", "vi_vn", "cy_gb"],
        264,
        [0.05, 0.1],
        "whisper-small",
    )

    assert complete == {"af_za", "vi_vn"}
    assert [item["lang_id"] for item in rows] == [
        "af_za", "af_za", "af_za", "vi_vn", "vi_vn", "vi_vn"
    ]


def test_baseline_only_resume_requires_only_baseline(tmp_path):
    path = tmp_path / "baseline.csv"
    write_csv(path, [row("af_za", "baseline", 0)])

    rows, complete = load_resume_rows(
        path, ["af_za"], 264, [], "whisper-small"
    )

    assert len(rows) == 1
    assert complete == {"af_za"}


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        (row("af_za", "baseline", 0, model="whisper-medium"), "model"),
        (row("af_za", "baseline", 0, n=32), "n_samples=32"),
        (row("vi_vn", "baseline", 0), "unrequested language"),
        (row("af_za", "A", 0.4), "unexpected condition"),
    ],
)
def test_incompatible_resume_is_rejected(tmp_path, bad_row, message):
    path = tmp_path / "incompatible.csv"
    write_csv(path, [bad_row])

    with pytest.raises(ValueError, match=message):
        load_resume_rows(path, ["af_za"], 264, [0.05, 0.1], "whisper-small")
