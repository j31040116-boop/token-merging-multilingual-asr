"""
Regression: default output directories must be CWD-relative, not derived
from `Path(__file__).parent.parent.parent`. When the package is installed
as a wheel, that ancestor path resolves to site-packages, so figure/eval
defaults would try to write inside the venv (usually read-only) instead
of under the user's working directory.

Strategy: chdir to a fresh temp directory before import and assert that
DEFAULT_OUT_DIR resolves *under that temp directory*. A `__file__`-derived
default cannot satisfy this — only a `Path.cwd()`-derived one can.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

FIGURE_MODULES = [
    "tmm_asr.figures.fig1_layer_similarity",
    "tmm_asr.figures.fig2_highres",
    "tmm_asr.figures.fig2_lowres",
    "tmm_asr.figures.fig3_cross_scale_ft",
]

EVAL_MODULES = [
    "tmm_asr.eval.main_sweep",
    "tmm_asr.eval.cross_scale",
    "tmm_asr.eval.ft_merge",
    "tmm_asr.eval.ft_holdout",
    "tmm_asr.eval.layer_similarity",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TMM_OUT_DIR", raising=False)
    yield


def _fresh_import(name: str):
    for m in list(sys.modules):
        if m == name:
            del sys.modules[m]
    return importlib.import_module(name)


class TestFigureDefaults:

    @pytest.mark.parametrize("modname", FIGURE_MODULES)
    def test_default_out_dir_is_cwd_relative(self, modname, tmp_path, monkeypatch):
        """DEFAULT_OUT_DIR must resolve under the CWD at import time, not
        under the package install location. Simulates a wheel install where
        <package>/../ is site-packages."""
        monkeypatch.chdir(tmp_path)
        mod = _fresh_import(modname)
        default = Path(getattr(mod, "DEFAULT_OUT_DIR")).resolve()
        try:
            default.relative_to(tmp_path.resolve())
        except ValueError:
            pytest.fail(
                f"{modname}.DEFAULT_OUT_DIR = {default!r} is not under CWD "
                f"({tmp_path!r}). Wheel installs would write into site-packages."
            )

    @pytest.mark.parametrize("modname", FIGURE_MODULES)
    def test_no_import_time_mkdir(self, modname, tmp_path, monkeypatch):
        """Importing must not create DEFAULT_OUT_DIR under a fresh cwd."""
        monkeypatch.chdir(tmp_path)
        _fresh_import(modname)
        leftovers = list(tmp_path.iterdir())
        assert not leftovers, (
            f"Importing {modname} created {[p.name for p in leftovers]!r} "
            f"in cwd; module must not mkdir at import time (only in main())."
        )


class TestEvalDefaults:

    @pytest.mark.parametrize("modname", EVAL_MODULES)
    def test_default_out_dir_is_cwd_relative(self, modname, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mod = _fresh_import(modname)
        default = Path(getattr(mod, "RESULTS_DIR")).resolve()
        try:
            default.relative_to(tmp_path.resolve())
        except ValueError:
            pytest.fail(
                f"{modname}.RESULTS_DIR = {default!r} is not under CWD "
                f"({tmp_path!r}). Wheel installs would write into site-packages."
            )

    @pytest.mark.parametrize("modname", EVAL_MODULES)
    def test_tmm_out_dir_env_var_honoured(self, modname, tmp_path, monkeypatch):
        target = tmp_path / "my-outputs"
        monkeypatch.setenv("TMM_OUT_DIR", str(target))
        mod = _fresh_import(modname)
        assert Path(mod.RESULTS_DIR) == target, (
            f"{modname} did not honour TMM_OUT_DIR (got {mod.RESULTS_DIR!r})."
        )
