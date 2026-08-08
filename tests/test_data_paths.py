"""Tests for writable, configurable cache and checkpoint directories."""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_env(monkeypatch):
    """Ensure each test starts with a clean env."""
    for var in ("TMM_CACHE_DIR", "TMM_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestFleursCacheDir:

    def test_default_is_not_inside_package(self):
        import tmm_asr
        import tmm_asr.data.fleurs as fleurs
        importlib.reload(fleurs)
        pkg_root = os.path.dirname(tmm_asr.__file__)
        assert not fleurs.CACHE_DIR.startswith(pkg_root), (
            f"FLEURS cache defaults to {fleurs.CACHE_DIR!r}, which is inside "
            f"the installed package at {pkg_root!r}. Wheel installs cannot "
            f"write there."
        )

    def test_tmm_cache_dir_env_var_honoured(self, tmp_path, monkeypatch):
        target = tmp_path / "my-cache"
        monkeypatch.setenv("TMM_CACHE_DIR", str(target))
        import tmm_asr.data.fleurs as fleurs
        importlib.reload(fleurs)
        assert fleurs.CACHE_DIR == str(target)


class TestDoRACheckpointDir:

    def test_default_is_not_inside_package(self):
        import tmm_asr
        import tmm_asr.train.dora as dora
        importlib.reload(dora)
        pkg_root = os.path.dirname(tmm_asr.__file__)
        assert not dora.CHECKPOINT_DIR.startswith(pkg_root), (
            f"DoRA checkpoint dir defaults to {dora.CHECKPOINT_DIR!r}, which "
            f"is inside the installed package at {pkg_root!r}."
        )

    def test_tmm_checkpoint_dir_env_var_honoured(self, tmp_path, monkeypatch):
        target = tmp_path / "my-checkpoints" / "run-1"
        monkeypatch.setenv("TMM_CHECKPOINT_DIR", str(target))
        import tmm_asr.train.dora as dora
        importlib.reload(dora)
        assert dora.CHECKPOINT_DIR == str(target)
