"""
Unit tests for tmm_asr.eval._dora_ckpt.resolve_dora_checkpoint.

No GPU, no network — HF Hub probing is mocked so tests run in CI.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tmm_asr.eval._dora_ckpt import (
    ResolvedCheckpoint,
    resolve_dora_checkpoint,
)

# Local directories

class TestLocalPath:

    def test_existing_directory_resolves_local(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        out = resolve_dora_checkpoint(str(d), revision="deadbeef", hub_probe=False)
        assert out.kind == "local"
        assert out.ident == str(d)
        assert out.revision is None

    def test_existing_dir_wins_over_hf_probe(self, tmp_path):
        """If both a local dir and a matching HF id would work, local wins
        and the HF probe is never called."""
        d = tmp_path / "org__repo"
        d.mkdir()
        # hub_probe=True but should be short-circuited by isdir; if HfApi
        # were reached, the mock would raise and fail the test.
        with patch("tmm_asr.eval._dora_ckpt.HfApi", side_effect=AssertionError("should not be called")):
            out = resolve_dora_checkpoint(str(d), revision="x", hub_probe=True)
        assert out.kind == "local"


# Invalid paths

class TestRejectsBadPaths:

    @pytest.mark.parametrize("bad", [
        "/absolute/does/not/exist",
        "./relative/does/not/exist",
        "../relative/parent",
        "~/tilde-home/adapter",
    ])
    def test_absolute_and_dot_prefixed_paths_rejected(self, bad):
        with pytest.raises(FileNotFoundError, match="looks like a local path"):
            resolve_dora_checkpoint(bad, revision="x", hub_probe=False)

    def test_dotdot_component_rejected(self):
        with pytest.raises(FileNotFoundError, match="looks like a local path"):
            resolve_dora_checkpoint("org/../etc/passwd", revision="x", hub_probe=False)

    def test_totally_bogus_string_rejected(self):
        # No slash at all — HF validator rejects. We wrap as ValueError.
        with pytest.raises(ValueError, match="not a valid HuggingFace repo id"):
            resolve_dora_checkpoint("just-a-name", revision="x", hub_probe=False)

    def test_multiple_slashes_rejected(self):
        # 'a/b/c' — HF validator rejects (org/repo has exactly one slash).
        with pytest.raises(ValueError, match="not a valid HuggingFace repo id"):
            resolve_dora_checkpoint("a/b/c", revision="x", hub_probe=False)


# Hugging Face Hub repositories

class TestHFBranch:

    def _fake_hf_api(self, ok: bool = True):
        """Return a MagicMock HfApi class whose repo_info() succeeds or fails."""
        class _FakeApi:
            def __init__(self, *a, **kw):
                pass
            def repo_info(self, repo_id, revision=None, repo_type=None):
                if ok:
                    self.last_call = (repo_id, revision, repo_type)
                    return self.last_call
                raise RuntimeError("simulated: repo not found")
        return _FakeApi

    def test_valid_id_probes_hub_and_returns_hf(self):
        rev = "ad9144916cf661ea2ef462ad273077343c3d803d"
        with patch("tmm_asr.eval._dora_ckpt.HfApi", self._fake_hf_api(ok=True)):
            out = resolve_dora_checkpoint(
                "dylan01163104/whisper-medium-dora-mix6", revision=rev
            )
        assert out.kind == "hf"
        assert out.ident == "dylan01163104/whisper-medium-dora-mix6"
        assert out.revision == rev

    def test_probe_failure_raises_filenotfound(self):
        """Repo not found on Hub -> FileNotFoundError with actionable message.

        RepositoryNotFoundError has a required `response` kwarg in current
        huggingface_hub; we subclass it and bypass __init__ so we can raise
        it in the test without constructing a real HTTP response.
        """
        from huggingface_hub.errors import RepositoryNotFoundError
        class _SimNotFound(RepositoryNotFoundError):
            def __init__(self, msg):
                Exception.__init__(self, msg)
        class _FakeApi:
            def __init__(self, *a, **kw): pass
            def repo_info(self, *a, **kw):
                raise _SimNotFound("simulated: repo not found")
        with patch("tmm_asr.eval._dora_ckpt.HfApi", _FakeApi):
            with pytest.raises(FileNotFoundError, match="not found on the Hub"):
                resolve_dora_checkpoint("user/wrong-repo", revision="abcd1234")

    def test_gated_repo_reports_auth_not_notfound(self):
        """Gated repositories report an authentication error."""
        from huggingface_hub.errors import GatedRepoError

        class _SimGated(GatedRepoError):
            def __init__(self, msg):
                Exception.__init__(self, msg)

        class _FakeApi:
            def __init__(self, *a, **kw): pass
            def repo_info(self, *a, **kw):
                raise _SimGated("simulated: access to this repo is gated")

        with patch("tmm_asr.eval._dora_ckpt.HfApi", _FakeApi):
            with pytest.raises(PermissionError) as excinfo:
                resolve_dora_checkpoint(
                    "meta-llama/Llama-3-8B", revision="abcd1234")
        msg = str(excinfo.value).lower()
        assert "gated" in msg or "access" in msg or "auth" in msg, (
            f"gated-repo message should point at auth/access, got: {excinfo.value!r}"
        )
        # Gated repositories exist, so they must not use the missing-repo path.
        assert "not found on the hub" not in msg

    def test_probe_skipped_when_hub_probe_false(self):
        """hub_probe=False avoids the network probe."""
        with patch("tmm_asr.eval._dora_ckpt.HfApi",
                   side_effect=AssertionError("HfApi should not be called")):
            out = resolve_dora_checkpoint(
                "dylan01163104/whisper-medium-dora-mix6",
                revision="rev-x",
                hub_probe=False,
            )
        assert out.kind == "hf"
        assert out.revision == "rev-x"

    def test_repo_with_dots_in_name_accepted(self):
        # Real HF repos often have dots, e.g. Qwen/Qwen2.5-7B
        with patch("tmm_asr.eval._dora_ckpt.HfApi", self._fake_hf_api(ok=True)):
            out = resolve_dora_checkpoint("Qwen/Qwen2.5-7B", revision=None)
        assert out.kind == "hf"

    def test_default_revision_of_the_paper_adapter(self):
        """Wired-up sanity: the constant in ft_merge/ft_holdout matches ours."""
        from tmm_asr.eval.ft_holdout import DORA_REVISION as R2
        from tmm_asr.eval.ft_merge import DORA_REVISION as R1
        # These are what the resolver will forward on the HF branch.
        assert R1 == R2 == "ad9144916cf661ea2ef462ad273077343c3d803d"


# Return type

class TestReturnShape:

    def test_resolved_is_frozen_dataclass(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        out = resolve_dora_checkpoint(str(d), revision=None, hub_probe=False)
        assert isinstance(out, ResolvedCheckpoint)
        with pytest.raises(Exception):  # frozen dataclass rejects assignment
            out.kind = "hf"


class TestPathAmbiguity:
    """Existing local path prefixes take precedence over Hub IDs."""

    def test_bare_path_with_existing_parent_dir_is_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()
        # The existing parent makes this an unambiguous local reference.
        with patch("tmm_asr.eval._dora_ckpt.HfApi",
                   side_effect=AssertionError("Hub probe called on local-shaped path")):
            with pytest.raises(FileNotFoundError, match="looks like a local path"):
                resolve_dora_checkpoint("checkpoints/run1", revision="x")

    def test_bare_path_without_parent_dir_is_hf(self, tmp_path, monkeypatch):
        """If the first segment doesn't exist locally, treat as HF id (probe)."""
        monkeypatch.chdir(tmp_path)
        # No `some-org/` local dir here → HF probe is legitimate.
        class _FakeApi:
            def __init__(self, *a, **kw): pass
            def repo_info(self, *a, **kw): return "ok"
        with patch("tmm_asr.eval._dora_ckpt.HfApi", _FakeApi):
            r = resolve_dora_checkpoint("some-org/some-repo", revision="rev1")
        assert r.kind == "hf"


class TestOfflineMode:
    """HF_HUB_OFFLINE=1 must skip the probe; from_pretrained handles cache."""

    def test_offline_env_var_skips_probe(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("tmm_asr.eval._dora_ckpt.HfApi",
                   side_effect=AssertionError("probe attempted in offline mode")):
            r = resolve_dora_checkpoint(
                "dylan01163104/whisper-medium-dora-mix6", revision="abc123",
            )
        assert r.kind == "hf"
        assert r.revision == "abc123"

    def test_offline_env_var_zero_still_probes(self, monkeypatch):
        """HF_HUB_OFFLINE=0 (or unset) must still perform the probe."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        called = []

        class _Api:
            def __init__(self, *args, **kwargs):
                pass

            def repo_info(self, *args, **kwargs):
                called.append(1)
                return "ok"

        with patch("tmm_asr.eval._dora_ckpt.HfApi", _Api):
            resolve_dora_checkpoint("some-org/some-repo", revision="r")
        assert called, "probe must run when offline mode is not enabled"


class TestHFValidationDelegatedToHub:
    """Repository IDs follow huggingface_hub's canonical validation."""

    def test_dot_in_org_accepted(self):
        """org.name/repo is a valid HF id."""
        class _Api:
            def __init__(self, *a, **kw): pass
            def repo_info(self, *a, **kw): return "ok"
        with patch("tmm_asr.eval._dora_ckpt.HfApi", _Api):
            r = resolve_dora_checkpoint("org.name/repo", revision="r")
        assert r.kind == "hf"

    @pytest.mark.parametrize("bad_id", [
        "org/repo--bad",   # double-dash forbidden
        "org/repo.git",    # .git suffix forbidden
        "org/repo.",       # trailing dot forbidden
    ])
    def test_hf_invalid_ids_rejected(self, bad_id):
        # HfApi must not even be reached — validation catches these upfront.
        with patch("tmm_asr.eval._dora_ckpt.HfApi",
                   side_effect=AssertionError("HfApi reached on invalid id")):
            with pytest.raises(ValueError):
                resolve_dora_checkpoint(bad_id, revision="r")
