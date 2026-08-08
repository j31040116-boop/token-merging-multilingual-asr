"""
Shared checkpoint-resolution helper for the DoRA evaluators
(ft_merge, ft_holdout).

Distinguishes three input shapes and fails FAST — before the caller loads the
several-second Whisper base model:

  1. local           an existing directory on disk
  2. hf              a valid HuggingFace `<org>/<repo>` id whose existence at
                     the pinned revision is confirmed by a HfApi.repo_info
                     probe (skipped in offline mode; see below)
  3. error           anything else — raised immediately

Ambiguity handling: `checkpoints/run1` and `some-org/some-repo` are
syntactically identical. We disambiguate by checking whether the first
segment names a directory in the current working directory:
  - `./checkpoints/` exists → `checkpoints/run1` is a broken LOCAL path
  - `./some-org/` does not exist → `some-org/some-repo` is treated as HF
Prefixing local relative paths with `./` also unambiguously routes local
(see _looks_like_local_path).

Validation of HF ids is delegated to `huggingface_hub.utils.validate_repo_id`
so we accept exactly what `from_pretrained` accepts (`org.name/repo` yes,
`org/repo--bad`, `org/repo.`, `org/repo.git` no).

Offline mode: when `HF_HUB_OFFLINE=1` is set in the environment, the online
repo_info probe is skipped and the caller's `from_pretrained` uses the local
cache. This mirrors what huggingface_hub itself does in offline mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import HfApi  # module-level: tests can patch it
from huggingface_hub.errors import (
    GatedRepoError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils import HFValidationError, validate_repo_id


@dataclass(frozen=True)
class ResolvedCheckpoint:
    kind: str            # "local" or "hf"
    ident: str           # local dir path OR HF repo id
    revision: Optional[str]   # None for local, pinned SHA for hf


def _offline_mode() -> bool:
    """
    True iff HF_HUB_OFFLINE is set to a truthy value in the current process
    env. Read from os.environ rather than huggingface_hub.constants.HF_HUB_OFFLINE
    because the constant is fixed at import time — monkeypatched env vars in
    tests wouldn't otherwise take effect.
    """
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")


def _looks_like_local_path(ckpt: str) -> bool:
    """
    True iff the input syntactically looks like a filesystem path (as opposed
    to a pure HF org/repo id).

    Catches:
      - absolute paths (`/…`, `~…`)
      - dot-prefixed relative paths (`./…`, `../…`)
      - windows backslash paths
      - any component equal to `..`
      - bare relative paths whose first segment is an existing local directory
        (so `checkpoints/run1` errors out cleanly if `./checkpoints/` exists,
        rather than being probed against HF as an org/repo)
    """
    if ckpt.startswith(("/", "./", "../", "~")):
        return True
    if os.sep != "/" and os.sep in ckpt:
        return True
    if ".." in ckpt.split("/"):
        return True
    # Bare "foo/bar" ambiguity: if "foo" exists as a directory locally, treat
    # the whole thing as a local path (which will error out below because the
    # full target doesn't exist).
    parts = ckpt.split("/", 1)
    if len(parts) == 2 and os.path.isdir(parts[0]):
        return True
    return False


def resolve_dora_checkpoint(
    ckpt: str,
    revision: Optional[str],
    *,
    hub_probe: bool = True,
) -> ResolvedCheckpoint:
    """
    Classify `ckpt` and (for HF ids) verify Hub reachability BEFORE the
    caller commits to loading the base Whisper model.

    Parameters
    ----------
    ckpt : str
        `--checkpoint` argument (local dir path OR HF `<org>/<repo>`).
    revision : str | None
        Pinned HF revision (SHA or tag) to probe. Ignored for local paths.
    hub_probe : bool
        If True (default), reachability is verified via HfApi.repo_info().
        Set False in unit tests to avoid the network call.

    Returns
    -------
    ResolvedCheckpoint(kind, ident, revision)

    Raises
    ------
    FileNotFoundError
        Local-looking path with no existing directory, OR HF repo/revision
        not found on the Hub.
    ValueError
        Input matches neither an existing directory nor a valid HF id
        (per `huggingface_hub.utils.validate_repo_id`).
    """
    # 1) Existing local directory wins unambiguously.
    if os.path.isdir(ckpt):
        return ResolvedCheckpoint(kind="local", ident=ckpt, revision=None)

    # 2) Any input that syntactically looks like a filesystem path but doesn't
    #    exist is an early-fail — never send it to the Hub.
    if _looks_like_local_path(ckpt):
        raise FileNotFoundError(
            f"Checkpoint {ckpt!r} looks like a local path but no such directory "
            f"exists. If you meant a HuggingFace repo id, drop the leading path "
            f"prefix (e.g. 'user/repo-name'). If you meant a local dir, create it "
            f"first."
        )

    # 3) HF-id validity. Require the `<org>/<repo>` shape (bare names like
    #    `just-a-name` are technically valid HF ids but far more likely to be
    #    typoed local paths in our adapter workflow), then delegate to
    #    huggingface_hub for the syntactic rules HF actually enforces
    #    (accepts `org.name/repo`; rejects `org/repo--bad`, `org/repo.git`,
    #    `org/repo.`, etc.).
    if "/" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt!r} is not a valid HuggingFace repo id: expected "
            f"the form '<org>/<repo>'. Bare names are not accepted for DoRA "
            f"adapters — always use '<owner>/<name>'."
        )
    try:
        validate_repo_id(ckpt)
    except HFValidationError as e:
        raise ValueError(
            f"Checkpoint {ckpt!r} is not a valid HuggingFace repo id: {e}"
        ) from e

    # 4) Optionally probe the Hub. Skip when offline mode is enabled — the
    #    caller's from_pretrained will use the cache.
    if hub_probe and not _offline_mode():
        try:
            HfApi().repo_info(ckpt, revision=revision, repo_type="model")
        except GatedRepoError as e:
            # GatedRepoError is a subclass of RepositoryNotFoundError, so it
            # must be caught first to preserve authentication semantics.
            raise PermissionError(
                f"HuggingFace repo {ckpt!r} exists but access is gated. "
                f"Log in (`huggingface-cli login`) and accept the model's "
                f"terms on its Hub page, then retry. If you have already "
                f"accepted, ensure HF_TOKEN is exported in this shell."
            ) from e
        except RepositoryNotFoundError as e:
            raise FileNotFoundError(
                f"HuggingFace repo {ckpt!r} not found on the Hub. Check the "
                f"org/repo spelling, or set HF_HUB_OFFLINE=1 if it is cached "
                f"locally but the network is unavailable."
            ) from e
        except RevisionNotFoundError as e:
            raise FileNotFoundError(
                f"HuggingFace repo {ckpt!r} exists but revision "
                f"{revision!r} is not present."
            ) from e
        except OfflineModeIsEnabled:
            # HF_HUB_OFFLINE became active between our check and the API call.
            # Skip the probe; let from_pretrained do cache lookup.
            pass
        # Let authentication, network, and other HTTP errors propagate.

    return ResolvedCheckpoint(kind="hf", ident=ckpt, revision=revision)
