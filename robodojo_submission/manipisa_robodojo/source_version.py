"""Load cuRobo's exact pinned source when Windows git version discovery fails."""
import importlib
import os
from pathlib import Path
import re


def load_curobo(checkout):
    checkout = Path(checkout).resolve()
    dotgit = checkout / ".git"
    gitdir = dotgit
    if dotgit.is_file():
        pointer = dotgit.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError("Invalid cuRobo git pointer")
        gitdir = (checkout / pointer.removeprefix("gitdir: ")).resolve()
    revision = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Use the official pinned detached cuRobo commit for reproducibility")
    try:
        module = importlib.import_module("curobo")
        fallback = False
    except LookupError:
        # This is local build metadata, explicitly not a claimed upstream release.
        # The exact commit remains the source of truth. The override exists only
        # during this package import and cannot affect other packages' versions.
        name = "SETUPTOOLS_SCM_PRETEND_VERSION"
        original = os.environ.get(name)
        os.environ[name] = "0.0.0+robodojo.g" + revision[:12]
        try:
            module = importlib.import_module("curobo")
        finally:
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
        fallback = True
    return {"source_revision": revision, "metadata_version": module.__version__,
            "local_metadata_fallback": fallback,
            "scope": "Version discovery only; no kinematics, physics, or scoring changes"}
