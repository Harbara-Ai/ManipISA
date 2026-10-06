"""Read-only provenance checks for the isolated submission and official clients."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SUBMISSION = Path(__file__).resolve().parents[1]
PROJECT = SUBMISSION.parent


def profile():
    return json.loads((SUBMISSION / "submission.json").read_text(encoding="utf-8"))


def verify_core_tree():
    expected = json.loads((SUBMISSION / "core-files.sha256.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, digest in expected.items():
        path = PROJECT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(name)
    if mismatches:
        raise ValueError("Submission core differs from its pinned snapshot: " + ", ".join(mismatches))
    return {"core_revision": profile()["core_revision"], "verified_files": len(expected)}


def verify_official_checkouts(root):
    root = Path(root).resolve()
    settings = profile()
    revisions = {}
    for label, path, expected in (
        ("RoboDojo", root, settings["robodojo_revision"]),
        ("XPolicyLab", root / "XPolicyLab", settings["xpolicylab_revision"]),
    ):
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True, timeout=30).strip()
        if revision != expected:
            raise ValueError(f"{label}: expected {expected}, found {revision}")
        # The added policy is untracked in the official checkout. Tracked source
        # changes, including staged ones, must not silently change the evaluator.
        result = subprocess.run(["git", "-C", str(path), "diff", "--quiet",
                                 "--ignore-submodules=all", "HEAD", "--"], timeout=60)
        if result.returncode:
            raise ValueError(f"{label}: tracked official source differs from the pinned revision")
        revisions[label] = revision
    return revisions
