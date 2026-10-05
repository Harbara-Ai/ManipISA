"""Register policy files in an existing XPolicyLab checkout; never edit the evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def install(xpolicylab):
    root = Path(xpolicylab).resolve()
    if not (root / "setup_policy_server.py").is_file() or not (root / "policy").is_dir():
        raise ValueError("Expected the RoboDojo/XPolicyLab checkout root")
    source = Path(__file__).resolve().parents[1] / "integrations" / "xpolicylab" / "ManipISA"
    destination = root / "policy" / "ManipISA"
    files = sorted(p for p in source.iterdir() if p.is_file())
    # Check every destination before writing any file; refuse local edits.
    for file in files:
        target = destination / file.name
        if target.exists() and target.read_bytes() != file.read_bytes():
            raise FileExistsError(f"Different file already exists; review it before updating: {target}")
    destination.mkdir(exist_ok=True)
    for file in files:
        shutil.copyfile(file, destination / file.name)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xpolicylab", type=Path, required=True)
    print(install(parser.parse_args().xpolicylab))
