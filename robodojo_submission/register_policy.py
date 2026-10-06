"""Add a new XPolicyLab policy directory. Existing files are never overwritten."""
import argparse
import json
from pathlib import Path


def register(root):
    root = Path(root).resolve()
    if not (root / "setup_policy_server.py").is_file():
        raise ValueError("Expected an XPolicyLab checkout")
    source = Path(__file__).resolve().parent / "policy" / "ManipISA_RoboDojo"
    destination = root / "policy" / source.name
    files = {p.name: p.read_bytes() for p in source.iterdir()
             if p.is_file() and p.suffix != ".pyc"}
    # Deployment-local pointer, generated after cloning; never copies credentials.
    files["submission-root.json"] = (json.dumps({
        "submission_root": str(Path(__file__).resolve().parent)
    }, indent=2) + "\n").encode("utf-8")
    for name, data in files.items():
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = destination / name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xpolicylab", required=True)
    print(register(parser.parse_args().xpolicylab))
