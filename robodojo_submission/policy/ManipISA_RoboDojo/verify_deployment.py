"""Verify pinned core/evaluator sources before the standard policy server starts."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
policy = Path(__file__).resolve().parent
pointer = policy / "submission-root.json"
if not pointer.is_file():
    raise SystemExit("Register this policy from the independent submission checkout first")
submission = Path(json.loads(pointer.read_text(encoding="utf-8"))["submission_root"]).resolve()
sys.path[:0] = [str(submission), str(submission.parent)]
from manipisa_robodojo.provenance import verify_core_tree, verify_official_checkouts

if __name__ == "__main__":
    print(json.dumps({"core": verify_core_tree(),
                      "official": verify_official_checkouts(policy.parents[2])}, indent=2))
