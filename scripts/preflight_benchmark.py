"""Audit all release tasks without starting physics or a model request."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manipisa.evaluation.catalog import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ur5_wuji_codex.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmark-preflight.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = inventory(ROOT / "Bench2Dex", physics_dt=config["physics_dt"],
                       budget_scale=config["expert_budget_scale"], stride=config["policy_stride"])
    if result["task_count"] != config["expected_task_count"]:
        raise ValueError("Release catalog differs from configured task count")
    result["experiment_config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    ready = sum(t["top_level_assets_present"] for t in result["tasks"])
    print(f"{result['task_count']} tasks; {ready} have all top-level assets; physics not verified")
    print(args.output.resolve())
    return 0 if ready == result["task_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
