"""Run the authorized 26 x 2 development matrix, preserving every attempt."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manipisa.evaluation.catalog import inventory, sha256
from manipisa.evaluation.budgets import EpisodeBudget
from manipisa.evaluation.processes import stop_process_tree


def source_manifest():
    files = list((ROOT / "manipisa").rglob("*.py"))
    files += [ROOT / "examples/bench2dex_episode.py", Path(__file__).resolve(), ROOT / "configs/ur5_wuji_codex.json"]
    files += [ROOT / "docs" / name for name in (
        "manipisa-v0.2-core-contracts.md", "concise-feedback.md", "runtime-quickstart.md")]
    for component in ("benchmark", "build", "collector", "utils", "robots", "success"):
        files += list((ROOT / "Bench2Dex" / component).rglob("*.py"))
    files += list((ROOT / "Bench2Dex/scenes").glob("*.yaml"))
    files += [ROOT / "Bench2Dex/configs/collect/default.yaml"]
    files += [ROOT / "IsaacLab/source/isaaclab/isaaclab/controllers" / name
              for name in ("differential_ik.py", "differential_ik_cfg.py")]
    files += [ROOT / "dex2bench_dataset/Robots_p/ur5+wuji/usd/Multi_UR5_wuji_with_flange.usd"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def wait_for_wsl_cleanup(episode_dir, timeout_s=60):
    """Never overlap a new paired episode with an orphaned model process."""
    metadata = episode_dir / "wsl-agent.json"
    if not metadata.exists() and not (episode_dir / "agent-launch.json").exists():
        return True  # The scene failed before a model was launched.
    deadline = time.monotonic() + timeout_s
    while True:
        if metadata.exists():
            try:
                state = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if state.get("phase") == "finished" and state.get("cleanup", {}).get("complete") is True:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", help="Optional one-task diagnostic, e.g. 27; not the full matrix")
    parser.add_argument("--first-task", help="Run this release task first while retaining all 26 tasks")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Record host execution timings for both methods")
    parser.add_argument("--initializations", type=Path, help="Read-only directory of existing paired initialization snapshots")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/ur5_wuji_codex.json").read_text(encoding="utf-8"))
    budget = EpisodeBudget.from_config(config)
    catalog = inventory(ROOT / "Bench2Dex", physics_dt=budget.physics_dt,
                        budget_scale=budget.expert_budget_scale, stride=budget.policy_stride)
    tasks = catalog["tasks"]
    if args.task:
        tasks = [t for t in tasks if t["task"].split("_")[0] == args.task]
        if not tasks:
            parser.error("Task must be one of the release catalog IDs")
    if args.first_task:
        if args.task or not any(t["task"].split("_")[0] == args.first_task for t in tasks):
            parser.error("--first-task requires the full catalog and a release task ID")
        tasks.sort(key=lambda t: t["task"].split("_")[0] != args.first_task)
    out = (args.output or ROOT / "artifacts" / ("benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen_sources = source_manifest()
    rng = random.Random(20261005)
    plan = []
    for task in tasks:
        methods = ["direct", "manipisa"]
        rng.shuffle(methods)
        if args.check_only:
            methods = ["manipisa"]
        for method in methods:
            plan.append({"episode_id": f'{task["task"]}-{method}-000', "task": task["task"],
                         "method": method, "seed": 20261005 + int(task["task"].split("_")[0]),
                         "scene_sha256": task["scene_sha256"], "physics_step_budget": task["physics_steps"],
                         "wall_time_limit_s": budget.wall_limit_s, "status": "planned"})
    if args.initializations:
        import shutil
        for task in tasks:
            source = args.initializations.resolve() / (task["task"] + ".json")
            if not source.is_file():
                raise FileNotFoundError(f"Missing paired initialization: {source}")
            target = out / "initializations" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError("Initialization copy checksum mismatch")
    metadata = {"config": config, "budget_config": budget.to_dict(), "catalog": catalog, "sources": frozen_sources, "episodes": plan,
                "scope": "scene_checks" if args.check_only else "development_single_base_layout_per_task",
                "initializations_source": str(args.initializations.resolve()) if args.initializations else None,
                "profile_enabled": args.profile}
    assets_path = ROOT / "artifacts/benchmark-asset-manifest.json"
    if assets_path.exists():
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        metadata["assets"] = {"manifest_sha256": sha256(assets_path), "revision": assets["revision"],
                              "verified": assets.get("verified", False), "file_count": assets["file_count"]}
    def save():
        temp = out / "plan.tmp"
        temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(out / "plan.json")
        if not args.check_only:
            from summarize_benchmark import summarize
            summary_temp = out / "summary.tmp"
            summary_temp.write_text(json.dumps(summarize(out), indent=2, ensure_ascii=False), encoding="utf-8")
            summary_temp.replace(out / "summary.json")
    save()
    print(out, flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES")
    for index, entry in enumerate(plan):
        if not args.check_only and source_manifest() != frozen_sources:
            metadata["source_integrity"] = "changed; remaining episodes not run"
            save()
            raise RuntimeError("Experiment source changed during suite; remaining episodes preserved as planned")
        entry["status"] = "starting"
        if args.check_only:
            # Initialization probes are not paired policy comparisons; retain
            # each probe's version while permitting independent metering fixes.
            entry["sources"] = source_manifest()
        save()
        episode_dir = out / entry["episode_id"]
        snapshot = out / "initializations" / (entry["task"] + ".json")
        command = [sys.executable, str(ROOT / "examples/bench2dex_episode.py"), "--task", entry["task"] + ".yaml",
                   "--method", entry["method"], "--seed", str(entry["seed"]), "--headless",
                   "--snapshot", str(snapshot), "--output", str(episode_dir), "--wall-limit", str(budget.wall_limit_s)]
        if args.check_only:
            command.append("--check-only")
        if args.profile:
            command.append("--profile")
        start = time.perf_counter()
        with (out / (entry["episode_id"] + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                entry["process_id"] = process.pid
                save()
                entry["exit_code"] = process.wait(timeout=budget.process_timeout_s if not args.check_only else 480)
            except subprocess.TimeoutExpired:
                stop_process_tree(process)
                entry["exit_code"] = None
                entry["status"] = "process_timeout"
        entry["process_wall_s"] = time.perf_counter() - start
        report_path = episode_dir / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            entry["status"] = report.get("status", "completed")
            entry["reason"] = report.get("reason")
            entry["report"] = report_path.relative_to(out).as_posix()
        elif entry["status"] == "starting":
            entry["status"] = "missing_report"
        if not args.check_only and config["agent"].get("execution_platform") == "wsl":
            entry["agent_cleanup_verified"] = wait_for_wsl_cleanup(episode_dir)
            if not entry["agent_cleanup_verified"]:
                metadata["stop_reason"] = "WSL agent cleanup could not be verified; remaining episodes not launched"
                save()
                raise RuntimeError(metadata["stop_reason"])
        save()
        print(f'{index+1}/{len(plan)} {entry["episode_id"]}: {entry["status"]} {entry.get("reason")}', flush=True)
    metadata["source_integrity"] = "unchanged" if source_manifest() == frozen_sources else "changed"
    save()
    print("Suite finished; review plan.json for missing or invalid episodes", flush=True)


if __name__ == "__main__":
    main()
