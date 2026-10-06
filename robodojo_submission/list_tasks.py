#!/usr/bin/env python3
"""Read the pinned official task/score definitions and emit a full evaluation plan.

This inventories configuration only. It neither launches tasks nor reports scores.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import yaml

WORK = Path(__file__).resolve().parent.parent
PINNED_REVISION = "266130a3ec41ba9b20b0e5648d2f3542f219c06c"
INVENTORY = "scripts/internal/task_inventory.py"
SUMMARY = "scripts/internal/summarize_result.py"
INDEX = "task/RoboDojo/config/_task.yml"
SWEEP = "scripts/internal/smoke_all_tasks.sh"
WINDOWS_BLOCKER = (
    "Non-empty Articulation configuration enables the official PhysX monitor, "
    "which imports Linux-only fcntl; the current Windows launcher blocks this task."
)


def literal_assignment(path, name):
    """Read constants without executing Isaac-dependent upstream modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(value)
    raise ValueError(f"Required official literal {name} not found in {path}")


def read_yaml(path):
    result = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def positive_count(value, source):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Expected positive integer episode count in {source}")
    return value


def build_inventory(root):
    root = root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"Expected pinned revision {PINNED_REVISION}; found {revision}")

    dimensions = literal_assignment(root / INVENTORY, "DIMENSION_TASKS")
    score_dimensions = literal_assignment(root / SUMMARY, "DIMENSIONS")
    seeds = literal_assignment(root / SUMMARY, "EXPECTED_SEEDS")
    standalone_count = positive_count(
        literal_assignment(root / SUMMARY, "STANDALONE_EPISODES"), SUMMARY)
    half_count = positive_count(
        literal_assignment(root / SUMMARY, "PAIRED_HALF_EPISODES"), SUMMARY)
    canonical_names = [task for names in dimensions.values() for task in names]
    score_names = {task for names in score_dimensions.values() for task in names}
    if len(canonical_names) != 42 or len(set(canonical_names)) != 42 or set(canonical_names) != score_names:
        raise ValueError("Official task inventory and score aggregator disagree on the 42 canonical tasks")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Invalid EXPECTED_SEEDS in official score aggregator")

    index = read_yaml(root / INDEX)
    common = index.get("common", {})
    overrides = index.get("tasks", {})
    # This matches utils/pipeline_utils.py::process_config, including its fallback.
    default_count = positive_count(common.get("eval_nums", 50), INDEX)
    sweep_text = (root / SWEEP).read_text(encoding="utf-8")
    seed_match = re.search(r'^seed="(\d+)"$', sweep_text, re.MULTILINE)
    if seed_match is None:
        raise ValueError("Cannot identify the official sweep's default seed")

    source_paths = {INVENTORY, SUMMARY, INDEX, SWEEP, "scripts/README.md",
                    "utils/pipeline_utils.py", "src/eval_client/main.py",
                    "src/eval_client/physx_warning_monitor.py"}
    tasks, configs, matrix = [], [], []
    for dimension, names in dimensions.items():
        for canonical in names:
            paired = dimension == "generalization"
            names_for_task = [canonical, canonical + "_random"] if paired else [canonical]
            task_configs = []
            for name in names_for_task:
                config_path = f"task/RoboDojo/config/{name}.yml"
                module_path = f"task/RoboDojo/tasks/{name}.py"
                cfg = read_yaml(root / config_path)
                module_tree = ast.parse((root / module_path).read_text(encoding="utf-8"))
                class_exists = any(isinstance(node, ast.ClassDef) and node.name == name
                                   for node in module_tree.body)
                if not class_exists:
                    raise ValueError(f"Official task class missing in {module_path}")
                settings = overrides.get(name, {})
                native_count = positive_count(settings.get("eval_nums", default_count), INDEX)
                required_count = half_count if paired else standalone_count
                if native_count != required_count:
                    raise ValueError(f"Native count and summary requirement disagree for {name}")
                needs_articulation = bool(cfg.get("Articulation"))
                record = {
                    "name": name,
                    "canonical_task": canonical,
                    "dimension": dimension,
                    "variant": "random" if name.endswith("_random") else "standard",
                    "config": config_path,
                    "module": module_path,
                    "official_task_class_present": class_exists,
                    "native_eval_nums": native_count,
                    "eval_nums_source": (f"{INDEX}:tasks.{name}.eval_nums" if "eval_nums" in settings
                                         else f"{INDEX}:common.eval_nums" if "eval_nums" in common
                                         else "utils/pipeline_utils.py:process_config fallback 50"),
                    "robot_config": settings.get("robot_config", common.get("robot_config", "dual_x5")),
                    "needs_articulation": needs_articulation,
                    "windows_known_blocker": WINDOWS_BLOCKER if needs_articulation else None,
                    "physics_validation": "pending; absence of this blocker does not establish Windows support",
                    "required_seeds": seeds,
                }
                configs.append(record)
                task_configs.append(record)
                source_paths.update((config_path, module_path))
                for seed in seeds:
                    matrix.append({
                        "task": name, "canonical_task": canonical, "seed": seed,
                        "native_eval_nums": native_count, "EVAL_NUM": "native",
                        "windows_known_blocker": record["windows_known_blocker"],
                        "windows_launcher_args": ["--task", name, "--seed", str(seed), "--full"],
                        "run_status": "planned_only", "official_score": None,
                    })
            tasks.append({
                "name": canonical,
                "dimension": dimension,
                "required_configs": names_for_task,
                "summary_rule": "25 standard + 25 random per seed" if paired else "50 episodes per seed",
                "native_eval_nums_by_config": {item["name"]: item["native_eval_nums"] for item in task_configs},
                "episodes_per_seed": sum(item["native_eval_nums"] for item in task_configs),
                "required_seeds": seeds,
                "total_required_episodes": sum(item["native_eval_nums"] for item in task_configs) * len(seeds),
                "windows_blocked_configs": [item["name"] for item in task_configs if item["windows_known_blocker"]],
            })

    counts = {
        "canonical_tasks": len(tasks), "runnable_configs": len(configs),
        "random_variant_configs": sum(item["variant"] == "random" for item in configs),
        "dimension_canonical_counts": dict(Counter(item["dimension"] for item in tasks)),
        "expected_seeds": seeds, "task_seed_summary_cells": len(tasks) * len(seeds),
        "config_seed_runs": len(matrix),
        "episodes_per_seed": sum(item["native_eval_nums"] for item in configs),
        "total_required_episodes": sum(item["native_eval_nums"] for item in matrix),
        "windows_blocked_canonical_tasks": sum(bool(item["windows_blocked_configs"]) for item in tasks),
        "windows_blocked_configs": sum(bool(item["windows_known_blocker"]) for item in configs),
    }
    return {
        "scope": "Read-only configuration inventory and planned coverage; no experiment results",
        "official_score": None, "robodojo_root": str(root), "robodojo_revision": revision,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sorted(source_paths)},
        "counts": counts,
        "official_sweep_default_seed": int(seed_match.group(1)),
        "seed_scheduling_note": "The official sweep accepts one --seed per invocation; explicitly schedule every EXPECTED_SEEDS value.",
        "random_variant_rule": "Generalization requires both standard and _random runs; _random is merged into its canonical task, not counted as another canonical task.",
        "summary_entrypoint": "scripts/robodojo.sh summarize",
        "summary_behavior": [
            "Uses the latest result timestamp for each task/policy/embodiment/seed.",
            "Requires the native episode count before filling a task-seed score cell.",
            "Partial averages are not complete 126-cell coverage.",
            "Checks head, left_wrist and right_wrist videos for included result records.",
        ],
        "coverage_plan": [
            {"stage": 1, "action": "Complete cross-system transport and one official stack_bowls diagnostic episode; diagnose CUDA/rendering/physics failures before a sweep."},
            {"stage": 2, "action": "Prepare official assets/layouts for all 54 configurations and all three seeds; resolve the seven Windows-blocked configurations using an environment that supports the unchanged official monitor."},
            {"stage": 3, "action": "Run every run_matrix entry with EVAL_NUM=native, preserving task-specific counts, layouts, task rules, and three-camera evidence."},
            {"stage": 4, "action": "Run the unmodified official summary script and confirm all 126 task-seed cells, including 25+25 generalization halves. Report incomplete coverage explicitly."},
        ],
        "canonical_tasks": tasks,
        "runnable_configs": configs,
        "run_matrix": matrix,
    }


def markdown(inventory):
    counts = inventory["counts"]
    lines = [f"Pinned RoboDojo: `{inventory['robodojo_revision']}`", "",
             f"{counts['canonical_tasks']} canonical tasks; {counts['runnable_configs']} configs; "
             f"seeds {counts['expected_seeds']}; {counts['total_required_episodes']} planned episodes. "
             "No experiment results or scores are inferred.", "",
             "| Canonical task | Dimension | Native counts per seed | Windows Articulation blocker |",
             "| --- | --- | --- | --- |"]
    for task in inventory["canonical_tasks"]:
        count_text = ", ".join(f"{name}: {count}" for name, count in task["native_eval_nums_by_config"].items())
        blocked = ", ".join(task["windows_blocked_configs"]) or "None identified; physics unverified"
        lines.append(f"| {task['name']} | {task['dimension']} | {count_text} | {blocked} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, default=WORK / "RoboDojo")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write the JSON inventory to a new file; refuse overwrite")
    args = parser.parse_args()
    inventory = build_inventory(args.robodojo)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(inventory, stream, indent=2)
            stream.write("\n")
    print(json.dumps(inventory, indent=2) if args.format == "json" else markdown(inventory), end="\n")


if __name__ == "__main__":
    main()
