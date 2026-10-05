"""Release-catalog and local asset preflight, without importing Isaac Sim."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

ROBOT_KEY = "multi_ur5_wuji_with_flange"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_tasks(bench_root: Path) -> list[Path]:
    catalog = (bench_root / "scenes/task.md").read_text(encoding="utf-8")
    names = re.findall(r"^\|\s*\d+\s*\|[^\n]*`([^`]+\.yaml)`", catalog, flags=re.M)
    if not names or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate release task entries")
    scenes = (bench_root / "scenes").resolve()
    paths = [(scenes / name).resolve() for name in names]
    if any(p.parent != scenes or not p.is_file() for p in paths):
        raise ValueError("Catalog contains a missing task or a path outside scenes")
    return paths


def inventory(bench_root: Path, *, physics_dt=1 / 60, budget_scale=1.5, stride=3) -> dict:
    import yaml
    if not math.isfinite(physics_dt) or physics_dt <= 0 or not math.isfinite(budget_scale) or budget_scale <= 0:
        raise ValueError("dt and budget_scale must be finite and positive")
    if type(stride) is not int or stride <= 0:
        raise ValueError("stride must be a positive integer")
    tasks = release_tasks(bench_root)
    rows, prefixes = [], set()
    asset_root = (bench_root.parent / "dex2bench_dataset").resolve()
    for path in tasks:
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        metrics = task.get("metrics", {})
        assets = []
        for key, spec in task.get("assets", {}).items():
            asset = (path.parent / spec["path"]).resolve()
            rel = asset.relative_to(asset_root)
            if len(rel.parts) < 2 or rel.parts[0] != "Objects":
                raise ValueError(f"Unexpected asset layout: {rel}")
            prefix = "/".join(rel.parts[:2])
            prefixes.add(prefix)
            assets.append({"key": key, "path": rel.as_posix(), "present": asset.is_file(), "prefix": prefix})
        expert = metrics.get("expert_time_step")
        if not isinstance(expert, (int, float)) or expert <= 0 or not math.isfinite(expert):
            raise ValueError(f"Missing positive expert budget: {path.name}")
        steps = math.ceil(expert * budget_scale / stride) * stride
        rows.append({"task": path.stem, "scene": path.relative_to(bench_root).as_posix(),
                     "scene_sha256": sha256(path), "description": task.get("description"),
                     "robot_key": ROBOT_KEY, "objects": len(task.get("objects", [])),
                     "stage_count": len(metrics.get("stages", [])),
                     "has_terminal": bool(metrics.get("terminal", {}).get("raw_condition")),
                     "expert_physics_steps": expert, "physics_steps": steps,
                     "physics_dt": physics_dt, "simulation_budget_s": steps * physics_dt,
                     "assets": assets, "top_level_assets_present": all(a["present"] for a in assets),
                     "physics_validation": "not_run"})
    return {"task_count": len(rows), "robot_key": ROBOT_KEY,
            "task_catalog_sha256": sha256(bench_root / "scenes/task.md"),
            "excluded_extra_yaml": sorted(p.name for p in (bench_root / "scenes").glob("*.yaml") if p.resolve() not in tasks),
            "asset_prefixes": sorted(prefixes), "tasks": rows}
