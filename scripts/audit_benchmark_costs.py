"""Recompute cost evidence without replacing the originally emitted report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manipisa.evaluation.scoring import ScoreConfig, score_episode
from manipisa.evaluation.telemetry import codex_costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for entry in plan["episodes"]:
        if not entry.get("report"):
            continue
        original = run / entry.get("original_report", entry["report"])
        report = json.loads(original.read_text(encoding="utf-8"))
        if "agent" not in report or "official" not in report:
            continue
        agent = report["agent"]
        records = [json.loads(line) for line in (original.parent / "otel.jsonl").read_text(encoding="utf-8").splitlines()]
        report["costs"] = codex_costs(original.parent / "agent-events.jsonl", records,
            wall_time_s=report["wall_time_s"], process_completed=agent["process_completed"],
            task_start_utc=agent["task_start_utc"], task_end_utc=agent["task_end_utc"])
        report["score"] = score_episode(report["official"], report["costs"], ScoreConfig(**plan["config"]["score"]))
        report["audit"] = {"timestamp_utc": stamp, "original_report": original.name,
                           "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                           "parser_sha256": hashlib.sha256((ROOT / "manipisa/evaluation/telemetry.py").read_bytes()).hexdigest()}
        audited = original.with_name("report.audit-" + stamp + ".json")
        with audited.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        entry["original_report"] = original.relative_to(run).as_posix()
        entry["report"] = audited.relative_to(run).as_posix()
        print(entry["episode_id"], "Overall=", report["score"]["Overall"], report["costs"]["limitation"])
    with (run / ("plan.before-audit-" + stamp + ".json")).open("x", encoding="utf-8") as stream:
        stream.write(plan_path.read_text(encoding="utf-8"))
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
