"""Summarize a saved development plan without dropping missing episodes."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(run):
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    methods = defaultdict(list)
    pairs = defaultdict(dict)
    for entry in plan["episodes"]:
        report = None
        if entry.get("report"):
            report = json.loads((run / entry["report"]).read_text(encoding="utf-8"))
        episode = (report or {}).get("episode", report or {})
        official, cost, score = (episode.get(k, {}) for k in ("official", "costs", "score"))
        valid = official.get("evaluation_valid") is True
        row = {"episode_id": entry["episode_id"], "task": entry["task"], "status": entry["status"],
               "reason": episode.get("reason"), "Overall": score.get("Overall"),
               "SR": int(official["stable_success"]) if valid else None,
               "SafeSR": int(official["stable_success"] and not official["safety_hard_violation"]) if valid else None,
               "LSCR": official.get("latched_stage_completion_rate") if valid else None,
               "success_simulation_s": official.get("time_to_stable_success_s") if valid and official.get("stable_success") else None,
               "wall_time_s": cost.get("wall_time_s", episode.get("wall_time_s")),
               "native_total_tokens": cost.get("native_total_tokens") if cost.get("tokens_complete") else None,
               "model_request_count": cost.get("model_request_count") if cost.get("requests_complete") else None,
               "score_issues": score.get("issues", []), "error": episode.get("error", (report or {}).get("error")),
               "initialization_sha256": episode.get("initialization_sha256")}
        methods[entry["method"]].append(row)
        pairs[entry["task"]][entry["method"]] = row
    summary = {"scope": plan["scope"], "planned_episodes": len(plan["episodes"]), "methods": {}, "paired_deltas": []}
    for method, rows in methods.items():
        result = {"planned": len(rows), "scored": sum(r["Overall"] is not None for r in rows),
                  "evaluated": sum(r["SR"] is not None for r in rows), "episodes": rows}
        for field in ("Overall", "SR", "SafeSR", "LSCR", "wall_time_s", "native_total_tokens", "model_request_count", "success_simulation_s"):
            available = [r[field] for r in rows if r[field] is not None]
            by_task = defaultdict(list)
            for row in rows:
                if row[field] is not None:
                    by_task[row["task"]].append(row[field])
            available_macro = mean(mean(values) for values in by_task.values()) if by_task else None
            complete = len(available) == len(rows)
            result[field] = {"value": available_macro if complete or field == "success_simulation_s" else None,
                             "available_only_mean": available_macro, "measured": len(available),
                             "coverage": len(available) / len(rows) if rows else 0}
        summary["methods"][method] = result
    for task, rows in pairs.items():
        left, right = rows.get("direct"), rows.get("manipisa")
        if not left or not right:
            continue
        matched = left["initialization_sha256"] is not None and left["initialization_sha256"] == right["initialization_sha256"]
        delta = right["Overall"] - left["Overall"] if matched and left["Overall"] is not None and right["Overall"] is not None else None
        summary["paired_deltas"].append({"task": task, "initialization_matched": matched, "ManipISA_minus_Direct": delta})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = summarize(args.run.resolve())
    output = args.run / "summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output.resolve())
    for name, value in result["methods"].items():
        print(f'{name}: {value["evaluated"]}/{value["planned"]} evaluated, {value["scored"]}/{value["planned"]} scored; Overall={value["Overall"]["value"]}')


if __name__ == "__main__":
    main()
