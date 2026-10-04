"""Compare the shared structured/RGB UR5 smoke trajectory and source revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("structured", type=Path)
parser.add_argument("rgb", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
reports = [json.loads((p/"report.json").read_text(encoding="utf-8")) for p in (args.structured,args.rgb)]
traces = [[json.loads(x) for x in (p/"trace.jsonl").read_text(encoding="utf-8").splitlines()] for p in (args.structured,args.rgb)]
n = min(map(len,traces))
metrics = {}
def metric(key,extract):
    a,b = [np.asarray([extract(x) for x in rows[:n]],dtype=float) for rows in traces]
    metrics[key] = {"maximum_absolute_difference":float(np.max(np.abs(a-b))), "within_tolerance":bool(np.allclose(a,b,atol=1e-5,rtol=1e-5))}
metric("simulation_time_s",lambda x:x["state"]["timestamp"])
for group,fields in (("joints",("position","velocity")),("frames",("position","orientation_wxyz","linear_velocity","angular_velocity"))):
    for field in fields:
        metric(group+"_"+field,lambda x,g=group,f=field:[v for k in sorted(x["state"][g]) for v in x["state"][g][k][f]])
hashes = reports[0]["source_sha256"]
hashes_match = hashes == reports[1]["source_sha256"] and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
outcomes = [[(x["operation"],x["status"],x["reason"]) for x in r["feedback"]["instructions"]] for r in reports]
result = {"scope":"One paired UR5+Wuji free-space run; not a task success score", "trace_lengths":list(map(len,traces)),
          "common_physics_steps":n,"source_hashes_match_current_code":hashes_match,"instruction_outcomes_match":outcomes[0]==outcomes[1],
          "both_runs_passed":all(r["passed"] for r in reports),"metrics":metrics}
result["paired_check_passed"] = result["both_runs_passed"] and hashes_match and outcomes[0]==outcomes[1] and all(v["within_tolerance"] for v in metrics.values())
args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["paired_check_passed"] else 1)
