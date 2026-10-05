"""Native Codex authorization/accounting probe; never a scored robot episode."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manipisa.evaluation.native_agent import run_native_agent
from manipisa.evaluation.programs import EpisodeStopped, execute_program, public_builtins
from manipisa.evaluation.telemetry import codex_costs


class Probe:
    started = terminal_time = reason = None
    wall_limit = 120
    first_observation = [{"type": "text", "text": "Connectivity probe only; no robot is present."}]

    def __init__(self, out):
        self.out = out
        self.calls = []
        self.namespace = {"__builtins__": public_builtins()}

    def begin(self):
        self.started = time.perf_counter()

    def stop(self, reason):
        if self.reason is None:
            self.reason, self.terminal_time = reason, time.perf_counter()

    def check_time(self):
        if time.perf_counter() - self.started > self.wall_limit:
            self.stop("wall_timeout")
            raise EpisodeStopped(self.reason)

    def agent_instructions(self):
        return ("This is a connectivity/accounting probe, not a robot task. In order, call read_api with name='index', "
                "observe, and execute_python with code='print(1+1)'. Then finish with exactly PROBE_OK. "
                "Do not perform any other action. No simulation score is assigned.")

    def tool(self, name, arguments):
        self.calls.append(name)
        if name == "read_api":
            result = "Probe: print(1+1) is the only requested program."
        elif name == "observe":
            result = "Probe observation; robot_present=false"
        elif name == "execute_python":
            result = execute_program(arguments["code"], self.namespace, self.started + self.wall_limit)
        else:
            raise ValueError(name)
        return {"content": [{"type": "text", "text": result}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    episode = Probe(out)
    agent = run_native_agent(episode)
    costs = codex_costs(out / "agent-events.jsonl", agent["telemetry"],
        wall_time_s=episode.terminal_time - episode.started, process_completed=agent["process_completed"],
        task_start_utc=agent["task_start_utc"], task_end_utc=agent["task_end_utc"])
    result = {"scope": "tool_authorization_and_metering_probe_only", "calls": episode.calls, "costs": costs,
              "models": sorted({r["model"] for r in agent["telemetry"] if r.get("model")}),
              "agent": {k: v for k, v in agent.items() if k != "telemetry"}}
    (out / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
