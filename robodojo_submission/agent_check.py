"""Check real WSL Codex tools against synthetic observations, without physics."""
from copy import deepcopy
import json
from pathlib import Path
import time

from manipisa_robodojo.model import Policy
from manipisa_robodojo.runtime import resolve_codex, result_path
from protocol_check import observation


def check(output, config=None):
    out = result_path(output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee", "eval_batch": False,
           "reasoning_effort": "high", "agent_wall_limit_s": 180,
           "agent_startup_timeout_s": 180, "action_timeout_s": 370}
    cfg.update(config or {})
    cfg["artifact_dir"] = str(out / "episodes")
    settings = resolve_codex(cfg)
    model = Policy(cfg)
    obs = observation()
    obs["instruction"] = (
        "This is an explicitly synthetic protocol fixture, not a RoboDojo benchmark task. "
        "Check the tool APIs, command the right tool to move +0.008 meters on world X from its observed "
        "pose using a ManipISA MOVE, then step until its contract completes and finish your response. "
        "The images are constant color test patterns; do not infer physical objects from them. "
        "Do not claim physical or benchmark success.")
    actions = []
    start = time.monotonic()
    error = None
    try:
        model.update_obs(obs)
        while time.monotonic() - start < 365:
            chunk = model.get_action()
            actions.append(deepcopy(chunk[0]))
            obs["state"].update(deepcopy(chunk[0]))
            model.update_obs(obs)
            if model._done.wait(.02):
                break
        if model._error:
            raise RuntimeError(model._error)
        if not model._done.is_set():
            raise TimeoutError("Live agent fixture did not finish")
        if not actions or actions[-1]["right_ee_pose"][0] <= .3:
            raise AssertionError("No MOVE was observed")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        before = time.monotonic()
        model.reset()
        report = {"passed": error is None, "error": error, **settings,
                  "action_count": len(actions), "reset_seconds": time.monotonic()-before,
                  "elapsed_seconds": time.monotonic()-start,
                  "scope": "Real WSL Codex with synthetic pose-echo observations; no physics or official score",
                  "official_score": None}
        (out / "check-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(check(parser.parse_args().output), indent=2))
