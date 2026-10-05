"""One isolated Bench2Dex scene/episode using the fixed UR5+Wuji robot."""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Bench2Dex"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--method", choices=["direct", "manipisa"], required=True)
parser.add_argument("--seed", type=int, default=20261005)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--wall-limit", type=float, default=None, help="Override the configured episode wall budget")
parser.add_argument("--profile", action="store_true", help="Record host execution timings without extra CUDA synchronization")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
from utils.isaac_rendering import configure_headless_camera_parity_experience
configure_headless_camera_parity_experience(args, log_prefix="benchmark")
args.output = args.output.resolve()
args.snapshot = args.snapshot.resolve()
args.output.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args)
app = launcher.app


def main():
    from manipisa.evaluation.simulation import SimulationEpisode
    from manipisa.evaluation.native_agent import run_native_agent
    from manipisa.evaluation.telemetry import codex_costs
    from manipisa.evaluation.scoring import ScoreConfig, score_episode
    os.chdir(ROOT / "Bench2Dex")
    episode = None
    try:
        episode = SimulationEpisode(ROOT, ROOT / "Bench2Dex/scenes" / args.task, args.method,
                                    args.seed, args.snapshot, args.output, app, wall_limit=args.wall_limit,
                                    profile=args.profile)
        if args.check_only:
            report = {"status": "scene_ready", "task": episode.task_path.stem, "robot_key": episode.world["robot_key"],
                      "joint_count": episode.robot.num_joints, "camera_ids": list(episode.rig.camera_ids),
                      "initialization_sha256": episode.snapshot_hash, "binding": episode.adapter.describe_bindings(),
                      "runtime_environment": episode.runtime_environment, "budget_config": episode.budget_config.to_dict(),
                      "physics_step_budget": episode.budget,
                      "scope": "Scene/adapter/RGB initialization only; no model or task rollout"}
        else:
            agent = run_native_agent(episode)
            report = episode.result()
            report["costs"] = codex_costs(args.output / "agent-events.jsonl", agent["telemetry"],
                                         wall_time_s=report["wall_time_s"], process_completed=agent["process_completed"],
                                         task_start_utc=agent["task_start_utc"], task_end_utc=agent["task_end_utc"])
            report["agent"] = {k: v for k, v in agent.items() if k != "telemetry"}
            config = json.loads((ROOT / "configs/ur5_wuji_codex.json").read_text(encoding="utf-8"))
            report["score"] = score_episode(report["official"], report["costs"], ScoreConfig(**config["score"]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(args.output / "report.json"), "status": report.get("status", report.get("reason"))}), flush=True)
        return 0
    except BaseException as exc:
        error = {"status": "infrastructure_error" if episode is None or episode.started is None else "episode_error",
                 "task": args.task, "method": args.method, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        if episode is not None and episode.started is not None:
            episode.error = error["error"]
            episode.stop("error")
            error["episode"] = episode.result()
        (args.output / "report.json").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
        traceback.print_exc()
        return 1


exit_code = main()
sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
