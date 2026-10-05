"""Bounded, model-free profiling of the actual Bench2Dex execution path."""
from pathlib import Path
import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import sys
import time
import traceback

CODE_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-root", type=Path, default=CODE_ROOT)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--task", default="27_ball_box_loading.yaml")
parser.add_argument("--seed", type=int, default=20261032)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--reference-contacts", type=Path)
known, _ = parser.parse_known_args()
sys.path.insert(0, str(known.asset_root / "Bench2Dex"))
sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 10 or args.steps > 180 or args.steps % 10:
    parser.error("--steps must be a multiple of 10 between 10 and 180")
args.asset_root = args.asset_root.resolve()
args.output = args.output.resolve()
args.snapshot = args.snapshot.resolve()
if args.reference_contacts:
    args.reference_contacts = args.reference_contacts.resolve()
    if not args.reference_contacts.is_file():
        parser.error("--reference-contacts must name an existing reference implementation")
if not args.snapshot.is_file():
    parser.error("Use an existing paired initialization snapshot")
args.output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = True
from utils.isaac_rendering import configure_headless_camera_parity_experience
configure_headless_camera_parity_experience(args, log_prefix="runtime-profile")
launcher = AppLauncher(args)
app = launcher.app


def main():
    import torch
    from manipisa import Instruction, Opcode, PoseGoal
    from manipisa.evaluation.simulation import SimulationEpisode
    os.chdir(args.asset_root / "Bench2Dex")
    episode = SimulationEpisode(args.asset_root, args.asset_root / "Bench2Dex/scenes" / args.task,
        "manipisa", args.seed, args.snapshot, args.output, app, profile=True)
    profiler = episode.profiler
    episode.begin()
    # Warm the exact runtime/evaluator path before collecting steady-state timings.
    episode.step(10)
    profiler.reset()
    reports = []
    for actor, frame in (("arm_a", "tool_a"), ("arm_b", "tool_b")):
        pose = episode.runtime._snapshot.frames[frame]
        goal = PoseGoal(frame, (pose.position[0], pose.position[1], pose.position[2] + 0.03),
                        pose.orientation_wxyz)
        reports.append(episode.runtime.submit(Instruction(Opcode.MOVE, (actor,), goal,
            max_joint_speed=3.0, timeout_s=5.0)).to_dict())
    samples = []
    detailed = cProfile.Profile()
    started = time.perf_counter()
    detailed.enable()
    for _ in range(args.steps // 10):
        before = time.perf_counter()
        with profiler.span("tool.total"):
            reply = episode.tool("execute_python", {"code": "step(10)"})
        result = json.loads(reply["content"][0]["text"])
        if reply.get("isError"):
            raise RuntimeError(result.get("error"))
        contacts = episode.runtime._snapshot.contacts
        samples.append({"physics_step": episode.steps, "wall_s": time.perf_counter() - before,
                        "present_contacts": sum(c.present for c in contacts.values()),
                        "invalid_contacts": sum(not c.valid for c in contacts.values()),
                        "invalid_wrenches": sum(not c.wrench_valid for c in contacts.values())})
    detailed.disable()
    elapsed = time.perf_counter() - started
    timings = profiler.snapshot()
    profiler.restore()
    detailed.dump_stats(str(args.output / "python-profile.pstats"))
    stream = io.StringIO()
    pstats.Stats(detailed, stream=stream).sort_stats("tottime").print_stats(45)
    (args.output / "python-profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    result = {"scope": "model_free_execution_diagnostic_not_a_benchmark_score",
              "task": args.task, "seed": args.seed, "steps_measured": args.steps,
              "warmup_steps": 10, "wall_s": elapsed, "steps_per_wall_second": args.steps / elapsed,
              "timing_caveat": "Host wall times include existing synchronization waits and cProfile overhead; no added CUDA synchronization in rollout.",
              "initialization_sha256": episode.snapshot_hash, "runtime_environment": episode.runtime_environment,
              "submitted": reports, "final_instructions": episode.runtime.feedback()["instructions"],
              "contact_binding_count": len(episode.adapter.contact_source.bindings),
              "samples": samples, "performance": timings,
              "source_sha256": {str(p.relative_to(CODE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (CODE_ROOT / "manipisa").rglob("*.py")}}
    if args.reference_contacts:
        import importlib.util
        spec = importlib.util.spec_from_file_location("reference_contacts", args.reference_contacts)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        source = episode.adapter.contact_source
        reference = module.PhysXContactSource.__new__(module.PhysXContactSource)
        reference.__dict__.update(source.__dict__)
        reference.__dict__.pop("observe", None)
        from dataclasses import asdict
        rows = []
        # Paired reads see exactly the same frozen PhysX state, in alternating order.
        # This extra synchronized microbenchmark is excluded from rollout timings.
        for i in range(10):
            torch.cuda.synchronize()
            row = {}; values = {}
            for name, reader in (("reference", reference), ("candidate", source)) if i % 2 == 0 else (("candidate", source), ("reference", reference)):
                reader._timestamp = None
                start = time.perf_counter()
                value = type(reader).observe(reader, float(episode.sim.current_time))
                row[name + "_s"] = time.perf_counter() - start
                values[name] = {k: asdict(v) for k, v in value.items()}
            if values["reference"] != values["candidate"]:
                raise AssertionError("Contact decoding differs on the same frozen physics state")
            rows.append(row)
        result["same_state_contact_comparison"] = {"exact_equal": True, "samples": rows,
            "present_contacts": sum(c.present for c in source._cache.values()),
            "reference_sha256": hashlib.sha256(args.reference_contacts.read_bytes()).hexdigest()}
    (args.output / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"profile": str(args.output / "profile.json"), "steps": args.steps, "wall_s": elapsed}), flush=True)


code = 0
try:
    main()
except BaseException:
    (args.output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    traceback.print_exc()
    code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
