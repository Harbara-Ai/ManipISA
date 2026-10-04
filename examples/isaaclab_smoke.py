"""Physical UR5+Wuji smoke run. No demonstrations, task labels or tactile inputs.

Run from the project root using the installed Isaac Lab Python environment:
    python examples/isaaclab_smoke.py --headless
    python examples/isaaclab_smoke.py --headless --rgb
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Bench2Dex"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rgb", action="store_true", help="Attach Bench2Dex CameraRig and test the RGB return switch")
parser.add_argument("--output", type=Path)
parser.add_argument("--adapter-checks", action="store_true", help="Verify nonzero palm TCP transforms and 12 native hand contact bindings")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.rgb
if args.rgb:
    from utils.isaac_rendering import configure_headless_camera_parity_experience
    configure_headless_camera_parity_experience(args, log_prefix="manipisa")
launcher = AppLauncher(args)
app = launcher.app


def main():
    import torch
    import numpy as np
    import isaaclab.sim as sim_utils
    from robots.multi_ur5_wuji_with_flange import spawn_multi_ur5_wuji_with_flange
    from manipisa import Runtime, Instruction, Opcode, Mode, ShapeGoal, PoseGoal, Status
    from manipisa.types import TERMINAL
    from manipisa.adapters import DualUr5WujiAdapter, DualUr5WujiConfig, SideBinding, ToolFrame, RigidTransform, WujiContact
    from manipisa.rgb import Bench2DexRGBSource, RGBFeedback

    torch.manual_seed(0)
    out = args.output or ROOT / "artifacts" / ("runtime-" + ("rgb-" if args.rgb else "structured-") + time.strftime("%Y%m%d-%H%M%S"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)
    # The upstream spawner resolves asset paths relative to Bench2Dex.
    original_cwd = Path.cwd()
    os.chdir(ROOT / "Bench2Dex")
    try:
        spawned = spawn_multi_ur5_wuji_with_flange((1.5, 1.0, 0.05), 0.75, "")
    finally:
        os.chdir(original_cwd)
    robot = spawned["interactive_objects"]["global_robot"]
    config, contacts, entities = DualUr5WujiConfig(), (), {}
    if args.adapter_checks:
        from isaaclab.assets import RigidObject, RigidObjectCfg
        config = DualUr5WujiConfig(
            right=SideBinding("arm_a","hand_a","tool_a","palm_a",ToolFrame("palm",RigidTransform((.012,-.003,.005), (np.cos(.1),0.,0.,np.sin(.1))))),
            left=SideBinding("arm_b","hand_b","tool_b","palm_b",ToolFrame("palm",RigidTransform((-.008,.002,.006), (np.cos(.15),np.sin(.15),0.,0.)))))
        # A distant target lets us verify report wiring without altering the
        # robot asset/materials or claiming a Wuji object manipulation task.
        entities["binding_target"] = RigidObject(RigidObjectCfg(prim_path="/World/BindingTarget",
            spawn=sim_utils.CuboidCfg(size=(.1,.1,.1), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.), collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.,3.,2.))))
        contacts = tuple(WujiContact(f"{side}_{region}_target",side,region,"binding_target","/World/BindingTarget",
            controller="arm" if region == "palm" else "hand") for side in ("right","left")
            for region in ("palm","thumb_tip","index_tip","middle_tip","ring_tip","little_tip"))
        DualUr5WujiAdapter.enable_contact_reports(sim.stage,robot.cfg.prim_path,contacts)
    rig = None
    if args.rgb:
        from collector.cameras import CameraRig
        from collector.config import CameraCollectConfig
        cameras = [
            CameraCollectConfig(camera_id="overview", position=(2.5, -2.0, 2.3), target=(0.5, 0.0, 1.1), width=320, height=240, focal_length=16.0),
            CameraCollectConfig(camera_id="overhead", position=(0.5, 0.1, 3.0), target=(0.5, 0.1, 0.8), width=320, height=240, focal_length=10.0),
        ]
        rig = CameraRig(sim, cameras, enable_rgb=True, enable_depth=False, robot_articulation=robot)
    sim.reset()
    # Isaac Lab default_joint_pos is configuration metadata until explicitly
    # applied. State writes are confined to this episode initialization.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    robot.reset()
    robot.update(sim.get_physics_dt())
    if rig is not None:
        rig.set_robot_articulation(robot)
        rig.initialize_after_reset()
    adapter = DualUr5WujiAdapter(sim, robot, config=config, contacts=contacts, entities=entities,
                                pre_step_hooks=spawned["pre_step_hooks"])
    # Physical settling with fixed initial joint targets; no state teleportation during control.
    for _ in range(150):
        adapter.step()
    rgb = RGBFeedback(Bench2DexRGBSource(sim, rig), enabled=False, period_s=0.10) if rig is not None else None
    runtime = Runtime(adapter, rgb=rgb)
    trace = []
    checks = {}
    checks["dedicated_adapter"] = type(adapter).__name__ == "DualUr5WujiAdapter"
    checks["four_actor_bindings_52_joints"] = sorted(len(r.joints) for r in adapter._actors.values()) == [6,6,20,20]
    def check_transforms():
        if not args.adapter_checks:
            return
        from manipisa.interaction import angle_between, rotation
        state = adapter.observe()
        for side in ("right","left"):
            spec = getattr(config,side)
            palm,tool = state.frames[spec.palm_frame],state.frames[spec.tool_frame]
            expected = RigidTransform(palm.position,palm.orientation_wxyz).compose(spec.tcp.offset)
            expected_v = np.array(palm.linear_velocity) + np.cross(palm.angular_velocity,rotation(palm.orientation_wxyz)@spec.tcp.offset.position)
            ok = np.linalg.norm(np.array(expected.position)-tool.position) < 2e-4 and angle_between(expected.orientation_wxyz,tool.orientation_wxyz) < 2e-3
            ok = bool(ok and np.linalg.norm(expected_v-tool.linear_velocity) < 2e-3)
            checks[f"{side}_palm_tcp_pose_velocity"] = checks.get(f"{side}_palm_tcp_pose_velocity",True) and ok
        values = list(state.contacts.values())
        checks["twelve_native_contact_pairs_valid"] = checks.get("twelve_native_contact_pairs_valid",True) and len(values)==12 and all(c.valid and c.wrench_valid for c in values)
    check_transforms()

    def tick():
        state = runtime.step()
        check_transforms()
        trace.append(runtime.feedback())
        return state

    def run_until_terminal(call_ids, max_steps=900):
        for _ in range(max_steps):
            if all(runtime.query(k).status in TERMINAL for k in call_ids):
                return
            if not app.is_running():
                raise RuntimeError("Simulation app stopped")
            tick()
        raise AssertionError("Runtime did not reach a terminal result within the bounded run")

    initial = adapter.observe()
    shape_target = list(initial.joints["hand_a"].position)
    shape_target[5] += 0.08
    shape = Instruction(Opcode.SHAPE_HAND, ("hand_a",), ShapeGoal(tuple(shape_target)),
                        timeout_s=6.0, max_joint_speed=0.8, hold_for_s=15.0)
    shape_call = runtime.submit(shape)
    conflict = runtime.submit(shape)
    checks["resource_conflict_rejected"] = conflict.status == Status.REJECTED and conflict.reason == "RESOURCE_CONFLICT"
    unsupported = runtime.submit(Instruction(Opcode.CONTROL_GRASP, ("hand_a",), None))
    checks["missing_grasp_operands_rejected"] = unsupported.reason == "INVALID_ARGUMENT"
    run_until_terminal([shape_call.call_id])
    checks["shape_succeeded"] = runtime.query(shape_call.call_id).status == Status.SUCCEEDED
    checks["rgb_initially_off"] = rgb is None or rgb.capture_count == 0
    if rgb is not None:
        runtime.set_rgb_enabled(True)

    state = adapter.observe()
    move_calls = []
    for actor, frame in (("arm_a", "tool_a"), ("arm_b", "tool_b")):
        pose = state.frames[frame]
        goal = PoseGoal(frame, (pose.position[0], pose.position[1], pose.position[2] + 0.02), pose.orientation_wxyz)
        request = Instruction(Opcode.MOVE, (actor,), goal, timeout_s=7.0, max_joint_speed=0.8, hold_for_s=10.0)
        move_calls.append(runtime.submit(request))
    run_until_terminal([r.call_id for r in move_calls])
    for i, call in enumerate(move_calls):
        checks[f"move_{i}_succeeded"] = runtime.query(call.call_id).status == Status.SUCCEEDED

    previous = runtime.query(move_calls[0].call_id)
    pose = adapter.observe().frames["tool_a"]
    sustain = Instruction(Opcode.MOVE, ("arm_a",), PoseGoal("tool_a", pose.position, pose.orientation_wxyz),
                          mode=Mode.SUSTAIN, sustain_s=0.25, timeout_s=2.0, dwell_s=0.05, hold_for_s=5.0)
    held = runtime.submit(sustain, replace_handle=previous.control_handle)
    run_until_terminal([held.call_id], max_steps=250)
    held_result = runtime.query(held.call_id)
    checks["sustain_succeeded_after_active"] = held_result.status == Status.SUCCEEDED and held_result.active_since is not None
    indefinite = Instruction(Opcode.MOVE, ("arm_a",), sustain.goal, mode=Mode.SUSTAIN,
                             timeout_s=2.0, dwell_s=0.02, hold_for_s=5.0)
    cancel_call = runtime.submit(indefinite, replace_handle=held_result.control_handle)
    for _ in range(10):
        tick()
    canceled = runtime.cancel(cancel_call.call_id)
    checks["cancel_retains_control"] = canceled.status == Status.CANCELED and canceled.control_handle is not None
    rgb_metadata = runtime.feedback()["rgb"]
    if rgb is not None:
        import cv2
        checks["rgb_frames_valid"] = all(f["valid"] for f in rgb_metadata["frames"].values()) and len(rgb_metadata["frames"]) == 2
        checks["rgb_has_content"] = all(frame.rgb is not None and float(frame.rgb.std()) > 1 for frame in rgb.frames().values())
        for camera_id, frame in rgb.frames().items():
            if frame.valid:
                cv2.imwrite(str(out / f"{camera_id}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        count = rgb.capture_count
        runtime.set_rgb_enabled(False)
        for _ in range(4):
            tick()
        checks["rgb_disable_stops_capture_and_clears_frames"] = rgb.capture_count == count and rgb.frames() == {}

    final = runtime.feedback()
    report = {"passed": all(checks.values()), "checks": checks, "rgb_enabled_run": args.rgb,
              "binding": adapter.describe_bindings(),
              "adapter_checks": args.adapter_checks,
              "scope": "Free-space physical MOVE/SHAPE_HAND smoke on local UR5+Wuji; not a Bench2Dex task success score",
              "device": str(sim.device), "physics_dt": adapter.dt, "initial_state": initial.to_dict(),
              "feedback": final, "rgb_before_disable": rgb_metadata,
              "note": "Simulation truth is explicit. No tactile, hidden task labels or demonstration actions used."}
    source_files = sorted((ROOT / "manipisa").rglob("*.py")) + [Path(__file__).resolve()]
    report["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(row, allow_nan=False) for row in trace), encoding="utf-8")
    print("MANIPISA_SMOKE " + json.dumps({"output": str(out), "passed": report["passed"], "checks": checks}), flush=True)
    if rig is not None:
        rig.close()
    if not report["passed"]:
        raise AssertionError("See report.json for failed checks")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Bench2Dex/replay.py uses immediate exit on this Windows stack because
        # Kit/Replicator shutdown can hang. Preserve failures, and flush only
        # after synchronous report/image writes. This is a standalone runner.
        if os.name == "nt":
            print(f"MANIPISA_PROCESS_EXIT {exit_code}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        app.close(wait_for_replicator=False)
    sys.exit(exit_code)
