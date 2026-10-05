"""Bench2Dex task host. Import only AFTER Isaac Lab AppLauncher."""
from __future__ import annotations

from dataclasses import asdict, replace
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .catalog import ROBOT_KEY, sha256
from .budgets import EpisodeBudget
from .programs import EpisodeStopped, public_builtins
from .tool_feedback import FEEDBACK_FORMAT, execute_with_feedback, _timed


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


class NativeRobot:
    """SDK control plus copied observations. Reset/teleport methods are absent."""
    def __init__(self, robot):
        self._robot = robot
        self.joint_names = tuple(robot.joint_names)
        self.body_names = tuple(robot.body_names)
        self.device = robot.device
        self.num_joints = robot.num_joints
        self.num_bodies = robot.num_bodies
        self.is_fixed_base = robot.is_fixed_base
        self.root_physx_view = SimpleNamespace(get_jacobians=lambda: robot.root_physx_view.get_jacobians().clone())

    @property
    def data(self):
        names = ("joint_pos", "joint_vel", "joint_pos_limits", "soft_joint_pos_limits", "default_joint_pos",
                 "body_state_w", "body_link_state_w", "root_state_w", "root_pos_w", "root_quat_w")
        return SimpleNamespace(**{name: getattr(self._robot.data, name).clone() for name in names})

    def find_joints(self, *args, **kwargs):
        return self._robot.find_joints(*args, **kwargs)

    def find_bodies(self, *args, **kwargs):
        return self._robot.find_bodies(*args, **kwargs)

    def set_joint_position_target(self, target, joint_ids=None):
        self._robot.set_joint_position_target(torch.as_tensor(target, device=self.device, dtype=torch.float32), joint_ids=joint_ids)

    def set_joint_velocity_target(self, target, joint_ids=None):
        self._robot.set_joint_velocity_target(torch.as_tensor(target, device=self.device, dtype=torch.float32), joint_ids=joint_ids)

    def set_joint_effort_target(self, target, joint_ids=None):
        self._robot.set_joint_effort_target(torch.as_tensor(target, device=self.device, dtype=torch.float32), joint_ids=joint_ids)


class SimulationEpisode:
    def __init__(self, root, task_path, method, seed, snapshot_path, out, app, *, wall_limit=None,
                 profile=False, collect_config_path=None):
        import yaml
        import isaaclab.sim as sim_utils
        from build import build_scene
        from collector.cameras import CameraRig
        from collector.config import load_collect_config
        from collector.state_reader import read_object_states, read_joint_state, read_joint_limits
        from utils.episode_runtime import initialize_scene_runtime_state
        from utils.seed_policy import seed_everything
        from manipisa.adapters import DualUr5WujiAdapter, WujiContact
        from manipisa import Runtime
        from benchmark.metric_tracker import MetricTracker

        self.root, self.out = Path(root), Path(out)
        self.task_path, self.method, self.seed = Path(task_path), method, seed
        self.task = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))
        experiment = json.loads((self.root / "configs/ur5_wuji_codex.json").read_text(encoding="utf-8"))
        self.budget_config = EpisodeBudget.from_config(experiment, wall_limit_s=wall_limit)
        self.dt, self.wall_limit = self.budget_config.physics_dt, self.budget_config.wall_limit_s
        self.steps, self.queries, self.tool_calls = 0, 0, 0
        self._observation_attempts, self.visual_feedback_count = 0, 0
        self.started, self.terminal_time, self.reason = None, None, None
        self.error, self.evaluation_valid = None, True
        self.read_objects, self.read_robot = read_object_states, read_joint_state
        seed_everything(seed)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device="cuda:0"))
        self.world = build_scene(self.task, str(self.task_path.parent), robot_key=ROBOT_KEY, generalization_enabled=False)
        self.objects = self.world["interactive_objects"]
        self.robot = self.objects["global_robot"]
        self.ids = [obj["id"] for obj in self.task["objects"]]
        self.hooks = self.world["robot_runtime"].get("pre_step_hooks", [])
        # Bind currently supported rigid-object contacts; articulated link
        # contacts require a separately calibrated adapter extension.
        rigid_ids = [key for key in self.ids if self.world["object_body_types"].get(key) == "dynamic"]
        contacts = tuple(WujiContact(f"{side}_{region}_{key}", side, region, key, self.world["object_prim_paths"][key],
                                    controller="arm" if region == "palm" else "hand")
                         for key in rigid_ids for side in ("right", "left")
                         for region in ("palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"))
        DualUr5WujiAdapter.enable_contact_reports(self.sim.stage, self.robot.cfg.prim_path, contacts)
        collect_path = Path(collect_config_path) if collect_config_path is not None else self.root / "Bench2Dex/configs/collect/default.yaml"
        collect = load_collect_config(str(collect_path), robot_key=ROBOT_KEY)
        camera_ids = experiment["observations"].get("camera_ids")
        if camera_ids is not None:
            available = {camera.camera_id: camera for camera in collect.cameras}
            if (not isinstance(camera_ids, list) or not camera_ids
                    or any(not isinstance(name, str) or name not in available for name in camera_ids)
                    or len(set(camera_ids)) != len(camera_ids)):
                raise ValueError("observations.camera_ids must be a nonempty list of unique configured camera IDs")
            collect.cameras = [available[name] for name in camera_ids]
        camera_config = {"source_path": str(collect_path.resolve()), "source_sha256": sha256(collect_path),
                         "robot_key": ROBOT_KEY, "cameras": [asdict(camera) for camera in collect.cameras]}
        (self.out / "camera-config.json").write_text(json.dumps(camera_config, indent=2), encoding="utf-8")
        self.rig = CameraRig(self.sim, collect.cameras, enable_rgb=True, enable_depth=False, robot_articulation=self.robot)
        initialize_scene_runtime_state(sim=self.sim, physics_dt=self.dt, interactive_objects=self.objects,
            object_prim_paths=self.world["object_prim_paths"], object_display_colors=self.world["object_display_colors"],
            collector=None, app_running_state_fn=app.is_running)
        self.rig.set_robot_articulation(self.robot)
        self.rig.initialize_after_reset()
        for _ in range(int(self.world["settle_steps"])):
            self.raw_step()
        self.restore_or_save(Path(snapshot_path))
        entities = {k: v for k, v in self.objects.items() if k != "global_robot" and hasattr(v.data, "root_state_w")}
        # USD object roots can be Xforms with a nested rigid body (e.g. frypan).
        # Bind the exact PhysX body after reset; report enabling above only
        # depends on the robot's sensor links, not these provisional targets.
        rigid_paths = {key: tuple(map(str, entities[key].root_physx_view.prim_paths)) for key in rigid_ids}
        if any(len(paths) != 1 for paths in rigid_paths.values()):
            raise ValueError("Rigid contact targets must each resolve to one PhysX body")
        contacts = tuple(replace(c, target_path=rigid_paths[c.target][0]) for c in contacts)
        self.adapter = DualUr5WujiAdapter(self.sim, self.robot, contacts=contacts, entities=entities, pre_step_hooks=self.hooks)
        self.runtime = Runtime(self.adapter)
        self.joint_limits = read_joint_limits(self.robot)
        spec = copy.deepcopy(self.task["metrics"])
        spec["policy_stride"] = self.budget_config.policy_stride
        spec["metrics_trace"] = True
        # Same runtime geometry injection as Bench2Dex run_policy.py.
        spec.setdefault("safety", {})["table_z"] = float(self.world["table_z"])
        if isinstance(spec.get("grasp"), dict):
            spec["grasp"]["table_z"] = float(self.world["table_z"])
        self.tracker = MetricTracker(spec, success_conditions=self.task.get("success_conditions"),
                                    dt=self.dt, robot_key=ROBOT_KEY,
                                    table_height_offset=self.world["table_z"] - self.world["nominal_table_z"])
        self.budget = self.budget_config.physics_steps(spec["expert_time_step"])
        self.tracker.record_joint_limit_baseline("task_delivery", robot_state=self.read_robot(self.robot), joint_limits=self.joint_limits)
        self.contact_reader = None
        if self.world.get("contact_pairs"):
            from collector.contact_sensor_reader import ContactSensorReader
            self.contact_reader = ContactSensorReader(self.world["contact_pairs"], self.world["object_prim_paths"])
            if not self.contact_reader.available:
                raise RuntimeError("Configured evaluator contact sensors are unavailable")
        self.namespace = self.make_namespace()
        import importlib.metadata
        import platform
        self.runtime_environment = {"python": platform.python_version(), "platform": platform.platform(),
                                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                    "gpu": torch.cuda.get_device_name(self.robot.device)}
        for package in ("isaaclab", "isaacsim", "numpy"):
            try:
                self.runtime_environment[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.runtime_environment[package] = "not available as package metadata"
        # Match replay.py's rendering warmup without consuming task physics.
        self.rig.sync_mounted_camera_poses()
        for _ in range(5):
            self.sim.render()
        self.first_observation = self.observe_content()
        self.profiler = None
        if profile:
            from manipisa.profiling import TimingProfiler, attach_episode_profiler
            self.profiler = attach_episode_profiler(self, TimingProfiler())

    def raw_step(self):
        for hook in self.hooks:
            hook()
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        for obj in self.objects.values():
            obj.update(self.dt)

    def state_snapshot(self):
        values = {}
        for key, obj in self.objects.items():
            if hasattr(obj.data, "root_state_w"):
                values[key] = {"root": jsonable(obj.data.root_state_w)}
                if hasattr(obj.data, "joint_pos"):
                    values[key].update(qpos=jsonable(obj.data.joint_pos), qvel=jsonable(obj.data.joint_vel))
        return {"scene_sha256": sha256(self.task_path), "robot_key": ROBOT_KEY,
                "seed": self.seed, "state": values, "scene_sample": jsonable(self.world["scene_generalization_sample"])}

    def restore_or_save(self, path):
        from utils.runtime_helpers import is_kinematic_rigid_object
        before = self.state_snapshot()
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            for field in ("scene_sha256", "robot_key", "seed", "scene_sample"):
                if before[field] != saved[field]:
                    raise ValueError(f"Paired initialization differs: {field}")
            if saved["state"].keys() != before["state"].keys():
                raise ValueError("Paired object roster differs")
            for key, state in saved["state"].items():
                obj = self.objects[key]
                root = torch.tensor(state["root"], device=obj.device, dtype=torch.float32)
                obj.write_root_pose_to_sim(root[:, :7])
                if not is_kinematic_rigid_object(obj):
                    obj.write_root_velocity_to_sim(root[:, 7:])
                if "qpos" in state:
                    qpos = torch.tensor(state["qpos"], device=obj.device, dtype=torch.float32)
                    qvel = torch.tensor(state["qvel"], device=obj.device, dtype=torch.float32)
                    obj.write_joint_state_to_sim(qpos, qvel)
                    if key == "global_robot":
                        obj.set_joint_position_target(qpos)
                obj.update(self.dt)
            self.sim.forward()
            after = self.state_snapshot()
            for key, state in saved["state"].items():
                for field, values in state.items():
                    if not np.allclose(values, after["state"][key][field], atol=1e-6, rtol=0):
                        raise ValueError(f"Paired restore mismatch: {key}/{field}")
        else:
            saved = before
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        self.snapshot_hash = sha256(path)
        self.initial_state = self.state_snapshot()

    def begin(self):
        if self.started is not None:
            raise RuntimeError("Task already delivered")
        self.started = time.perf_counter()

    def stop(self, reason):
        if self.reason is None:
            self.reason, self.terminal_time = reason, time.perf_counter()

    def check_time(self):
        if self.reason is not None:
            raise EpisodeStopped(self.reason)
        if self.started is None:
            raise RuntimeError("Task has not been delivered")
        if time.perf_counter() - self.started >= self.wall_limit:
            self.stop("wall_timeout")
            raise EpisodeStopped(self.reason)

    def step(self, n=1):
        if type(n) is not int or n < 1:
            raise ValueError("step(n) requires a positive integer")
        for _ in range(n):
            self.check_time()
            if self.method == "manipisa":
                self.runtime.step(return_snapshot=False)
            else:
                self.raw_step()
            self.steps += 1
            states = self.read_objects(self.objects, self.ids)
            if self.contact_reader is not None:
                self.contact_reader.update(self.dt)
                for key, forces in self.contact_reader.read().items():
                    if key in states:
                        states[key]["contact_forces"] = forces
            if set(states) != set(self.ids):
                self.evaluation_valid = False
                raise RuntimeError("Evaluator object state missing")
            self.tracker.update(states, sim_step=self.steps, dt=self.dt,
                                robot_state=self.read_robot(self.robot), joint_limits=self.joint_limits)
            if self.tracker.success:
                self.stop("stable_success")
            elif self.steps >= self.budget:
                self.stop("max_steps")
            if self.reason:
                raise EpisodeStopped(self.reason)

    def public_state(self):
        return {"state_source": "simulation_privileged", "physics_step": self.steps,
                "physics_dt": self.dt, "physics_step_budget": self.budget,
                "robot": jsonable(self.read_robot(self.robot)), "joint_limits": jsonable(self.joint_limits),
                "bodies": {name: jsonable(self.robot.data.body_link_state_w[0, i]) for i, name in enumerate(self.robot.body_names)},
                "objects": jsonable(self.read_objects(self.objects, self.ids)),
                "object_geometry": jsonable(self.world["asset_local_bbox"]),
                "contacts": self.adapter.observe().to_dict()["contacts"],
                "frame_conventions": {"bodies": "position_xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz",
                                      "objects.pose_world": "position_xyz, quaternion_xyzw"}}

    def _capture_observation(self, *, compact=False):
        """One synchronized render; disk evidence and reply describe the same step."""
        from PIL import Image
        if self.started is not None:
            self.check_time()
        capture_step = self.steps
        observation_id = f"{self._observation_attempts:04d}"
        self._observation_attempts += 1  # Failed attempts never overwrite old images.
        with _timed(self, "visual.render"):
            self.rig.sync_mounted_camera_poses()
            self.sim.render()
        with _timed(self, "visual.read"):
            frames = self.rig.capture(self.dt)
        camera_ids = list(self.rig.camera_ids)
        if not camera_ids or set(frames) != set(camera_ids):
            raise RuntimeError("Incomplete camera set; no images returned")
        if self.started is not None:
            self.check_time()
        with _timed(self, "visual.state"):
            if compact:
                robot = self.read_robot(self.robot)
                state = {"state_source": "simulation_privileged", "physics_step": capture_step,
                         "state_scope": "current_joint_positions_velocities_and_objects",
                         "joint_order": "initial_observation.robot.joint_names",
                         "robot": {key: jsonable(robot[key]) for key in ("qpos", "qvel")},
                         "objects": jsonable(self.read_objects(self.objects, self.ids))}
            else:
                state = self.public_state()
        state.update(observation_id=observation_id, camera_ids=camera_ids, cameras={})
        images, encoded = [], {}
        folder = self.out / "observations" / observation_id
        with _timed(self, "visual.encode"):
            for name in camera_ids:
                frame = frames[name]
                rgb = np.asarray(frame.rgb)
                if rgb.ndim != 3 or rgb.shape[2] < 3 or not np.isfinite(rgb).all():
                    raise RuntimeError(f"Invalid RGB frame: {name}")
                if rgb.dtype != np.uint8 or tuple(rgb.shape[:2]) != tuple(frame.image_shape):
                    raise RuntimeError(f"Unexpected RGB dtype or shape: {name}/{rgb.dtype}/{rgb.shape}")
                stream = io.BytesIO()
                Image.fromarray(rgb[:, :, :3]).save(stream, format="PNG")
                data = stream.getvalue()
                encoded[name] = data
                images.append({"type": "image", "mimeType": "image/png",
                               "data": base64.b64encode(data).decode()})
                state["cameras"][name] = {
                    "physics_step": capture_step, "camera_model": frame.camera_model,
                    "image_shape": jsonable(frame.image_shape),
                    "intrinsic": jsonable(frame.intrinsic),
                    "extrinsic_world_from_cam": jsonable(frame.extrinsic_world_from_cam),
                    "distortion_coefficients": jsonable(frame.distortion_coefficients),
                    "fisheye_camera_matrix": jsonable(frame.fisheye_camera_matrix),
                    "image_path": (Path("observations") / observation_id / f"{name}.png").as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        if self.steps != capture_step:
            raise RuntimeError("Physics advanced during RGB capture; no images returned")
        with _timed(self, "visual.log"):
            folder.mkdir(parents=True, exist_ok=False)
            for name, data in encoded.items():
                (folder / f"{name}.png").write_bytes(data)
            (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        if self.started is not None:
            self.check_time()
        self.queries += 1
        return state, images

    def observe_content(self):
        state, images = self._capture_observation()
        return [{"type": "text", "text": json.dumps(state)}, *images]

    def capture_visual_feedback(self):
        state, images = self._capture_observation(compact=True)
        self.visual_feedback_count += 1
        metadata = {"status": "captured", "observation_id": state["observation_id"],
                    "physics_step": state["physics_step"], "camera_ids": state["camera_ids"],
                    "state": state}
        return metadata, images

    def make_namespace(self):
        import isaaclab.utils.math as math_utils
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        namespace = {"__builtins__": public_builtins(), "np": np, "torch": torch, "math": math,
                     "math_utils": math_utils, "step": self.step, "state": self.public_state}
        if self.method == "direct":
            namespace.update(robot=NativeRobot(self.robot), DifferentialIKController=DifferentialIKController,
                             DifferentialIKControllerCfg=DifferentialIKControllerCfg)
        else:
            import manipisa.types as types
            for name in ("Instruction", "Opcode", "Mode", "PoseGoal", "ShapeGoal", "ContactGoal", "GraspGoal", "WrenchGoal", "ContactRequirement", "ContactPoint", "Status"):
                if hasattr(types, name):
                    namespace[name] = getattr(types, name)
            namespace["runtime"] = SimpleNamespace(**{name: getattr(self.runtime, name) for name in
                           ("submit", "query", "cancel", "update", "feedback")})
        return namespace

    def documents(self):
        sdk = self.root / "IsaacLab/source/isaaclab/isaaclab"
        docs = {"sdk_ik": sdk / "controllers/differential_ik.py", "sdk_ik_config": sdk / "controllers/differential_ik_cfg.py"}
        if self.method == "manipisa":
            docs.update(instructions=self.root / "manipisa/types.py", contracts=self.root / "docs/manipisa-v0.2-core-contracts.md",
                        feedback=Path(__file__).resolve().parents[2] / "docs/concise-feedback.md",
                        quickstart=Path(__file__).resolve().parents[2] / "docs/runtime-quickstart.md")
        return docs

    def tool(self, name, arguments):
        self.check_time()
        self.tool_calls += 1
        if name == "observe":
            try:
                return {"content": self.observe_content()}
            finally:
                # An ordinary capture failure must not hide an expired budget.
                self.check_time()
        if name == "read_api":
            docs = self.documents()
            requested = arguments.get("name")
            text = json.dumps(list(docs)) if requested == "index" else docs[requested].read_text(encoding="utf-8")
            return {"content": [{"type": "text", "text": text}]}
        if name != "execute_python":
            raise ValueError("Unknown tool")
        code = arguments["code"]
        with (self.out / "programs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tool_call": self.tool_calls, "physics_step": self.steps, "code": code}) + "\n")
        return execute_with_feedback(self, code, normalize=jsonable)

    def result(self):
        if self.reason is None:
            self.stop("agent_stopped")
        official = asdict(self.tracker.finalize(steps=self.steps, terminated_reason=self.reason,
                          policy_query_count=self.queries, policy_query_count_at_stable_success=self.queries if self.tracker.success else None))
        official["evaluation_valid"] = self.evaluation_valid
        return {"task": self.task_path.stem, "method": self.method, "robot_key": ROBOT_KEY,
                "tool_feedback_format": FEEDBACK_FORMAT,
                "rgb_observations": {"total": self.queries, "automatic_after_execution": self.visual_feedback_count},
                "performance": self.profiler.snapshot() if self.profiler is not None else None,
                "seed": self.seed, "initialization_sha256": self.snapshot_hash,
                "scene_sha256": sha256(self.task_path), "official": jsonable(official), "reason": self.reason,
                "wall_time_s": self.terminal_time - self.started if self.started is not None else None,
                "physics_steps": self.steps, "physics_step_budget": self.budget, "tool_calls": self.tool_calls,
                "wall_time_limit_s": self.wall_limit,
                "budget_config": self.budget_config.to_dict(),
                "runtime_environment": self.runtime_environment,
                "initial_state": self.initial_state, "observation": "simulation_privileged+RGB",
                "limitations": ["No calibrated grasp friction or separation geometry; corresponding contact instructions can reject",
                                "Articulated object link contacts are not bound to ManipISA",
                                "Paired root/joint state restored; PhysX internal solver state is not serialized"],
                "channel": "base_layout_development", "error": self.error}
