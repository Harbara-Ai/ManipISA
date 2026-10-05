"""Policy-side RoboDojo adapter; the official environment owns physics and IK.

Only public policy observations are accepted. In particular, RoboDojo's
``*_ee_joint_state`` is a previous control target, not a measured gripper
position. It must never satisfy a SHAPE_HAND or contact contract.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from manipisa.types import (
    AdapterError, Instruction, JointState, Opcode, Pose, PoseGoal,
    PreparedTask, StateSnapshot,
)

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SIDES = ("left", "right")
# Preserve ManipISA's existing right=a / left=b convention.
ACTORS = {"arm_a": ("right", "tool_a"), "arm_b": ("left", "tool_b")}
ROBODOJO_REVISION = "266130a3ec41ba9b20b0e5648d2f3542f219c06c"
XPOLICYLAB_REVISION = "bb9a0b5f5136a74503b679af830bfd0a3a837d5c"


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected {size} finite coordinates")
    return array.copy()


def public_observation(obs):
    """Allowlist and copy the already-decoded XPolicyLab observation.

    Do not decode JPEGs here: XPolicyLab owns decoding and RGB channel order.
    No evaluator labels, object ground truth or action history pass this boundary.
    """
    if obs.get("data_format_version") != "v1.0":
        raise ValueError("Expected RoboDojo data_format_version v1.0")
    if type(obs.get("env_idx", 0)) is not int or obs.get("env_idx", 0) != 0:
        raise ValueError("This integration requires one environment, env_idx=0")
    frequency = float(obs["additional_info"]["frequency"])
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Invalid observation frequency")
    instruction = obs["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing language instruction")
    state = {}
    for side in SIDES:
        for suffix, size in (("arm_joint_state", 6), ("ee_pose", 7), ("ee_joint_state", 1)):
            key = f"{side}_{suffix}"
            state[key] = _vector(obs["state"][key], size, key)
        pose = state[f"{side}_ee_pose"]
        if abs(np.linalg.norm(pose[3:]) - 1.0) > 1e-3:
            raise ValueError(f"{side}_ee_pose: expected normalized wxyz quaternion")
        if not 0 <= state[f"{side}_ee_joint_state"][0] <= 1:
            raise ValueError("Normalized gripper control target must be in [0, 1]")
    vision = {}
    for name in CAMERAS:
        pixels = np.asarray(obs["vision"][name]["color"])
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0 or pixels.dtype != np.uint8:
            raise ValueError(f"{name}: expected decoded HWC uint8 RGB")
        vision[name] = {"color": pixels.copy()}
    return {"data_format_version": "v1.0", "env_idx": 0,
            "additional_info": {"frequency": frequency}, "instruction": instruction,
            "state": state, "vision": vision}


def _rotation(q_wxyz):
    return Rotation.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]])


class RoboDojoAdapter:
    """Synchronous Runtime adapter driven by an action/observation exchange.

    exchange(action) must execute exactly ONE official take_action and return
    its next observation. No local simulator, hidden state, or robot asset is
    needed. World means RoboDojo's environment-origin frame (env_idx=0).
    """
    def __init__(self, initial_observation, exchange, *, max_linear_speed=0.10,
                 max_angular_speed=0.50):
        if not all(math.isfinite(v) and v > 0 for v in (max_linear_speed, max_angular_speed)):
            raise ValueError("Motion rates must be finite and positive")
        self.exchange = exchange
        self.max_linear_speed, self.max_angular_speed = max_linear_speed, max_angular_speed
        self.observation = public_observation(initial_observation)
        self.dt = 1.0 / self.observation["additional_info"]["frequency"]
        self.steps = 0
        self._previous = None
        self._action = self.hold_action(self.observation)
        self._snapshot = self._make_snapshot()

    @staticmethod
    def hold_action(obs):
        return {f"{side}_{suffix}": obs["state"][f"{side}_{suffix}"].copy()
                for side in SIDES for suffix in ("ee_pose", "ee_joint_state")}

    def _make_snapshot(self):
        now = self.steps * self.dt
        joints, frames = {}, {}
        for actor, (side, frame) in ACTORS.items():
            current = self.observation["state"]
            q, pose = current[f"{side}_arm_joint_state"], current[f"{side}_ee_pose"]
            if self._previous is None:
                velocity, linear, angular = (float("nan"),) * 6, (float("nan"),) * 3, (float("nan"),) * 3
            else:
                prior = self._previous["state"]
                velocity = tuple((q - prior[f"{side}_arm_joint_state"]) / self.dt)
                previous_pose = prior[f"{side}_ee_pose"]
                linear = tuple((pose[:3] - previous_pose[:3]) / self.dt)
                angular = tuple((_rotation(pose[3:]) * _rotation(previous_pose[3:]).inv()).as_rotvec() / self.dt)
            # Velocity is an interval estimate, not a high-rate physical measurement.
            source = "robodojo_public_state+finite_difference_velocity"
            joints[actor] = JointState(tuple(q), velocity, now, self._previous is not None, source,
                                       tuple(f"{side}_joint_{i}" for i in range(6)))
            frames[frame] = Pose(tuple(pose[:3]), tuple(pose[3:]), linear, angular, now,
                                 self._previous is not None, source)
        return StateSnapshot(now, 0, joints, frames, state_source="robodojo_public_observation")

    def observe(self):
        return deepcopy(self._snapshot)

    def prepare(self, command: Instruction, snapshot):
        if command.operation != Opcode.MOVE:
            raise AdapterError("UNSUPPORTED_CAPABILITY",
                "RoboDojo public observations lack measured gripper/contact/wrench evidence; "
                "only MOVE contracts are supported. set_gripper is an unverified control command.")
        if len(command.actors) != 1 or command.actors[0] not in ACTORS:
            raise AdapterError("UNSUPPORTED_CAPABILITY", "MOVE requires arm_a (right) or arm_b (left)")
        actor = command.actors[0]
        side, frame = ACTORS[actor]
        goal = command.goal
        if not isinstance(goal, PoseGoal) or goal.controlled_target != frame or goal.reference_frame != "world" or goal.target_type != "ACTOR_FRAME":
            raise AdapterError("UNSUPPORTED_CAPABILITY", f"Expected world ACTOR_FRAME {frame}")
        if command.execution_domain != "free_space":
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Contact-controlled motion is unavailable")
        return PreparedTask(frozenset(f"{side}_arm_joint_{i}" for i in range(6)), frozenset(), binding=actor)

    def command(self, command, prepared):
        side, frame = ACTORS[prepared.binding]
        actual = self._snapshot.frames[frame]
        goal = command.goal
        delta = np.asarray(goal.position) - actual.position
        norm = np.linalg.norm(delta)
        position = np.asarray(actual.position) + delta * min(1.0, self.max_linear_speed * self.dt / max(norm, 1e-12))
        current_rotation = _rotation(actual.orientation_wxyz)
        rotation_delta = (_rotation(goal.orientation_wxyz) * current_rotation.inv()).as_rotvec()
        angle = np.linalg.norm(rotation_delta)
        fraction = min(1.0, self.max_angular_speed * self.dt / max(angle, 1e-12))
        quaternion = (Rotation.from_rotvec(rotation_delta * fraction) * current_rotation).as_quat()[[3, 0, 1, 2]]
        self._action[f"{side}_ee_pose"] = np.concatenate((position, quaternion))

    def hold(self, prepared):
        side, _ = ACTORS[prepared.binding]
        self._action[f"{side}_ee_pose"] = self.observation["state"][f"{side}_ee_pose"].copy()

    def set_gripper(self, side, value):
        """Queue a normalized gripper command, without claiming measured success."""
        if side not in SIDES or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Expected side left/right and a finite value in [0, 1]")
        self._action[f"{side}_ee_joint_state"] = np.array([value], dtype=float)
        return {"queued": True, "side": side, "target": float(value), "physical_success_verified": False}

    def step(self):
        received = public_observation(self.exchange(deepcopy(self._action)))
        if received["additional_info"]["frequency"] != self.observation["additional_info"]["frequency"]:
            raise ValueError("Observation frequency changed inside an episode")
        if received["instruction"] != self.observation["instruction"]:
            raise ValueError("Instruction changed without reset")
        self._previous, self.observation = self.observation, received
        self.steps += 1
        self._snapshot = self._make_snapshot()

    def gripper_targets(self):
        return {side: {"reported_control_target": float(self.observation["state"][f"{side}_ee_joint_state"][0]),
                       "requested_target": float(self._action[f"{side}_ee_joint_state"][0]),
                       "measured_position_available": False} for side in SIDES}
