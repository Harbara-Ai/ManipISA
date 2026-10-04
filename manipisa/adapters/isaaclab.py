"""Isaac Lab adapter for a deliberately bounded first execution slice.

MOVE: one fixed-base actor's frame, absolute world pose, REACH/SUSTAIN.
SHAPE_HAND: one hand binding's joint-coordinate configuration, REACH.
Bounded contact, grasp and wrench control share the same observation/step path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

from manipisa.types import (
    AdapterError, Evidence, Instruction, JointState, Opcode, Pose, PoseGoal,
    PreparedTask, ShapeGoal, StateSnapshot, Truth, ContactGoal, GraspGoal, WrenchGoal,
)
from .interaction_control import InteractionControl, InteractionPlan


@dataclass(frozen=True)
class ActorBinding:
    articulation: Any
    joint_names: tuple[str, ...]
    role: str  # "arm" or "hand"
    body_name: str | None = None
    frame_id: str | None = None
    tcp_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Declare shared physical task domains when known (e.g. a common object).
    coupling_groups: tuple[str, ...] = ()


@dataclass
class _Resolved:
    actor: str
    spec: ActorBinding
    joints: list[int]
    body: int | None
    controller: Any = None


class IsaacLabAdapter(InteractionControl):
    def __init__(self, sim, actors: dict[str, ActorBinding], *, pre_step_hooks=(), entities=None,
                 predicate_provider: Callable[[float], dict[str, Evidence]] | None = None, contact_source=None):
        # Import after AppLauncher has initialized Isaac Sim.
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.utils import math as math_utils
        self.torch, self.math = torch, math_utils
        self.sim, self.dt = sim, float(sim.get_physics_dt())
        self._actors, self._robots, self._targets = {}, {}, {}
        self.pre_step_hooks = tuple(pre_step_hooks)
        self.entities = dict(entities or {})
        self.predicate_provider = predicate_provider
        self.contact_source = contact_source
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid physics timestep")
        for actor, spec in actors.items():
            robot = spec.articulation
            if robot.num_instances != 1 or not robot.is_fixed_base:
                raise AdapterError("UNSUPPORTED_CAPABILITY", "MVP supports one environment and fixed-base articulations")
            if not spec.joint_names or len(set(spec.joint_names)) != len(spec.joint_names):
                raise ValueError("Joint bindings must be nonempty and unique")
            joints = [robot.joint_names.index(name) for name in spec.joint_names]
            body = robot.body_names.index(spec.body_name) if spec.body_name is not None else None
            if spec.role not in ("arm", "hand"):
                raise ValueError("Binding role must be arm or hand")
            if spec.role == "arm" and (body is None or body == 0 or not spec.frame_id):
                raise ValueError("Arm needs a non-root controlled frame")
            if len(spec.tcp_position) != 3 or len(spec.tcp_orientation_wxyz) != 4:
                raise ValueError("Invalid TCP offset dimensions")
            if not all(math.isfinite(v) for v in (*spec.tcp_position, *spec.tcp_orientation_wxyz)) or abs(sum(v*v for v in spec.tcp_orientation_wxyz)-1) > 1e-3:
                raise ValueError("Invalid TCP transform")
            resolved = _Resolved(actor, spec, joints, body)
            if spec.role == "arm":
                cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
                resolved.controller = DifferentialIKController(cfg, num_envs=1, device=robot.device)
            self._actors[actor] = resolved
            self._robots[id(robot)] = robot
            self._targets[id(robot)] = robot.data.joint_pos.clone()
        frames = [r.spec.frame_id for r in self._actors.values() if r.spec.frame_id]
        if len(frames) != len(set(frames)) or set(frames) & set(self.entities):
            raise ValueError("Observed frame IDs must be unique")

    def _tcp(self, r):
        robot, t, m = r.spec.articulation, self.torch, self.math
        state = robot.data.body_link_state_w[:, r.body]
        offset = t.tensor([r.spec.tcp_position], device=robot.device, dtype=state.dtype)
        offset_q = t.tensor([r.spec.tcp_orientation_wxyz], device=robot.device, dtype=state.dtype)
        position, quaternion = m.combine_frame_transforms(state[:, :3], state[:, 3:7], offset, offset_q)
        world_offset = m.quat_apply(state[:, 3:7], offset)
        linear = state[:, 7:10] + t.cross(state[:, 10:13], world_offset, dim=-1)
        return position, quaternion, linear, state[:, 10:13], world_offset

    def observe(self):
        t = self.torch
        now = float(self.sim.current_time)
        joints, frames, predicates = {}, {}, {}
        for actor, r in self._actors.items():
            robot = r.spec.articulation
            q = robot.data.joint_pos[0, r.joints]
            v = robot.data.joint_vel[0, r.joints]
            valid = bool(t.isfinite(q).all() and t.isfinite(v).all())
            joints[actor] = JointState(tuple(q.cpu().tolist()), tuple(v.cpu().tolist()), now, valid,
                                       coordinate_names=r.spec.joint_names)
            limits = robot.data.joint_pos_limits[0, r.joints]
            in_limits = valid and bool(((q >= limits[:, 0] - 1e-3) & (q <= limits[:, 1] + 1e-3)).all())
            predicates[f"joint_limits:{actor}"] = Evidence(
                (Truth.SATISFIED if in_limits else Truth.VIOLATED) if valid else Truth.UNKNOWN,
                now, "IsaacLab.Articulation", "Observed joint positions against physical limits")
            if r.spec.frame_id is not None and r.body is not None:
                p, qtcp, lin, ang, _ = self._tcp(r)
                values = t.cat((p, qtcp, lin, ang), dim=-1)[0]
                frames[r.spec.frame_id] = Pose(tuple(p[0].cpu().tolist()), tuple(qtcp[0].cpu().tolist()),
                    tuple(lin[0].cpu().tolist()), tuple(ang[0].cpu().tolist()), now, bool(t.isfinite(values).all()))
        # Read-only entity feedback; it does not enable MOVE(ENTITY_FRAME).
        for frame_id, entity in self.entities.items():
            state = entity.data.root_state_w[0]
            frames[frame_id] = Pose(tuple(state[:3].cpu().tolist()), tuple(state[3:7].cpu().tolist()),
                tuple(state[7:10].cpu().tolist()), tuple(state[10:13].cpu().tolist()), now,
                bool(t.isfinite(state).all()), "simulator_entity_state")
        if self.predicate_provider is not None:
            additional = self.predicate_provider(now)
            if set(additional) & set(predicates):
                raise ValueError("External predicates cannot override built-in observations")
            if not all(isinstance(e, Evidence) for e in additional.values()):
                raise ValueError("Predicate provider must return timestamped Evidence")
            predicates.update(additional)
        contacts = self.contact_source.observe(now) if self.contact_source is not None else {}
        return StateSnapshot(now, 0, joints, frames, predicates, contacts=contacts)

    def prepare(self, command: Instruction, snapshot: StateSnapshot):
        if isinstance(command.goal, (ContactGoal, GraspGoal, WrenchGoal)):
            return self.prepare_interaction(command, snapshot)
        if command.operation not in (Opcode.MOVE, Opcode.SHAPE_HAND):
            raise AdapterError("UNSUPPORTED_CAPABILITY", f"{command.operation} is not implemented by this adapter")
        if command.execution_domain != "free_space" or len(command.actors) != 1:
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Only single-actor free-space execution is implemented")
        actor = command.actors[0]
        if actor not in self._actors:
            raise AdapterError("UNSUPPORTED_CAPABILITY", f"Unknown actor: {actor}")
        r = self._actors[actor]
        if command.operation == Opcode.MOVE:
            goal = command.goal
            if r.spec.role != "arm" or not isinstance(goal, PoseGoal):
                raise AdapterError("UNSUPPORTED_CAPABILITY", "MOVE needs an arm binding and PoseGoal")
            if goal.target_type != "ACTOR_FRAME" or goal.reference_frame != "world" or goal.controlled_target != r.spec.frame_id:
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Only the actor's declared frame in world coordinates is supported")
        else:
            if r.spec.role != "hand" or not isinstance(command.goal, ShapeGoal):
                raise AdapterError("UNSUPPORTED_CAPABILITY", "SHAPE_HAND needs a hand binding and ShapeGoal")
            if len(command.goal.configuration) != len(r.joints):
                raise AdapterError("INVALID_ARGUMENT", "Configuration dimension differs from binding")
            limits = r.spec.articulation.data.joint_pos_limits[0, r.joints].cpu().tolist()
            if any(value < low or value > high for value, (low, high) in zip(command.goal.configuration, limits)):
                raise AdapterError("UNREACHABLE", "Configuration exceeds joint limits")
        robot_id = id(r.spec.articulation)
        # Free-space disjoint joint groups may run concurrently. Shared-object tasks
        # are unsupported; declared coupling groups remain mutually exclusive.
        return PreparedTask(frozenset(f"{robot_id}/joint/{j}" for j in r.joints),
                            frozenset(r.spec.coupling_groups), (f"joint_limits:{actor}",), actor)

    def command(self, command, prepared):
        if isinstance(prepared.binding, InteractionPlan):
            return self.command_interaction(command, prepared)
        r = self._actors[prepared.binding]
        self._drive_target(command, r, command.goal)

    def _drive_target(self, command, r, goal):
        robot, t = r.spec.articulation, self.torch
        current_q = robot.data.joint_pos[:, r.joints]
        if isinstance(goal, PoseGoal):
            p, q, _, _, world_offset = self._tcp(r)
            target = t.tensor([(*goal.position, *goal.orientation_wxyz)],
                              device=robot.device, dtype=current_q.dtype)
            # PhysX spatial Jacobians and goals are both expressed in world coordinates.
            jacobian = robot.root_physx_view.get_jacobians()[:, r.body - 1, :, r.joints].clone()
            jacobian[:, :3, :] -= self.math.skew_symmetric_matrix(world_offset) @ jacobian[:, 3:, :]
            if not t.isfinite(jacobian).all():
                raise AdapterError("EVIDENCE_UNAVAILABLE", "Nonfinite Jacobian")
            r.controller.set_command(target)
            candidate = r.controller.compute(p, q, jacobian, current_q)
        else:
            candidate = t.tensor([goal.configuration], device=robot.device, dtype=current_q.dtype)
        if not t.isfinite(candidate).all():
            raise AdapterError("SOLVER_FAILED", "Controller returned nonfinite joint targets")
        # Rate-limited command increments; actual velocity is measured separately.
        # Contact preload needs a bounded drive offset from actual position.
        # Rate-limit the command trajectory itself for interaction controllers;
        # limiting offset from actual q would impose an unintended force ceiling.
        reference_q = self._targets[id(robot)][:, r.joints] if command.operation in (
            Opcode.MAKE_CONTACT, Opcode.BREAK_CONTACT, Opcode.CONTROL_GRASP, Opcode.APPLY_WRENCH) else current_q
        increment = t.clamp(candidate - reference_q, -command.max_joint_speed*self.dt, command.max_joint_speed*self.dt)
        candidate = reference_q + increment
        limits = robot.data.joint_pos_limits[:, r.joints]
        candidate = t.maximum(t.minimum(candidate, limits[:, :, 1]), limits[:, :, 0])
        self._targets[id(robot)][:, r.joints] = candidate

    def hold(self, prepared):
        if isinstance(prepared.binding, InteractionPlan):
            # Retain the final bounded position targets, including existing preload.
            # This continuation is not a continuing grasp/wrench certificate.
            return
        r = self._actors[prepared.binding]
        q = r.spec.articulation.data.joint_pos[:, r.joints]
        if not self.torch.isfinite(q).all():
            raise AdapterError("EVIDENCE_UNAVAILABLE", "Cannot latch position from invalid joint state")
        self._targets[id(r.spec.articulation)][:, r.joints] = q

    def step(self):
        for key, robot in self._robots.items():
            robot.set_joint_position_target(self._targets[key])
        for hook in self.pre_step_hooks:
            hook()
        for robot in self._robots.values():
            robot.write_data_to_sim()
        self.sim.step(render=False)
        for robot in self._robots.values():
            robot.update(self.dt)
        for entity in self.entities.values():
            if id(entity) not in self._robots:
                entity.update(self.dt)
