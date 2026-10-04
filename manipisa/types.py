"""Shared contracts; names, dimensions and frames are resolved by an adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math
import numbers
from typing import Any


class Opcode(StrEnum):
    SHAPE_HAND = "SHAPE_HAND"
    MAKE_CONTACT = "MAKE_CONTACT"
    BREAK_CONTACT = "BREAK_CONTACT"
    MOVE = "MOVE"
    CONTROL_GRASP = "CONTROL_GRASP"
    APPLY_WRENCH = "APPLY_WRENCH"


class Mode(StrEnum):
    REACH = "REACH"
    SUSTAIN = "SUSTAIN"


class Status(StrEnum):
    RUNNING = "RUNNING"
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


TERMINAL = frozenset((Status.SUCCEEDED, Status.FAILED, Status.REJECTED, Status.CANCELED))


class Truth(StrEnum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Evidence:
    truth: Truth
    timestamp: float
    source: str
    detail: str = ""
    failure_reason: str = "CONSTRAINT_VIOLATED"

    def fresh(self, now: float, max_age: float) -> bool:
        return math.isfinite(self.timestamp) and -1e-9 <= now - self.timestamp <= max_age + 1e-9


@dataclass(frozen=True)
class JointState:
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    timestamp: float
    valid: bool = True
    source: str = "simulator_joint_state"
    coordinate_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    timestamp: float
    valid: bool = True
    source: str = "simulator_body_state"


@dataclass(frozen=True)
class StateSnapshot:
    timestamp: float
    env_id: int
    joints: dict[str, JointState] = field(default_factory=dict)
    frames: dict[str, Pose] = field(default_factory=dict)
    predicates: dict[str, Evidence] = field(default_factory=dict)
    # Explicitly identify privileged observations; no evaluator fields are accepted.
    state_source: str = "simulation_privileged"
    contacts: dict[str, ContactState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class PoseGoal:
    controlled_target: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    reference_frame: str = "world"
    target_type: str = "ACTOR_FRAME"
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    linear_speed_tolerance: float = 0.04
    angular_speed_tolerance: float = 0.15


@dataclass(frozen=True)
class ShapeGoal:
    # Configuration coordinates are defined by the actor binding, not a fixed hand model.
    configuration: tuple[float, ...]
    tolerance: float = 0.015
    speed_tolerance: float = 0.10


@dataclass(frozen=True)
class ContactPoint:
    position_w: tuple[float, float, float]
    # Unit normal pointing in the direction the actor pushes the target.
    normal_on_target_w: tuple[float, float, float]
    normal_force: float


@dataclass(frozen=True)
class ContactState:
    actor: str
    target: str
    timestamp: float
    present: bool
    normal_force: float
    # Complete normal + friction resultant ON TARGET, world origin moment.
    force_on_target_w: tuple[float, float, float]
    torque_on_target_world_origin_w: tuple[float, float, float]
    points: tuple[ContactPoint, ...] = ()
    separation_m: float | None = None
    valid: bool = True
    wrench_valid: bool = False
    source: str = "physics_contact"
    detail: str = ""


@dataclass(frozen=True)
class ContactRequirement:
    contact: str
    min_force: float = 0.05
    max_force: float = 50.0
    separation_m: float = 0.002


@dataclass(frozen=True)
class BoundedMotion:
    actor: str
    target: PoseGoal | ShapeGoal
    # Bounds measured from the accepted initial ACTUAL state, not new commands.
    max_translation: float = 0.05
    max_rotation: float = 0.20
    max_configuration_delta: float = 0.20


@dataclass(frozen=True)
class ContactGoal:
    contacts: tuple[ContactRequirement, ...]
    motions: tuple[BoundedMotion, ...]
    protected_contacts: tuple[ContactRequirement, ...] = ()
    forbidden_contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspGoal:
    contacts: tuple[ContactRequirement, ...]
    motions: tuple[BoundedMotion, ...]
    object_frame: str
    reference_frame: str
    # Explicit finite external-wrench load cases at the object origin, world axes.
    # Capacity is certified ONLY under the declared quasi-static point-contact model.
    load_cases: tuple[tuple[float, float, float, float, float, float], ...]
    friction_coefficient: float
    max_relative_translation: float = 0.005
    max_relative_rotation: float = 0.10
    linear_speed_tolerance: float = 0.02
    angular_speed_tolerance: float = 0.10
    require_valid_at_entry: bool = False
    protected_contacts: tuple[ContactRequirement, ...] = ()
    forbidden_contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class WrenchGoal:
    contacts: tuple[ContactRequirement, ...]
    controlled_target: str
    wrench: tuple[float, float, float, float, float, float]
    reference_frame: str
    reference_point: tuple[float, float, float]
    controlled_axes: tuple[bool, bool, bool, bool, bool, bool] = (True, True, True, True, True, True)
    tolerances: tuple[float, float, float, float, float, float] = (0.5, 0.5, 0.5, 0.05, 0.05, 0.05)
    admittance: tuple[float, float, float, float, float, float] = (0.001, 0.001, 0.001, 0.01, 0.01, 0.01)
    max_translation: float = 0.02
    max_rotation: float = 0.10
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.10
    protected_contacts: tuple[ContactRequirement, ...] = ()
    forbidden_contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Instruction:
    operation: Opcode
    actors: tuple[str, ...]
    goal: PoseGoal | ShapeGoal | ContactGoal | GraspGoal | WrenchGoal | None
    mode: Mode = Mode.REACH
    env_id: int = 0
    dwell_s: float = 0.10
    timeout_s: float = 5.0
    activation_timeout_s: float = 3.0
    sustain_s: float | None = None
    max_observation_age_s: float = 0.10
    max_joint_speed: float = 0.30
    entry_guard: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    active_invariants: tuple[str, ...] = ()
    # MVP retains position control on every terminal outcome; handoff is explicit.
    continuation: str = "HOLD_POSITION"
    hold_for_s: float = 2.0
    # On expiry, the lease is faulted and remains reserved with position hold.
    # This is an explicit fallback, not a claim that the old task remains satisfied.
    hold_expiry: str = "FAULT_AND_HOLD"
    execution_domain: str = "free_space"


@dataclass(frozen=True)
class PreparedTask:
    resources: frozenset[str]
    coupling_keys: frozenset[str]
    mandatory_invariants: tuple[str, ...] = ()
    binding: Any = None


@dataclass(frozen=True)
class GoalCheck:
    evidence: Evidence
    errors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReport:
    call_id: str
    operation: Opcode
    status: Status
    timestamp: float
    reason: str | None = None
    detail: str = ""
    goal: GoalCheck | None = None
    evidence: dict[str, Evidence] = field(default_factory=dict)
    active_since: float | None = None
    control_handle: str | None = None
    relation: dict | None = None

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


class AdapterError(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def json_safe(value):
    """Strict JSON: unknown numeric observations serialize as null, never NaN."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, int)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return value


def validate_instruction(command: Instruction) -> None:
    if not isinstance(command.operation, Opcode) or not isinstance(command.mode, Mode):
        raise AdapterError("INVALID_ARGUMENT", "Use an Opcode and Mode value")
    if not command.actors or len(set(command.actors)) != len(command.actors):
        raise AdapterError("INVALID_ARGUMENT", "Actors must be nonempty and unique")
    for name in ("timeout_s", "activation_timeout_s", "max_observation_age_s", "max_joint_speed", "hold_for_s"):
        value = getattr(command, name)
        if not math.isfinite(value) or value <= 0:
            raise AdapterError("INVALID_ARGUMENT", f"{name} must be finite and positive")
    if not math.isfinite(command.dwell_s) or command.dwell_s < 0:
        raise AdapterError("INVALID_ARGUMENT", "dwell_s must be finite and nonnegative")
    if command.sustain_s is not None and (not math.isfinite(command.sustain_s) or command.sustain_s < 0):
        raise AdapterError("INVALID_ARGUMENT", "sustain_s must be finite and nonnegative")
    if command.mode == Mode.REACH and command.sustain_s is not None:
        raise AdapterError("INVALID_ARGUMENT", "sustain_s requires SUSTAIN")
    if command.continuation != "HOLD_POSITION" or command.hold_expiry != "FAULT_AND_HOLD":
        raise AdapterError("UNSUPPORTED_CAPABILITY", "Only explicit position-hold continuation is implemented")
    if command.operation == Opcode.SHAPE_HAND and command.mode != Mode.REACH:
        raise AdapterError("UNSUPPORTED_CAPABILITY", "SHAPE_HAND supports REACH only")
    if command.operation == Opcode.MOVE and not isinstance(command.goal, PoseGoal):
        raise AdapterError("INVALID_ARGUMENT", "MOVE requires PoseGoal")
    if command.operation == Opcode.SHAPE_HAND and not isinstance(command.goal, ShapeGoal):
        raise AdapterError("INVALID_ARGUMENT", "SHAPE_HAND requires ShapeGoal")
    goal = command.goal
    if isinstance(goal, PoseGoal):
        if len(goal.position) != 3 or len(goal.orientation_wxyz) != 4:
            raise AdapterError("INVALID_ARGUMENT", "Pose needs 3 position and 4 wxyz components")
        if not all(math.isfinite(v) for v in (*goal.position, *goal.orientation_wxyz)):
            raise AdapterError("INVALID_ARGUMENT", "Pose must be finite")
        if abs(sum(v*v for v in goal.orientation_wxyz) - 1.0) > 1e-3:
            raise AdapterError("INVALID_ARGUMENT", "Quaternion must be normalized (wxyz)")
        tolerances = (goal.position_tolerance, goal.orientation_tolerance,
                      goal.linear_speed_tolerance, goal.angular_speed_tolerance)
    elif isinstance(goal, ShapeGoal):
        if not goal.configuration or not all(math.isfinite(v) for v in goal.configuration):
            raise AdapterError("INVALID_ARGUMENT", "Configuration must be nonempty and finite")
        tolerances = (goal.tolerance, goal.speed_tolerance)
    else:
        tolerances = ()
    if any(not math.isfinite(v) or v <= 0 for v in tolerances):
        raise AdapterError("INVALID_ARGUMENT", "Goal tolerances must be finite and positive")
    from .interaction import validate_interaction
    validate_interaction(command)
    if isinstance(goal, (ContactGoal, GraspGoal)):
        from dataclasses import replace
        for motion in goal.motions:
            validate_instruction(replace(command, operation=Opcode.MOVE if isinstance(motion.target, PoseGoal) else Opcode.SHAPE_HAND,
                                         actors=(motion.actor,), goal=motion.target, mode=Mode.REACH, sustain_s=None))
