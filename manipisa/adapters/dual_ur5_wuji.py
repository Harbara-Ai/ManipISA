"""Embodiment binding for Bench2Dex's dual UR5 + dual Wuji articulation.

No IK or interaction controller is reimplemented here. Import is safe before
AppLauncher; USD/PhysX imports occur only when constructing a scene binding.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable
import math
import numpy as np

from manipisa.interaction import rotation
from manipisa.types import Pose
from .isaaclab import ActorBinding, IsaacLabAdapter
from .physx_contacts import ContactBinding, PhysXContactSource


ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
FINGERS = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class RigidTransform:
    """Pose of a child frame in its parent: metres, quaternion wxyz."""
    position: tuple[float, float, float] = (0., 0., 0.)
    orientation_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)

    def __post_init__(self):
        p, q = tuple(map(float, self.position)), tuple(map(float, self.orientation_wxyz))
        if len(p) != 3 or len(q) != 4 or not np.isfinite((*p,*q)).all() or abs(np.linalg.norm(q)-1) > 1e-3:
            raise ValueError("RigidTransform requires finite xyz and a normalized wxyz quaternion")
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "orientation_wxyz", tuple(np.asarray(q)/np.linalg.norm(q)))

    def compose(self, child: RigidTransform) -> RigidTransform:
        a, b = np.array(self.orientation_wxyz), np.array(child.orientation_wxyz)
        q = np.r_[a[0]*b[0]-a[1:]@b[1:], a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
        p = np.array(self.position) + rotation(a) @ child.position
        return RigidTransform(tuple(p),tuple(q))

    def inverse(self) -> RigidTransform:
        q = np.array(self.orientation_wxyz) * (1.,-1.,-1.,-1.)
        return RigidTransform(tuple(-rotation(q) @ self.position),tuple(q))


@dataclass(frozen=True)
class ToolFrame:
    # Preserve existing MOVE tool-frame semantics by default. Set parent="palm"
    # to command the physical hand palm or a calibrated tool attached to it.
    parent: str = "wrist"
    offset: RigidTransform = field(default_factory=RigidTransform)

    def __post_init__(self):
        if self.parent not in ("wrist", "palm") or not isinstance(self.offset,RigidTransform):
            raise ValueError("ToolFrame parent must be wrist or palm, with a RigidTransform offset")


@dataclass(frozen=True)
class SideBinding:
    arm_actor: str
    hand_actor: str
    tool_frame: str
    palm_frame: str
    tcp: ToolFrame = field(default_factory=ToolFrame)
    # Additional task/mechanical coupling declarations, passed to the backend.
    arm_coupling_groups: tuple[str, ...] = ()
    hand_coupling_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class DualUr5WujiConfig:
    right: SideBinding = field(default_factory=lambda:SideBinding("arm_a","hand_a","tool_a","palm_a"))
    left: SideBinding = field(default_factory=lambda:SideBinding("arm_b","hand_b","tool_b","palm_b"))

    def __post_init__(self):
        actors = [x for s in (self.right,self.left) for x in (s.arm_actor,s.hand_actor)]
        frames = [x for s in (self.right,self.left) for x in (s.tool_frame,s.palm_frame)]
        for values in (actors,frames):
            if any(not isinstance(v,str) or not v for v in values) or len(values) != len(set(values)):
                raise ValueError("Actor IDs and frame IDs must each be nonempty and unique")


@dataclass(frozen=True)
class WujiContact:
    name: str
    side: str
    region: str  # palm; thumb/index/middle/ring/little_tip or *_link1..4
    target: str  # key in the adapter's entities map
    target_path: str  # exact target rigid body path, verified against entity view
    controller: str = "hand"  # arm for arm-driven contact / APPLY_WRENCH
    separation: Callable[[float], tuple[float,float]] | None = None
    friction_lower_bound: float | None = None

    def __post_init__(self):
        if not self.name or not self.target or self.controller not in ("arm","hand"):
            raise ValueError("Contact needs a name, target and arm/hand controller")
        contact_body_name(self.side,self.region)
        _exact_path(self.target_path)
        if self.separation is not None and not callable(self.separation):
            raise ValueError("Separation must be a timestamped geometry observation callback")
        if self.friction_lower_bound is not None and (not math.isfinite(self.friction_lower_bound) or self.friction_lower_bound <= 0):
            raise ValueError("Friction lower bound must be finite and positive")


def _exact_path(path):
    if not isinstance(path,str) or not path.startswith("/") or any(c in path for c in "*?[]()|\\") or path.endswith("/"):
        raise ValueError("Expected an exact absolute USD prim path")
    return path


def contact_body_name(side, region):
    if side not in ("right","left"):
        raise ValueError("Side must be right or left")
    if region == "palm":
        return f"{side}_palm_link"
    for i, finger in enumerate(FINGERS,1):
        if region == finger+"_tip":
            return f"{side}_finger{i}_tip_link"
        for j in range(1,5):
            if region == f"{finger}_link{j}":
                return f"{side}_finger{i}_link{j}"
    raise ValueError(f"Unknown Wuji contact region: {region}")


def _body_paths(stage, robot_path):
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(_exact_path(robot_path))
    if not root.IsValid():
        raise ValueError(f"Robot prim missing: {robot_path}")
    paths = {}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
            if prim.GetName() in paths:
                raise ValueError(f"Ambiguous rigid body name: {prim.GetName()}")
            paths[prim.GetName()] = str(prim.GetPath())
    return paths


def _fixed_transform(stage, robot_path, start, end):
    """Compose joint-frame transforms along enabled FIXED joints only."""
    from pxr import Usd, UsdPhysics
    graph = {}
    def frame(joint, suffix):
        p = getattr(joint,f"GetLocalPos{suffix}Attr")().Get()
        q = getattr(joint,f"GetLocalRot{suffix}Attr")().Get()
        return RigidTransform(tuple(p),(q.GetReal(),*q.GetImaginary()))
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.IsA(UsdPhysics.FixedJoint):
            j = UsdPhysics.FixedJoint(prim)
            if not j.GetJointEnabledAttr().Get():
                continue
            a,b = j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()
            if len(a) == len(b) == 1:
                a,b = str(a[0]),str(b[0])
                transform = frame(j,0).compose(frame(j,1).inverse())
                graph.setdefault(a,[]).append((b,transform))
                graph.setdefault(b,[]).append((a,transform.inverse()))
    queue, visited = deque([(start,RigidTransform())]), {start}
    while queue:
        path, transform = queue.popleft()
        if path == end:
            return transform
        for neighbor, edge in graph.get(path,()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor,transform.compose(edge)))
    raise ValueError(f"No enabled fixed-joint chain from {start} to {end}")


def _actor_bindings(robot, config, wrist_to_palm):
    actors = {}
    for side in ("right","left"):
        spec = getattr(config,side)
        arm = ARM_JOINTS if side == "right" else tuple("L_arm_"+n for n in ARM_JOINTS)
        hand = tuple(f"{side}_finger{i}_joint{j}" for i in range(1,6) for j in range(1,5))
        wrist = "wrist_3_link" if side == "right" else "L_arm_wrist_3_link"
        offset = spec.tcp.offset if spec.tcp.parent == "wrist" else wrist_to_palm[side].compose(spec.tcp.offset)
        missing = set((*arm,*hand))-set(robot.joint_names)
        if missing:
            raise ValueError(f"Dual UR5/Wuji joint binding missing: {sorted(missing)}")
        actors[spec.arm_actor] = ActorBinding(robot,arm,"arm",wrist,spec.tool_frame,
            offset.position,offset.orientation_wxyz,spec.arm_coupling_groups)
        actors[spec.hand_actor] = ActorBinding(robot,hand,"hand",f"{side}_palm_link",spec.palm_frame,
            coupling_groups=spec.hand_coupling_groups)
    return actors


class DualUr5WujiAdapter(IsaacLabAdapter):
    """Construct after sim.reset(); use enable_contact_reports BEFORE reset.

    The caller owns spawning/reset, actuator gains, materials and gravity hooks.
    This adapter does not change those benchmark settings or advance physics.
    """
    @staticmethod
    def enable_contact_reports(stage, robot_path, contacts):
        from pxr import PhysxSchema
        paths = _body_paths(stage,robot_path)
        # Resolve the complete batch before changing the stage.
        selected = sorted({paths[contact_body_name(c.side,c.region)] for c in contacts})
        for path in selected:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        return tuple(selected)

    def __init__(self, sim, robot, *, config=None, contacts=(), entities=None,
                 pre_step_hooks=(), predicate_provider=None):
        from pxr import PhysxSchema, UsdGeom, UsdPhysics
        self.config = config or DualUr5WujiConfig()
        self.robot = robot
        self.robot_path = _exact_path(robot.cfg.prim_path)
        self.contact_specs = tuple(contacts)
        if len({c.name for c in self.contact_specs}) != len(self.contact_specs):
            raise ValueError("Contact IDs must be unique")
        stage = sim.stage
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.,abs_tol=1e-9):
            raise ValueError("DualUr5WujiAdapter expects a metre-based stage")
        self.body_paths = _body_paths(stage,self.robot_path)
        # Fixed-joint translations and PhysX link poses use rigid metre frames.
        # A scaled/mirrored asset needs a separate calibration, not silent reuse.
        cache = UsdGeom.XformCache()
        for name in ("wrist_3_link","L_arm_wrist_3_link","right_palm_link","left_palm_link"):
            matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.body_paths[name])),dtype=float)[:3,:3]
            if not np.allclose(matrix@matrix.T,np.eye(3),atol=1e-5,rtol=0) or np.linalg.det(matrix) < 0:
                raise ValueError("Scaled or mirrored robot link frames require explicit asset calibration")
        self.wrist_to_palm = {}
        for side,wrist in (("right","wrist_3_link"),("left","L_arm_wrist_3_link")):
            self.wrist_to_palm[side] = _fixed_transform(stage,self.robot_path,
                self.body_paths[wrist],self.body_paths[f"{side}_palm_link"])
        actors = _actor_bindings(robot,self.config,self.wrist_to_palm)
        bindings = {}
        entity_map = dict(entities or {})
        for c in self.contact_specs:
            body = contact_body_name(c.side,c.region)
            if body not in robot.body_names:
                raise ValueError(f"Contact body is not a resolved articulation link: {body}")
            sensor_path = self.body_paths[body]
            if not stage.GetPrimAtPath(sensor_path).HasAPI(PhysxSchema.PhysxContactReportAPI):
                raise ValueError("Enable contact reports before sim.reset() using DualUr5WujiAdapter.enable_contact_reports")
            if PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(sensor_path)).GetThresholdAttr().Get() != 0.:
                raise ValueError("Contact report threshold must be zero to observe low-force contacts")
            if c.target not in entity_map:
                raise ValueError(f"Contact target has no observed entity: {c.target}")
            target_paths = tuple(map(str,entity_map[c.target].root_physx_view.prim_paths))
            if target_paths != (c.target_path,):
                raise ValueError(f"Contact target path does not match the observed single rigid entity: {c.target}")
            target_prim = stage.GetPrimAtPath(c.target_path)
            if not target_prim.HasAPI(UsdPhysics.RigidBodyAPI) or c.target_path.startswith(self.robot_path+"/"):
                raise ValueError("Contact target must be an external rigid body")
            owner = getattr(self.config,c.side)
            bindings[c.name] = ContactBinding(owner.arm_actor if c.controller == "arm" else owner.hand_actor,
                c.target,sensor_path,c.target_path,c.separation,c.friction_lower_bound)
        if len({tuple(sorted((b.sensor_path,b.target_path))) for b in bindings.values()}) != len(bindings):
            raise ValueError("A physical contact pair must have one binding ID and one controller")
        super().__init__(sim,actors,entities=entity_map,pre_step_hooks=pre_step_hooks,
                         predicate_provider=predicate_provider)
        # Create PhysX views only after all pure bindings and backend validation.
        if bindings:
            self.contact_source = PhysXContactSource(sim,bindings)

    def body_pose(self, side, region) -> Pose:
        """Live link pose for calibrated region geometry; does not step physics."""
        index = self.robot.body_names.index(contact_body_name(side,region))
        state = self.robot.data.body_link_state_w[0,index]
        values = state.detach().cpu().tolist()
        return Pose(tuple(values[:3]),tuple(values[3:7]),tuple(values[7:10]),tuple(values[10:13]),
                    float(self.sim.current_time),bool(self.torch.isfinite(state).all()),"IsaacLab.body_link_state")

    def describe_bindings(self):
        """Reviewable binding metadata, separate from Agent task observations."""
        return {"adapter":type(self).__name__,"robot_path":self.robot_path,
            "sides":{side:asdict(getattr(self.config,side)) for side in ("right","left")},
            "wrist_to_palm":{side:asdict(tf) for side,tf in self.wrist_to_palm.items()},
            "actors":{a:{"joints":list(r.spec.joint_names),"body":r.spec.body_name,"frame":r.spec.frame_id,
                "tcp_position":r.spec.tcp_position,"tcp_orientation_wxyz":r.spec.tcp_orientation_wxyz}
                for a,r in self._actors.items()},
            "contacts":{name:{"actor":b.actor,"target":b.target,"sensor_path":b.sensor_path,"target_path":b.target_path,
                "has_separation":b.separation is not None,"friction_lower_bound":b.friction_lower_bound}
                for name,b in (self.contact_source.bindings.items() if self.contact_source else ())},
            "pre_step_hook_count":len(self.pre_step_hooks),"ik":"IsaacLab.DifferentialIKController/dls"}
