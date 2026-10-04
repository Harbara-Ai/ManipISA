"""Bounded interaction execution shared by the Isaac Lab adapter's six opcodes."""
from dataclasses import dataclass, field, replace
import numpy as np

from manipisa.types import (AdapterError, BoundedMotion, ContactGoal, GraspGoal, WrenchGoal,
    PoseGoal, ShapeGoal, PreparedTask, Evidence, Truth, Opcode)
from manipisa.interaction import (angle_between, rotation, valid_pose, contact_valid,
    contact_check, wrench_at, interaction_goal)


@dataclass
class InteractionPlan:
    actors: tuple[str, ...]
    joint_anchors: dict
    pose_anchors: dict
    relative_anchor: tuple | None = None
    servo_targets: dict = field(default_factory=dict)


class InteractionControl:
    def prepare_interaction(self, command, snapshot):
        goal = command.goal
        if command.execution_domain != "bounded_contact" or self.contact_source is None:
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Interaction needs bounded_contact domain and a structured contact source")
        cs = (*goal.contacts, *goal.protected_contacts)
        names = [r.contact for r in cs] + list(goal.forbidden_contacts)
        if any(name not in self.contact_source.bindings for name in names):
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Contact pair is not bound to a physical observation")
        physical_pairs = [self.contact_source.pair_key(n) for n in names]
        if len(physical_pairs) != len(set(physical_pairs)):
            raise AdapterError("INVALID_ARGUMENT", "Contact aliases cannot duplicate physical pairs")
        if any(self.contact_source.bindings[r.contact].actor not in command.actors for r in goal.contacts):
            raise AdapterError("INVALID_ARGUMENT", "Every target contact must be controlled by a declared actor")
        if isinstance(goal, GraspGoal) and any(self.contact_source.bindings[r.contact].target != goal.object_frame for r in goal.contacts):
            raise AdapterError("INVALID_ARGUMENT", "Grasp contact targets differ from object_frame")
        if isinstance(goal, GraspGoal):
            for requirement in goal.contacts:
                calibrated = self.contact_source.bindings[requirement.contact].friction_lower_bound
                if calibrated is None or goal.friction_coefficient > calibrated:
                    raise AdapterError("UNSUPPORTED_CAPABILITY", "Requested grasp friction exceeds the bound calibrated by the adapter")
        if isinstance(goal, WrenchGoal):
            if len(command.actors) != 1:
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Wrench admittance currently uses one controlled actor frame")
            r = self._actors.get(command.actors[0])
            if r is None or r.spec.role != "arm" or r.spec.frame_id != goal.controlled_target:
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Wrench control needs a bound arm frame")
            if len({self.contact_source.bindings[r.contact].target for r in goal.contacts}) != 1:
                raise AdapterError("INVALID_ARGUMENT", "Wrench interface must have one target entity")
            motions = (BoundedMotion(command.actors[0], PoseGoal(goal.controlled_target,
                       snapshot.frames[goal.controlled_target].position, snapshot.frames[goal.controlled_target].orientation_wxyz),
                       goal.max_translation, goal.max_rotation),)
        else:
            motions = goal.motions
        resources, couplings, mandatory = set(), set(), []
        joints, poses = {}, {}
        for motion in motions:
            sub = replace(command, operation=Opcode.MOVE if isinstance(motion.target, PoseGoal) else Opcode.SHAPE_HAND,
                          actors=(motion.actor,), goal=motion.target, execution_domain="free_space")
            prepared = self.prepare(sub, snapshot)
            if resources & prepared.resources:
                raise AdapterError("RESOURCE_CONFLICT", "Participants overlap physical joints")
            resources.update(prepared.resources)
            couplings.update(prepared.coupling_keys)
            mandatory.extend(prepared.mandatory_invariants)
            joints[motion.actor] = snapshot.joints[motion.actor].position
            if isinstance(motion.target, PoseGoal):
                p = snapshot.frames[motion.target.controlled_target]
                if not valid_pose(p, command, snapshot):
                    raise AdapterError("EVIDENCE_UNAVAILABLE", "Invalid bounded motion anchor")
                poses[motion.actor] = p
                if np.linalg.norm(np.array(p.position)-motion.target.position) > motion.max_translation or angle_between(p.orientation_wxyz, motion.target.orientation_wxyz) > motion.max_rotation:
                    raise AdapterError("ACQUISITION_LIMIT", "Pose target outside allowed local motion")
            elif max(abs(a-b) for a,b in zip(joints[motion.actor], motion.target.configuration)) > motion.max_configuration_delta:
                raise AdapterError("ACQUISITION_LIMIT", "Configuration target outside allowed local motion")
        for name in names:
            resources.add("contact/" + self.contact_source.pair_key(name))
            # Physical target body, never merely a user-facing entity alias.
            couplings.add("object/" + self.contact_source.target_key(name))
        relative = None
        if isinstance(goal, GraspGoal):
            obj, ref = snapshot.frames.get(goal.object_frame), snapshot.frames.get(goal.reference_frame)
            if not valid_pose(obj, command, snapshot) or not valid_pose(ref, command, snapshot):
                raise AdapterError("EVIDENCE_UNAVAILABLE", "Grasp object/reference state unavailable")
            relative = (rotation(ref.orientation_wxyz).T @ (np.array(obj.position)-ref.position),
                        rotation(ref.orientation_wxyz).T @ rotation(obj.orientation_wxyz))
        return PreparedTask(frozenset(resources), frozenset(couplings), tuple(mandatory),
                            InteractionPlan(command.actors, joints, poses, relative))

    def check(self, command, prepared, state, entering=False):
        if not isinstance(prepared.binding, InteractionPlan):
            return {}
        goal, plan = command.goal, prepared.binding
        evidence = {}
        def add(name, ok, reason, detail="", timestamp=None):
            evidence[name] = Evidence(Truth.UNKNOWN if ok is None else (Truth.SATISFIED if ok else Truth.VIOLATED),
                state.timestamp if timestamp is None else timestamp, "interaction_monitor", detail, reason)
        for requirement in (*goal.contacts, *goal.protected_contacts):
            c = state.contacts.get(requirement.contact)
            if not contact_valid(c, command, state):
                add("contact_state:"+requirement.contact, None, "EVIDENCE_UNAVAILABLE")
                continue
            add("force_limit:"+requirement.contact, c.normal_force <= requirement.max_force, "FORCE_LIMIT", timestamp=c.timestamp)
            if isinstance(goal, (GraspGoal, WrenchGoal)) or requirement in goal.protected_contacts:
                ok = c.present and c.normal_force > 0
                if requirement in goal.protected_contacts:
                    ok = contact_check(requirement, command, state).truth == Truth.SATISFIED
                add("keep_contact:"+requirement.contact, ok, "CONTACT_LOST", timestamp=c.timestamp)
        for name in goal.forbidden_contacts:
            c = state.contacts.get(name)
            add("forbidden_contact:"+name, None if not contact_valid(c, command, state) else not c.present,
                "UNEXPECTED_CONTACT", timestamp=c.timestamp if c else None)
        motions = goal.motions if isinstance(goal, (ContactGoal, GraspGoal)) else (
            BoundedMotion(command.actors[0], PoseGoal(goal.controlled_target, plan.pose_anchors[command.actors[0]].position,
            plan.pose_anchors[command.actors[0]].orientation_wxyz), goal.max_translation, goal.max_rotation),)
        for motion in motions:
            if isinstance(motion.target, PoseGoal):
                current, anchor = state.frames.get(motion.target.controlled_target), plan.pose_anchors[motion.actor]
                ok = None if not valid_pose(current, command, state) else (
                    np.linalg.norm(np.array(current.position)-anchor.position) <= motion.max_translation + 1e-6
                    and angle_between(current.orientation_wxyz, anchor.orientation_wxyz) <= motion.max_rotation + 1e-6)
            else:
                joint = state.joints.get(motion.actor)
                ok = None if joint is None or not joint.valid or not np.isfinite(joint.position).all() else (
                    max(abs(a-b) for a,b in zip(joint.position, plan.joint_anchors[motion.actor])) <= motion.max_configuration_delta+1e-6)
            add("acquisition:"+motion.actor, ok, "ACQUISITION_LIMIT")
        if isinstance(goal, GraspGoal):
            obj, ref = state.frames.get(goal.object_frame), state.frames.get(goal.reference_frame)
            if valid_pose(obj, command, state) and valid_pose(ref, command, state):
                R = rotation(ref.orientation_wxyz)
                p = R.T @ (np.array(obj.position)-ref.position)
                deltaR = plan.relative_anchor[1].T @ R.T @ rotation(obj.orientation_wxyz)
                angle = np.arccos(np.clip((np.trace(deltaR)-1)/2, -1, 1))
                ok = np.linalg.norm(p-plan.relative_anchor[0]) <= goal.max_relative_translation and angle <= goal.max_relative_rotation
                add("grasp_relative_motion", ok, "GRASP_INVALID", timestamp=min(obj.timestamp, ref.timestamp))
            else:
                add("grasp_relative_motion", None, "GRASP_INVALID")
            if entering and goal.require_valid_at_entry:
                check = interaction_goal(command, state)
                evidence["grasp_at_entry"] = replace(check.evidence, failure_reason="GRASP_INVALID")
        return evidence

    def command_interaction(self, command, prepared):
        goal = command.goal
        state = self.observe()
        if isinstance(goal, WrenchGoal):
            measured = wrench_at(goal, command, state)
            if measured is None:
                raise AdapterError("EVIDENCE_UNAVAILABLE", "Wrench feedback unavailable")
            actual, _, R, point_w = measured
            twist = np.array(goal.admittance) * (np.array(goal.wrench)-actual) * np.array(goal.controlled_axes)
            for part, limit in ((slice(0,3), goal.max_linear_speed), (slice(3,6), goal.max_angular_speed)):
                norm = np.linalg.norm(twist[part])
                twist[part] *= min(1., limit / max(norm, 1e-12))
            actor = command.actors[0]
            r = self._actors[actor]
            pose = state.frames[goal.controlled_target]
            nominal = prepared.binding.servo_targets.get(actor)
            if nominal is None:
                # Bumpless takeover: reconstruct the small existing PD offset
                # from the Jacobian and retained joint command. Starting at the
                # measured pose would discard preload and break the contact.
                robot = r.spec.articulation
                _, _, _, _, offset = self._tcp(r)
                J = robot.root_physx_view.get_jacobians()[:, r.body-1, :, r.joints].clone()
                J[:, :3, :] -= self.math.skew_symmetric_matrix(offset) @ J[:, 3:, :]
                dq = self._targets[id(robot)][:,r.joints] - robot.data.joint_pos[:,r.joints]
                delta = (J @ dq.unsqueeze(-1))[0,:,0].cpu().numpy()
                from scipy.spatial.transform import Rotation
                old_rotation = Rotation.from_quat(np.roll(pose.orientation_wxyz, -1))
                new_rotation = (Rotation.from_rotvec(delta[3:]) * old_rotation).as_quat()
                nominal = replace(pose, position=tuple(np.array(pose.position)+delta[:3]),
                                  orientation_wxyz=tuple(np.roll(new_rotation,1)))
            angular = R @ twist[3:]
            linear = R @ twist[:3] + np.cross(angular, np.array(pose.position)-point_w)
            # Rotation.from_rotvec uses world angular increments, premultiplied.
            from scipy.spatial.transform import Rotation
            old = Rotation.from_quat(np.roll(nominal.orientation_wxyz, -1))
            q = (Rotation.from_rotvec(angular*self.dt) * old).as_quat()
            target = PoseGoal(goal.controlled_target, tuple(np.array(nominal.position)+linear*self.dt), tuple(np.roll(q, 1)))
            anchor = prepared.binding.pose_anchors[actor]
            if np.linalg.norm(np.array(target.position)-anchor.position) > goal.max_translation or angle_between(target.orientation_wxyz, anchor.orientation_wxyz) > goal.max_rotation:
                raise AdapterError("ACQUISITION_LIMIT", "Admittance exhausted its allowed local motion")
            self._drive_target(command, r, target)
            prepared.binding.servo_targets[actor] = target
        else:
            for motion in goal.motions:
                if isinstance(goal, ContactGoal):
                    own = [r for r in goal.contacts if self.contact_source.bindings[r.contact].actor == motion.actor]
                    if own and all(contact_check(r, command, state, separated=command.operation == Opcode.BREAK_CONTACT).truth == Truth.SATISFIED for r in own):
                        continue  # retain the last PD target/preload during dwell
                self._drive_target(command, self._actors[motion.actor], motion.target)

    def relation(self, command, prepared, state):
        if not isinstance(command.goal, GraspGoal):
            return None
        goal = command.goal
        return {"object_frame": goal.object_frame, "reference_frame": goal.reference_frame,
                "contacts": [r.contact for r in goal.contacts], "load_cases_world_at_object_origin": goal.load_cases,
                "friction_coefficient": goal.friction_coefficient, "model": "quasistatic_inner_friction_pyramid",
                "timestamp": state.timestamp, "valid_at_timestamp_only": True,
                "scope": "Declared finite loads under rigid point-contact/controllable preload assumptions; not dynamic force closure"}
