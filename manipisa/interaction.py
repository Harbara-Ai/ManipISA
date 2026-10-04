"""Embodiment-independent contact evidence, wrench transforms and grasp checks.

The grasp certificate is a finite-load quasi-static friction-pyramid feasibility
test using OBSERVED preload, not a claim of arbitrary force closure or dynamics.
"""
import math
import numpy as np

from .types import (AdapterError, ContactGoal, GraspGoal, WrenchGoal, Evidence,
                    GoalCheck, Truth, Opcode, Mode, PoseGoal, ShapeGoal)


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def angle_between(a, b):
    return 2 * math.acos(float(np.clip(abs(np.dot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1)))


def valid_pose(p, command, state):
    return (p is not None and p.valid and len(p.position) == 3 and len(p.orientation_wxyz) == 4
            and len(p.linear_velocity) == 3 and len(p.angular_velocity) == 3
            and np.isfinite((*p.position, *p.orientation_wxyz, *p.linear_velocity, *p.angular_velocity)).all()
            and abs(np.linalg.norm(p.orientation_wxyz)-1) < 1e-3
            and Evidence(Truth.SATISFIED, p.timestamp, p.source).fresh(state.timestamp, command.max_observation_age_s))


def contact_valid(c, command, state, wrench=False):
    return (c is not None and c.valid and (not wrench or c.wrench_valid)
            and isinstance(c.present, bool) and math.isfinite(c.normal_force) and c.normal_force >= 0
            and len(c.force_on_target_w) == 3 and len(c.torque_on_target_world_origin_w) == 3
            and np.isfinite((*c.force_on_target_w, *c.torque_on_target_world_origin_w)).all()
            and Evidence(Truth.SATISFIED, c.timestamp, c.source).fresh(state.timestamp, command.max_observation_age_s))


def contact_check(requirement, command, state, *, separated=False):
    c = state.contacts.get(requirement.contact)
    if not contact_valid(c, command, state):
        return Evidence(Truth.UNKNOWN, state.timestamp, "contact", f"Invalid/missing/stale {requirement.contact}")
    if separated:
        if c.separation_m is None or not math.isfinite(c.separation_m):
            return Evidence(Truth.UNKNOWN, c.timestamp, c.source, "Separation geometry unavailable")
        ok = not c.present and c.separation_m >= requirement.separation_m
    else:
        ok = c.present and requirement.min_force <= c.normal_force <= requirement.max_force
    return Evidence(Truth.SATISFIED if ok else Truth.VIOLATED, c.timestamp, c.source,
                    f"{requirement.contact}: present={c.present}, normal_force={c.normal_force}, gap={c.separation_m}")


def wrench_at(goal, command, state):
    """Sum this interface only, transform moment origin before rotating axes."""
    cs = [state.contacts.get(r.contact) for r in goal.contacts]
    if not all(contact_valid(c, command, state, wrench=True) for c in cs):
        return None
    if goal.reference_frame == "world":
        R, origin, stamp = np.eye(3), np.zeros(3), state.timestamp
    else:
        frame = state.frames.get(goal.reference_frame)
        if not valid_pose(frame, command, state):
            return None
        R, origin, stamp = rotation(frame.orientation_wxyz), np.array(frame.position), frame.timestamp
    point_w = origin + R @ np.array(goal.reference_point)
    force = sum((np.array(c.force_on_target_w) for c in cs), np.zeros(3))
    moment = sum((np.array(c.torque_on_target_world_origin_w) for c in cs), np.zeros(3)) - np.cross(point_w, force)
    return np.r_[R.T @ force, R.T @ moment], min(stamp, *(c.timestamp for c in cs)), R, point_w


def grasp_capacity(goal, command, state):
    """Inner friction pyramids with normal budget no larger than measured preload.

    Assumptions: rigid point contacts, calibrated conservative Coulomb mu, and
    controllable redistribution within observed normal budgets. Explicit load
    cases only; dynamic/actuator robustness is outside this certificate.
    """
    from scipy.optimize import linprog
    obj = state.frames.get(goal.object_frame)
    if not valid_pose(obj, command, state):
        return None
    columns, groups, budgets = [], [], []
    for group, requirement in enumerate(goal.contacts):
        c = state.contacts.get(requirement.contact)
        if not contact_valid(c, command, state):
            return None
        budgets.append(c.normal_force)
        if c.present and not c.points:
            return None
        for point in c.points:
            n = np.array(point.normal_on_target_w, dtype=float)
            p = np.array(point.position_w, dtype=float)
            if n.shape != (3,) or p.shape != (3,) or not np.isfinite(np.r_[n,p,point.normal_force]).all() or abs(np.linalg.norm(n)-1) > 1e-3 or point.normal_force < 0:
                return None
            tangent = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(n, tangent)
            for direction in (tangent, -tangent, bitangent, -bitangent):
                ray = n + goal.friction_coefficient * direction
                columns.append(np.r_[ray, np.cross(p-np.array(obj.position), ray)])
                groups.append(group)
    if not columns:
        return False
    A = np.stack(columns, axis=1)
    if not np.isfinite(A).all():
        return None
    bounds = np.array([[float(g == i) for g in groups] for i in range(len(budgets))])
    for load in goal.load_cases:
        try:
            result = linprog(np.zeros(A.shape[1]), A_ub=bounds, b_ub=budgets,
                             A_eq=A, b_eq=-np.array(load), bounds=(0, None), method="highs")
        except (ValueError, RuntimeError):
            return None
        if result.status == 2:
            return False
        if not result.success or np.max(np.abs(A @ result.x + load)) > 1e-6:
            return None
    return True


def interaction_goal(command, state):
    goal = command.goal
    evidence = [contact_check(r, command, state, separated=command.operation == Opcode.BREAK_CONTACT) for r in goal.contacts]
    errors = {}
    if isinstance(goal, WrenchGoal):
        result = wrench_at(goal, command, state)
        if result is None:
            evidence.append(Evidence(Truth.UNKNOWN, state.timestamp, "contact_wrench", "Complete wrench unavailable"))
        else:
            measured, stamp, _, _ = result
            differences = np.abs(measured-np.array(goal.wrench))
            ok = all(d <= tol for d,tol,on in zip(differences, goal.tolerances, goal.controlled_axes) if on)
            evidence.append(Evidence(Truth.SATISFIED if ok else Truth.VIOLATED, stamp, "contact_wrench"))
            errors.update({f"wrench_error_{i}": float(d) for i,d in enumerate(differences) if goal.controlled_axes[i]})
    elif isinstance(goal, GraspGoal):
        obj, ref = state.frames.get(goal.object_frame), state.frames.get(goal.reference_frame)
        if not valid_pose(obj, command, state) or not valid_pose(ref, command, state):
            evidence.append(Evidence(Truth.UNKNOWN, state.timestamp, "grasp", "Relative state unavailable"))
        else:
            relative_linear = np.array(obj.linear_velocity) - ref.linear_velocity - np.cross(ref.angular_velocity, np.array(obj.position)-ref.position)
            linear, angular = float(np.linalg.norm(relative_linear)), float(np.linalg.norm(np.array(obj.angular_velocity)-ref.angular_velocity))
            errors.update(relative_linear_speed=linear, relative_angular_speed=angular)
            ok = linear <= goal.linear_speed_tolerance and angular <= goal.angular_speed_tolerance
            evidence.append(Evidence(Truth.SATISFIED if ok else Truth.VIOLATED, min(obj.timestamp, ref.timestamp), "relative_motion"))
            capacity = grasp_capacity(goal, command, state)
            evidence.append(Evidence(Truth.UNKNOWN if capacity is None else (Truth.SATISFIED if capacity else Truth.VIOLATED),
                                     min(obj.timestamp, *(e.timestamp for e in evidence)), "quasistatic_friction_pyramid",
                                     "Finite declared external load cases; measured preload budget"))
    truth = Truth.UNKNOWN if any(e.truth == Truth.UNKNOWN for e in evidence) else (Truth.SATISFIED if all(e.truth == Truth.SATISFIED for e in evidence) else Truth.VIOLATED)
    return GoalCheck(Evidence(truth, min(e.timestamp for e in evidence), "interaction_evidence",
                              "; ".join(e.detail for e in evidence if e.detail)), errors)


def validate_interaction(command):
    goal = command.goal
    expected = {Opcode.MAKE_CONTACT: ContactGoal, Opcode.BREAK_CONTACT: ContactGoal,
                Opcode.CONTROL_GRASP: GraspGoal, Opcode.APPLY_WRENCH: WrenchGoal}
    if command.operation not in expected:
        return
    if not isinstance(goal, expected[command.operation]):
        raise AdapterError("INVALID_ARGUMENT", f"{command.operation} requires {expected[command.operation].__name__}")
    if command.operation in (Opcode.MAKE_CONTACT, Opcode.BREAK_CONTACT) and command.mode != Mode.REACH:
        raise AdapterError("INVALID_ARGUMENT", "Contact topology changes require REACH")
    if isinstance(goal, WrenchGoal) and command.mode != Mode.SUSTAIN:
        raise AdapterError("INVALID_ARGUMENT", "APPLY_WRENCH requires SUSTAIN")
    def positive(values):
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise AdapterError("INVALID_ARGUMENT", "Interaction limits must be finite and positive")
    ids = [r.contact for r in goal.contacts]
    if not ids or len(ids) != len(set(ids)):
        raise AdapterError("INVALID_ARGUMENT", "Contact set must be nonempty and unique")
    all_ids = ids + [r.contact for r in goal.protected_contacts] + list(goal.forbidden_contacts)
    if any(not x for x in all_ids) or len(all_ids) != len(set(all_ids)):
        raise AdapterError("INVALID_ARGUMENT", "Target, protected and forbidden contacts must be disjoint and unique")
    for r in (*goal.contacts, *goal.protected_contacts):
        positive((r.min_force, r.max_force, r.separation_m))
        if r.min_force >= r.max_force:
            raise AdapterError("INVALID_ARGUMENT", "Contact min_force must be below max_force")
    if isinstance(goal, (ContactGoal, GraspGoal)):
        if set(m.actor for m in goal.motions) != set(command.actors) or len(goal.motions) != len(command.actors):
            raise AdapterError("INVALID_ARGUMENT", "Exactly one bounded motion per actor is required")
        for m in goal.motions:
            positive((m.max_translation, m.max_rotation, m.max_configuration_delta))
            if not isinstance(m.target, (PoseGoal, ShapeGoal)):
                raise AdapterError("INVALID_ARGUMENT", "Motion target requires a pose or configuration")
    if isinstance(goal, GraspGoal):
        positive((goal.friction_coefficient, goal.max_relative_translation, goal.max_relative_rotation,
                  goal.linear_speed_tolerance, goal.angular_speed_tolerance))
        if not goal.object_frame or not goal.reference_frame or goal.object_frame == goal.reference_frame:
            raise AdapterError("INVALID_ARGUMENT", "Grasp needs distinct object and reference frames")
        if not goal.load_cases or any(len(w) != 6 or not np.isfinite(w).all() for w in goal.load_cases) or not any(any(w) for w in goal.load_cases):
            raise AdapterError("INVALID_ARGUMENT", "Specify nonzero finite 6D external load cases")
    if isinstance(goal, WrenchGoal):
        for values in (goal.wrench, goal.tolerances, goal.admittance, goal.controlled_axes):
            if len(values) != 6:
                raise AdapterError("INVALID_ARGUMENT", "Wrench vectors/mask need six entries")
        if len(goal.reference_point) != 3 or not np.isfinite((*goal.wrench, *goal.reference_point)).all():
            raise AdapterError("INVALID_ARGUMENT", "Invalid wrench/reference point")
        if not all(isinstance(v, bool) for v in goal.controlled_axes) or not any(goal.controlled_axes):
            raise AdapterError("INVALID_ARGUMENT", "At least one controlled wrench direction is required")
        positive((*goal.tolerances, *goal.admittance, goal.max_translation, goal.max_rotation, goal.max_linear_speed, goal.max_angular_speed))
