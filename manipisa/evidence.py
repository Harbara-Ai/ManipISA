"""Goal checks use observed state, never the last motor command."""
import math
import numpy as np

from .types import Evidence, GoalCheck, PoseGoal, ShapeGoal, Truth, ContactGoal, GraspGoal, WrenchGoal


def evaluate_goal(command, snapshot) -> GoalCheck:
    goal = command.goal
    if isinstance(goal, (ContactGoal, GraspGoal, WrenchGoal)):
        from .interaction import interaction_goal
        return interaction_goal(command, snapshot)
    missing = GoalCheck(Evidence(Truth.UNKNOWN, snapshot.timestamp, "runtime", "Goal observation unavailable"))
    if isinstance(goal, PoseGoal):
        if goal.reference_frame != "world":
            return missing
        state = snapshot.frames.get(goal.controlled_target)
        if state is None or not state.valid:
            return missing
        values = (*state.position, *state.orientation_wxyz, *state.linear_velocity, *state.angular_velocity)
        if not all(math.isfinite(x) for x in values):
            return missing
        q = np.asarray(state.orientation_wxyz, dtype=float)
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > 1e-3:
            return missing
        target_q = np.asarray(goal.orientation_wxyz, dtype=float)
        target_q /= np.linalg.norm(target_q)
        angle = 2.0 * math.acos(float(np.clip(abs(np.dot(q / norm, target_q)), 0, 1)))
        errors = {
            "position_m": float(np.linalg.norm(np.asarray(state.position) - goal.position)),
            "orientation_rad": angle,
            "linear_speed_m_s": float(np.linalg.norm(state.linear_velocity)),
            "angular_speed_rad_s": float(np.linalg.norm(state.angular_velocity)),
        }
        limits = (goal.position_tolerance, goal.orientation_tolerance,
                  goal.linear_speed_tolerance, goal.angular_speed_tolerance)
    elif isinstance(goal, ShapeGoal):
        if len(command.actors) != 1:
            return missing
        state = snapshot.joints.get(command.actors[0])
        if state is None or not state.valid or len(state.position) != len(goal.configuration):
            return missing
        if len(state.velocity) != len(state.position) or not all(math.isfinite(x) for x in (*state.position, *state.velocity)):
            return missing
        errors = {"configuration_max_error": max(abs(a-b) for a,b in zip(state.position, goal.configuration)),
                  "configuration_max_speed": max(abs(x) for x in state.velocity)}
        limits = (goal.tolerance, goal.speed_tolerance)
    else:
        return missing
    evidence = Evidence(Truth.SATISFIED if all(e <= t for e,t in zip(errors.values(), limits)) else Truth.VIOLATED,
                        state.timestamp, state.source)
    if not evidence.fresh(snapshot.timestamp, command.max_observation_age_s):
        evidence = Evidence(Truth.UNKNOWN, state.timestamp, state.source, "Stale or future-dated goal observation")
    return GoalCheck(evidence, errors)
