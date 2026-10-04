"""Contract tests with a deterministic test double; these are not physics validation."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import numpy as np

from manipisa import (Runtime, Instruction, Opcode, Mode, ShapeGoal, PoseGoal,
                      Status, Evidence, Truth, JointState, Pose, StateSnapshot)
from manipisa.types import PreparedTask, AdapterError, TERMINAL
from manipisa.rgb import RGBFeedback


class TestAdapter:
    """One-dimensional deterministic plant for observing scheduler behavior only."""
    def __init__(self):
        self.now, self.dt = 0.0, 0.05
        self.q, self.velocity, self.target = 0.0, 0.0, 0.0
        self.speed = 1.0
        self.valid = True
        self.timestamp_override = None
        self.guard = Truth.SATISFIED
        self.invariant = Truth.SATISFIED
        self.commands = 0
        self.steps = 0
        self.stuck = False
        self.coupled = False
        self.unknown_during_prepare = False

    def observe(self):
        timestamp = self.now if self.timestamp_override is None else self.timestamp_override
        joint = JointState((self.q,), (self.velocity,), timestamp, self.valid, "test_double")
        pose = Pose((self.q, 0., 0.), (1., 0., 0., 0.), (self.velocity, 0., 0.), (0.,0.,0.), timestamp, self.valid, "test_double")
        return StateSnapshot(self.now, 0, {"hand": joint, "alias": joint, "arm": joint, "other": joint}, {"tool": pose}, {
            "entry": Evidence(self.guard, timestamp, "test_double"),
            "keep": Evidence(self.invariant, timestamp, "test_double"),
        }, state_source="test_double")

    def prepare(self, instruction, snapshot):
        if instruction.operation not in (Opcode.MOVE, Opcode.SHAPE_HAND):
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Not implemented in test double")
        if isinstance(instruction.goal, ShapeGoal) and len(instruction.goal.configuration) != 1:
            raise AdapterError("INVALID_ARGUMENT", "Expected one coordinate")
        if isinstance(instruction.goal, PoseGoal) and instruction.goal.target_type != "ACTOR_FRAME":
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Entity control is not implemented")
        if self.unknown_during_prepare:
            self.guard = Truth.UNKNOWN
        resource = "different_joint" if instruction.actors == ("other",) else "shared_joint"
        return PreparedTask(frozenset((resource,)), frozenset(("object",)) if self.coupled else frozenset())

    def command(self, instruction, prepared):
        self.commands += 1
        self.target = instruction.goal.configuration[0] if isinstance(instruction.goal, ShapeGoal) else instruction.goal.position[0]
        self.speed = instruction.max_joint_speed

    def hold(self, prepared):
        self.target = self.q

    def step(self):
        increment = 0. if self.stuck else float(np.clip(self.target-self.q, -self.speed*self.dt, self.speed*self.dt))
        self.q += increment
        self.velocity = increment / self.dt
        self.now = round(self.now + self.dt, 10)
        self.steps += 1


def shape(target=.10, **kwargs):
    return Instruction(Opcode.SHAPE_HAND, ("hand",), ShapeGoal((target,), tolerance=.001, speed_tolerance=.01),
                       max_joint_speed=1., **kwargs)


def move(target=.1, **kwargs):
    return Instruction(Opcode.MOVE, ("arm",), PoseGoal("tool", (target,0.,0.), (1.,0.,0.,0.),
        position_tolerance=.001, linear_speed_tolerance=.01), max_joint_speed=1., **kwargs)


def finish(runtime, call_id, limit=100):
    for _ in range(limit):
        if runtime.query(call_id).status in TERMINAL:
            return runtime.query(call_id)
        runtime.step()
    raise AssertionError("Test command did not terminate")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TestAdapter()
        self.runtime = Runtime(self.adapter)

    def test_reach_waits_for_real_observation_speed_and_dwell(self):
        call = self.runtime.submit(shape())
        self.assertEqual(call.status, Status.RUNNING)
        self.assertEqual(self.adapter.commands, 0)
        self.runtime.step()
        self.assertNotEqual(self.runtime.query(call.call_id).status, Status.SUCCEEDED)
        result = finish(self.runtime, call.call_id)
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertAlmostEqual(self.adapter.q, .1)
        self.assertGreaterEqual(result.timestamp, .2)
        self.assertIsNotNone(result.control_handle)

    def test_alias_cannot_overwrite_resource(self):
        self.runtime.submit(shape())
        rejected = self.runtime.submit(replace(shape(), actors=("alias",)))
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")
        self.assertEqual(self.adapter.commands, 0)

    def test_coupling_conflict_without_shared_joint(self):
        self.adapter.coupled = True
        self.runtime.submit(move())
        rejected = self.runtime.submit(replace(move(), actors=("other",)))
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")

    def test_unknown_entry_rejected_without_actuation(self):
        self.adapter.guard = Truth.UNKNOWN
        result = self.runtime.submit(shape(entry_guard=("entry",)))
        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.reason, "PRECONDITION_FAILED")
        self.assertEqual(self.adapter.commands, 0)

    def test_entry_rechecked_after_prepare(self):
        self.adapter.unknown_during_prepare = True
        result = self.runtime.submit(shape(entry_guard=("entry",)))
        self.assertEqual(result.status, Status.REJECTED)

    def test_entry_rechecked_before_first_control_tick(self):
        call = self.runtime.submit(shape(entry_guard=("entry",)))
        self.adapter.guard = Truth.VIOLATED
        self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).status, Status.FAILED)
        self.assertEqual(self.adapter.commands, 0)

    def test_actual_joint_speed_violation_is_detected(self):
        call = self.runtime.submit(shape())
        self.adapter.velocity = 2.
        self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).reason, "CONSTRAINT_VIOLATED")
        self.assertEqual(self.adapter.commands, 0)

    def test_invalid_nan_and_negative_tolerance_rejected(self):
        for request in (shape(float("nan")), replace(shape(), goal=ShapeGoal((.1,), tolerance=-1.)),
                        replace(shape(), timeout_s=float("inf"))):
            self.assertEqual(self.runtime.submit(request).status, Status.REJECTED)

    def test_stale_or_future_observation_rejected(self):
        for timestamp in (-1., 1.):
            self.adapter.timestamp_override = timestamp
            self.assertEqual(self.runtime.submit(shape()).reason, "EVIDENCE_UNAVAILABLE")

    def test_missing_observation_fails_and_does_not_recover(self):
        call = self.runtime.submit(shape())
        self.adapter.valid = False
        self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).reason, "EVIDENCE_UNAVAILABLE")
        self.adapter.valid = True
        for _ in range(4):
            self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).status, Status.FAILED)
        self.assertFalse(self.runtime.update(call.call_id, shape())['accepted'])

    def test_violation_latches_even_if_state_recovers(self):
        call = self.runtime.submit(shape(invariants=("keep",)))
        self.adapter.invariant = Truth.VIOLATED
        self.runtime.step()
        self.adapter.invariant = Truth.SATISFIED
        self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).reason, "CONSTRAINT_VIOLATED")

    def test_cached_observation_does_not_accumulate_dwell(self):
        self.adapter.timestamp_override = 0.
        call = self.runtime.submit(shape(0., dwell_s=.2, max_observation_age_s=1.))
        for _ in range(8):
            self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).status, Status.RUNNING)

    def test_cancel_retains_ownership_and_explicit_handoff(self):
        call = self.runtime.submit(shape())
        self.runtime.step()
        canceled = self.runtime.cancel(call.call_id)
        self.assertEqual(canceled.status, Status.CANCELED)
        self.assertIsNotNone(canceled.control_handle)
        self.assertEqual(self.runtime.submit(shape()).reason, "RESOURCE_CONFLICT")
        successor = self.runtime.submit(shape(.2), replace_handle=canceled.control_handle)
        self.assertEqual(successor.status, Status.RUNNING)
        self.assertEqual(finish(self.runtime, successor.call_id).status, Status.SUCCEEDED)

    def test_failed_handoff_preserves_old_owner(self):
        call = self.runtime.submit(shape())
        canceled = self.runtime.cancel(call.call_id)
        rejected = self.runtime.submit(replace(move(), actors=("other",)), replace_handle=canceled.control_handle)
        self.assertEqual(rejected.status, Status.REJECTED)
        self.assertIn(canceled.control_handle, self.runtime.feedback()["control_handles"])

    def test_sustain_is_active_before_completion(self):
        call = self.runtime.submit(move(0., mode=Mode.SUSTAIN, dwell_s=.05, sustain_s=.2))
        self.runtime.step()
        active = self.runtime.query(call.call_id)
        self.assertEqual(active.status, Status.ACTIVE)
        result = finish(self.runtime, call.call_id)
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertGreaterEqual(result.timestamp - result.active_since + 1e-9, .2)

    def test_active_goal_loss_fails(self):
        call = self.runtime.submit(move(0., mode=Mode.SUSTAIN, dwell_s=0.))
        self.runtime.step()
        self.adapter.q = .2
        self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).reason, "CONSTRAINT_VIOLATED")

    def test_activation_constraints_are_not_checked_as_entry_conditions(self):
        self.adapter.invariant = Truth.VIOLATED
        call = self.runtime.submit(move(mode=Mode.SUSTAIN, active_invariants=("keep",)))
        self.assertEqual(call.status, Status.RUNNING)
        result = finish(self.runtime, call.call_id)
        self.assertEqual(result.reason, "CONSTRAINT_VIOLATED")

    def test_timeout_and_activation_timeout(self):
        self.adapter.stuck = True
        call = self.runtime.submit(move(timeout_s=.1))
        self.assertEqual(finish(self.runtime, call.call_id).reason, "TIMEOUT")
        adapter = TestAdapter()
        adapter.stuck = True
        runtime = Runtime(adapter)
        call = runtime.submit(move(mode=Mode.SUSTAIN, activation_timeout_s=.1))
        self.assertEqual(finish(runtime, call.call_id).reason, "ACTIVATION_TIMEOUT")

    def test_update_preserves_time_budget_and_rejects_obligation_changes(self):
        self.adapter.stuck = True
        call = self.runtime.submit(shape(timeout_s=.2))
        self.runtime.step()
        self.assertFalse(self.runtime.update(call.call_id, shape(timeout_s=2.))['accepted'])
        self.assertTrue(self.runtime.update(call.call_id, shape(.3, timeout_s=.2))['accepted'])
        result = finish(self.runtime, call.call_id)
        self.assertEqual(result.reason, "TIMEOUT")
        self.assertLess(result.timestamp, .4)

    def test_hold_expiration_remains_visible_and_owned(self):
        call = self.runtime.submit(shape(0., dwell_s=0., hold_for_s=.1))
        result = finish(self.runtime, call.call_id)
        for _ in range(3):
            self.runtime.step()
        self.assertEqual(self.runtime.feedback()["control_handles"][result.control_handle]["state"], "EXPIRED")
        self.assertEqual(self.runtime.submit(shape()).reason, "RESOURCE_CONFLICT")

    def test_missing_interaction_operands_do_not_move(self):
        for operation in (Opcode.MAKE_CONTACT, Opcode.BREAK_CONTACT, Opcode.CONTROL_GRASP, Opcode.APPLY_WRENCH):
            result = self.runtime.submit(Instruction(operation, ("hand",), None))
            self.assertEqual(result.reason, "INVALID_ARGUMENT")
        self.assertEqual(self.adapter.commands, 0)

    def test_environment_and_entity_control_rejected(self):
        self.assertEqual(self.runtime.submit(replace(shape(), env_id=1)).reason, "UNSUPPORTED_CAPABILITY")
        request = move()
        self.assertEqual(self.runtime.submit(replace(request, goal=replace(request.goal, target_type="ENTITY_FRAME"))).reason,
                         "UNSUPPORTED_CAPABILITY")

    def test_json_feedback_has_no_evaluator_scores(self):
        self.runtime.submit(shape())
        decoded = json.loads(json.dumps(self.runtime.feedback(), allow_nan=False))
        self.assertNotIn("task_success", decoded)
        self.assertNotIn("stage_progress", decoded)
        self.assertEqual(decoded["state"]["state_source"], "test_double")

    def test_backend_observation_failure_marks_active_calls_failed(self):
        call = self.runtime.submit(shape())
        def broken_observe():
            raise RuntimeError("state stream failed")
        self.adapter.observe = broken_observe
        with self.assertRaises(RuntimeError):
            self.runtime.step()
        self.assertEqual(self.runtime.query(call.call_id).status, Status.FAILED)
        self.assertFalse(self.runtime.feedback()["state_valid"])

    def test_failed_continuation_cannot_report_success(self):
        def broken_hold(prepared):
            raise RuntimeError("actuator write failed")
        self.adapter.hold = broken_hold
        call = self.runtime.submit(shape(0., dwell_s=0.))
        result = finish(self.runtime, call.call_id)
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.reason, "CONTINUATION_FAILED")
        self.assertEqual(self.runtime.feedback()["control_handles"][result.control_handle]["state"], "FAULTED")


class CameraSource:
    camera_ids = ("a", "b")
    def __init__(self):
        self.calls = 0
        self.broken = False

    def capture(self, dt):
        self.calls += 1
        if self.broken:
            return {}
        return {key: SimpleNamespace(rgb=np.ones((4,6,3), dtype=np.uint8), intrinsic=np.eye(3),
                extrinsic_world_from_cam=np.eye(4), camera_model="pinhole") for key in self.camera_ids}


class RGBTests(unittest.TestCase):
    def test_disabled_channel_never_captures_and_clears_cached_pixels(self):
        source = CameraSource()
        rgb = RGBFeedback(source)
        rgb.sample(0.)
        self.assertEqual(source.calls, 0)
        rgb.set_enabled(True)
        rgb.sample(.1)
        self.assertEqual(len(rgb.frames()), 2)
        rgb.set_enabled(False)
        rgb.sample(.2)
        self.assertEqual(source.calls, 1)
        self.assertEqual(rgb.frames(), {})

    def test_sampling_timestamp_calibration_and_staleness(self):
        rgb = RGBFeedback(CameraSource(), enabled=True, period_s=.1, max_age_s=.2)
        rgb.sample(.1)
        rgb.sample(.15)
        self.assertEqual(rgb.capture_count, 1)
        metadata = rgb.metadata(.15)["frames"]["a"]
        self.assertEqual(metadata["timestamp"], .1)
        self.assertEqual(metadata["intrinsic"], np.eye(3).tolist())
        self.assertFalse(rgb.metadata(.4)["frames"]["a"]["valid"])

    def test_missing_frames_are_invalid_not_black_success(self):
        source = CameraSource()
        source.broken = True
        rgb = RGBFeedback(source, enabled=True)
        rgb.sample(.1)
        self.assertIsNone(rgb.frames()["a"].rgb)
        self.assertFalse(rgb.metadata(.1)["frames"]["a"]["valid"])

    def test_rgb_on_off_preserves_structured_control_trajectory(self):
        trajectories = []
        for enabled in (False, True):
            adapter = TestAdapter()
            runtime = Runtime(adapter, RGBFeedback(CameraSource(), enabled=enabled))
            call = runtime.submit(shape())
            trajectory = []
            for _ in range(10):
                trajectory.append(runtime.step().to_dict())
            trajectories.append(trajectory)
            self.assertEqual(runtime.query(call.call_id).status, Status.SUCCEEDED)
        self.assertEqual(trajectories[0], trajectories[1])

    def test_render_failure_does_not_change_instruction_result(self):
        class BrokenSource(CameraSource):
            def capture(self, dt):
                raise RuntimeError("render failed")
        runtime = Runtime(TestAdapter(), RGBFeedback(BrokenSource(), enabled=True))
        call = runtime.submit(shape())
        self.assertEqual(finish(runtime, call.call_id).status, Status.SUCCEEDED)
        self.assertFalse(runtime.feedback()["rgb"]["frames"]["a"]["valid"])


if __name__ == "__main__":
    unittest.main()
