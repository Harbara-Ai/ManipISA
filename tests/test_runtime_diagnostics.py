"""Actionable API diagnostics and executable documentation; no simulator needed."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import re
import time
import unittest
from unittest.mock import patch

from manipisa import Runtime, Instruction, Opcode, PoseGoal, ShapeGoal, Status, Evidence, Truth
from manipisa.types import AdapterError
from manipisa.evaluation.programs import execute_program, public_builtins
from test_runtime import TestAdapter, shape, finish


class LimitedTestAdapter(TestAdapter):
    def observe(self):
        state = super().observe()
        evidence = Evidence(Truth.SATISFIED if -.5 <= self.q <= .5 else Truth.VIOLATED,
                            state.timestamp, "test.current_joint_limits", f"Observed q={self.q}; limits=[-0.5, 0.5]")
        return replace(state, predicates={**state.predicates, "joint_limits:hand": evidence})

    def prepare(self, instruction, snapshot):
        prepared = super().prepare(instruction, snapshot)
        if isinstance(instruction.goal, ShapeGoal) and not -.5 <= instruction.goal.configuration[0] <= .5:
            raise AdapterError("UNREACHABLE", "Configuration exceeds joint limits")
        return replace(prepared, mandatory_invariants=("joint_limits:hand",))


class RuntimeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TestAdapter()
        self.runtime = Runtime(self.adapter)

    def test_query_and_cancel_reject_reports_and_dicts_before_observation(self):
        report = self.runtime.submit(shape())
        cases = ((report, "report.call_id"), (report.to_dict(), "report['call_id']"), ([], "call_id string"),
                 (12, "call_id string"), (None, "call_id string"))
        with patch.object(self.adapter, "observe", side_effect=AssertionError("Unexpected observation")):
            for operation in (self.runtime.query, self.runtime.cancel):
                for value, hint in cases:
                    with self.subTest(operation=operation.__name__, value=type(value).__name__):
                        with self.assertRaises(TypeError) as raised:
                            operation(value)
                        self.assertIn(hint, str(raised.exception))
                        self.assertNotIn("unhashable", str(raised.exception))
        self.assertEqual(self.runtime.query(report.call_id).status, Status.RUNNING)
        self.assertEqual(self.adapter.commands, 0)
        self.assertEqual(self.adapter.steps, 0)

    def test_unknown_call_and_control_handle_have_distinct_errors(self):
        report = self.runtime.submit(shape())
        terminal = finish(self.runtime, report.call_id)
        with patch.object(self.adapter, "observe", side_effect=AssertionError("Unexpected observation")):
            for operation in (self.runtime.query, self.runtime.cancel):
                with self.assertRaisesRegex(KeyError, "Unknown call_id"):
                    operation("not-a-call")
                with self.assertRaisesRegex(ValueError, "received a control_handle"):
                    operation(terminal.control_handle)
        self.assertEqual(self.runtime.query(report.call_id), terminal)
        self.assertIn(terminal.control_handle, self.runtime.feedback()["control_handles"])

    def test_update_preserves_result_shape_and_known_terminal_reason(self):
        report = self.runtime.submit(shape())
        with patch.object(self.adapter, "observe", side_effect=AssertionError("Unexpected observation")):
            for value in (report, report.to_dict(), None, "unknown", "hold_wrong"):
                result = self.runtime.update(value, shape())
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "INVALID_ARGUMENT")
                self.assertIn("call_id", result["detail"])
            result = self.runtime.update(report.call_id, {"goal": "not an Instruction"})
            self.assertEqual(result["reason"], "INVALID_ARGUMENT")
            self.assertIn("Instruction instance", result["detail"])
        terminal = finish(self.runtime, report.call_id)
        self.assertEqual(self.runtime.update(terminal.call_id, shape())["reason"], "TERMINAL_CALL")

    def test_replace_handle_explains_call_id_and_does_not_cancel(self):
        report = self.runtime.submit(shape())
        with patch.object(self.adapter, "observe", side_effect=AssertionError("Unexpected observation")):
            rejected = self.runtime.submit(shape(.2), replace_handle=report.call_id)
        self.assertEqual(rejected.status, Status.REJECTED)
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")
        self.assertIn("received a call_id", rejected.detail)
        self.assertIn("RUNNING", rejected.detail)
        self.assertEqual(self.runtime.query(report.call_id).status, Status.RUNNING)
        self.assertEqual(self.adapter.commands, 0)
        canceled = self.runtime.cancel(report.call_id)
        rejected = self.runtime.submit(shape(.2), replace_handle=canceled.call_id)
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")
        self.assertIn(canceled.control_handle, rejected.detail)
        self.assertIn(canceled.control_handle, self.runtime.feedback()["control_handles"])
        accepted = self.runtime.submit(shape(.2), replace_handle=canceled.control_handle)
        self.assertEqual(accepted.status, Status.RUNNING)
        self.assertNotIn(canceled.control_handle, self.runtime.feedback()["control_handles"])

    def test_replace_handle_wrong_types_unknown_and_resource_mismatch_preserve_reasons(self):
        report = self.runtime.submit(shape())
        canceled = self.runtime.cancel(report.call_id)
        for value, hint in ((canceled, "report.control_handle"), (canceled.to_dict(), "report['control_handle']")):
            rejected = self.runtime.submit(shape(.2), replace_handle=value)
            self.assertEqual(rejected.reason, "INVALID_ARGUMENT")
            self.assertIn(hint, rejected.detail)
        rejected = self.runtime.submit(shape(.2), replace_handle="hold_unknown")
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")
        self.assertIn("Unknown or already replaced", rejected.detail)
        rejected = self.runtime.submit(replace(shape(.2), actors=("other",)), replace_handle=canceled.control_handle)
        self.assertEqual(rejected.reason, "RESOURCE_CONFLICT")
        self.assertIn("exactly the retained resource set", rejected.detail)
        self.assertIn(canceled.control_handle, self.runtime.feedback()["control_handles"])

    def test_precondition_rejection_retains_truth_timestamp_source_and_detail(self):
        self.adapter.guard = Truth.UNKNOWN
        rejected = self.runtime.submit(shape(entry_guard=("entry",)))
        self.assertEqual(rejected.reason, "PRECONDITION_FAILED")
        evidence = rejected.evidence["entry"]
        self.assertEqual(evidence.truth, Truth.UNKNOWN)
        self.assertEqual(evidence.timestamp, self.adapter.now)
        self.assertEqual(evidence.source, "test_double")
        self.assertEqual(evidence.detail, "")
        self.assertIn("joint_speed:hand", rejected.evidence)
        self.assertIsNotNone(rejected.goal)
        rejected.evidence.clear()
        self.assertIn("entry", self.runtime.query(rejected.call_id).evidence)
        self.assertEqual(self.adapter.commands, 0)
        self.assertEqual(self.adapter.steps, 0)

    def test_goal_observation_unknown_keeps_rejection_reason_and_goal_evidence(self):
        self.adapter.valid = False
        rejected = self.runtime.submit(shape())
        self.assertEqual(rejected.reason, "EVIDENCE_UNAVAILABLE")
        self.assertEqual(rejected.goal.evidence.truth, Truth.UNKNOWN)
        self.assertEqual(rejected.evidence["joint_speed:hand"].truth, Truth.UNKNOWN)
        self.assertEqual(self.adapter.commands, 0)

    def test_current_state_limit_violation_is_distinct_from_invalid_target(self):
        adapter = LimitedTestAdapter()
        runtime = Runtime(adapter)
        target_rejected = runtime.submit(shape(target=.8))
        self.assertEqual(target_rejected.reason, "UNREACHABLE")
        self.assertIn("Requested goal rejected", target_rejected.detail)
        self.assertEqual(target_rejected.evidence, {})
        adapter.q = .8
        state_rejected = runtime.submit(shape(target=0.0))
        self.assertEqual(state_rejected.reason, "PRECONDITION_FAILED")
        self.assertIn("Current-state", state_rejected.detail)
        evidence = state_rejected.evidence["joint_limits:hand"]
        self.assertEqual(evidence.truth, Truth.VIOLATED)
        self.assertEqual(evidence.source, "test.current_joint_limits")
        self.assertIn("Observed q=0.8", evidence.detail)
        self.assertEqual(adapter.q, .8)
        self.assertEqual(adapter.commands, 0)
        self.assertEqual(adapter.steps, 0)

    def test_quickstart_code_blocks_execute_with_test_adapter(self):
        path = Path(__file__).resolve().parents[1] / "docs/runtime-quickstart.md"
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
        self.assertEqual(len(blocks), 3)

        def context():
            adapter = TestAdapter()
            runtime = Runtime(adapter)
            def step(count):
                for _ in range(count):
                    runtime.step()
            return {"__builtins__": public_builtins(), "runtime": runtime, "step": step, "Instruction": Instruction, "Opcode": Opcode,
                    "PoseGoal": PoseGoal, "ShapeGoal": ShapeGoal, "Status": Status,
                    "arm_actor": "arm", "hand_actor": "hand", "tool_frame": "tool",
                    "target_xyz": (.1, 0., 0.), "target_quat": (1., 0., 0., 0.),
                    "next_target_xyz": (.2, 0., 0.), "hand_target": (.1,), "step_budget": 60}

        namespace = context()
        with redirect_stdout(io.StringIO()):
            execute_program(blocks[0], namespace, time.perf_counter() + 5)
            self.assertEqual(namespace["move_result"].status, Status.SUCCEEDED)
            old_handle = namespace["move_result"].control_handle
            with patch.object(namespace["runtime"], "cancel", side_effect=AssertionError("Implicit cancellation")):
                execute_program(blocks[2], namespace, time.perf_counter() + 5)
            self.assertEqual(namespace["next_result"].status, Status.SUCCEEDED)
            self.assertNotIn(old_handle, namespace["runtime"].feedback()["control_handles"])
            namespace = context()
            execute_program(blocks[1], namespace, time.perf_counter() + 5)
            self.assertEqual(namespace["shape_result"].status, Status.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
