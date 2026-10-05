"""Response-boundary regression tests; no simulator or model is launched."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from manipisa import Runtime, Instruction, Opcode, ShapeGoal
from manipisa.evaluation.programs import EpisodeStopped, execute_program, public_builtins
from manipisa.evaluation.tool_feedback import execute_with_feedback, stdout_preview, STDOUT_LIMIT, FEEDBACK_FORMAT
from test_runtime import TestAdapter


class Episode:
    def __init__(self, out, method="manipisa"):
        self.out = Path(out)
        self.method = method
        self.steps = 0
        self.tool_calls = 1
        self.started = time.perf_counter()
        self.wall_limit = 60
        self.runtime = Runtime(TestAdapter())
        self.terminate_on_step = False
        self.visual_captures = 0
        self.namespace = {"__builtins__": public_builtins(), "step": self.step,
                          "runtime": SimpleNamespace(feedback=self.runtime.feedback,
                                                     query=self.runtime.query)}

    def step(self, count):
        for _ in range(count):
            self.runtime.step()
            self.steps += 1
            if self.terminate_on_step:
                raise EpisodeStopped("max_steps")

    def check_time(self):
        if time.perf_counter() >= self.started + self.wall_limit:
            raise EpisodeStopped("wall_timeout")

    def capture_visual_feedback(self):
        self.visual_captures += 1
        camera_ids = ["cam_overhead", "cam_wrist_left", "cam_wrist_right"]
        metadata = {"status": "captured", "observation_id": self.visual_captures,
                    "physics_step": self.steps, "camera_ids": camera_ids,
                    "cameras": {name: {"physics_step": self.steps} for name in camera_ids},
                    "state": {"objects": {"ball": {"center": [0.0, 0.0, 1.0]}}}}
        images = [{"type": "image", "mimeType": "image/png", "data": "mock-image-payload-" + name}
                  for name in camera_ids]
        return metadata, images


class ToolFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.episode = Episode(self.folder.name)

    def call(self, code):
        reply = execute_with_feedback(self.episode, code)
        return reply, json.loads(reply["content"][0]["text"])

    def records(self):
        return [json.loads(line) for line in
                (self.episode.out / "tool-feedback.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_error_preserves_partial_motion_stdout_and_current_instruction(self):
        report = self.episode.runtime.submit(Instruction(Opcode.SHAPE_HAND, ("hand",),
                                                       ShapeGoal((1.,)), max_joint_speed=1.))
        reply, result = self.call("step(2)\nprint('moved already')\n1 / 0")
        self.assertTrue(reply["isError"])
        self.assertEqual(result["error"]["type"], "ZeroDivisionError")
        self.assertEqual(result["physics_step"], 2)
        self.assertEqual(result["stdout"], "moved already\n")
        self.assertEqual(result["runtime"]["instructions"][0]["call_id"], report.call_id)
        record = self.records()[0]
        self.assertEqual(record["physics_step_before"], 0)
        self.assertEqual(record["physics_step"], 2)
        self.assertEqual(record["runtime"], self.episode.runtime.feedback())
        self.assertEqual(self.episode.visual_captures, 1)
        self.assertEqual(result["visual_feedback"]["physics_step"], 2)
        self.assertEqual(len(reply["content"]), 4)

    def test_long_stdout_is_marked_and_full_text_is_logged_for_both_methods(self):
        for method in ("direct", "manipisa"):
            self.episode.method = method
            reply, result = self.call("print('BEGIN' + 'x' * 8000 + 'END')")
            self.assertNotIn("isError", reply)
            self.assertEqual(len(result["stdout"]), STDOUT_LIMIT)
            self.assertTrue(result["stdout"].startswith("BEGIN"))
            self.assertTrue(result["stdout"].endswith("END\n"))
            self.assertIn("stdout truncated", result["stdout"])
            self.assertGreater(result["stdout_omitted_chars"], 0)
            self.assertEqual(self.records()[-1]["stdout"], "BEGIN" + "x" * 8000 + "END\n")
            self.assertEqual("runtime" in result, method == "manipisa")

    def test_syntax_and_validation_errors_also_return_feedback(self):
        for code, kind in (("if:", "SyntaxError"), ("import os", "ValueError")):
            reply, result = self.call(code)
            self.assertTrue(reply["isError"])
            self.assertEqual(result["error"]["type"], kind)
            self.assertEqual(result["stdout"], "")
            self.assertIn("runtime", result)
            self.assertEqual(result["physics_step"], 0)
            self.assertEqual(result["visual_feedback"], {"status": "unchanged", "physics_step": 0})
        self.assertEqual(self.episode.visual_captures, 0)

    def test_terminal_boundary_records_partial_output_without_policy_reply(self):
        self.episode.terminate_on_step = True
        with self.assertRaisesRegex(EpisodeStopped, "max_steps"):
            self.call("print('before stop')\nstep(2)\nprint('must not run')")
        record = self.records()[0]
        self.assertTrue(record["reply_withheld"])
        self.assertEqual(record["stdout"], "before stop\n")
        self.assertEqual(record["physics_step"], 1)
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))
        self.assertEqual(self.episode.visual_captures, 0)
        self.assertEqual(record["visual_feedback"]["status"], "withheld")

    def test_ordinary_query_does_not_consume_host_baseline_or_change_physics(self):
        report = self.episode.runtime.submit(Instruction(Opcode.SHAPE_HAND, ("missing",),
                                                       ShapeGoal((1.,))))
        self.episode.namespace["call_id"] = report.call_id
        _, first = self.call("runtime.query(call_id)")
        self.assertEqual([r["call_id"] for r in first["runtime"]["instructions"]], [report.call_id])
        _, second = self.call("runtime.feedback()\nruntime.query(call_id)")
        self.assertEqual(second["runtime"]["instructions"], [])
        self.assertEqual(len(self.records()[-1]["runtime"]["instructions"]), 1)
        self.assertEqual(self.episode.steps, 0)
        self.assertEqual(self.episode.visual_captures, 0)
        self.assertEqual(second["visual_feedback"], {"status": "unchanged", "physics_step": 0})

    def test_deadline_crossed_after_execution_still_withholds_reply(self):
        def expire():
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
        self.episode.namespace["expire"] = expire
        with self.assertRaises(EpisodeStopped):
            self.call("print('done')\nexpire()")
        self.assertTrue(self.records()[0]["reply_withheld"])
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_deadline_crossed_during_logging_appends_withheld_correction(self):
        with patch.object(self.episode, "check_time", side_effect=[None, EpisodeStopped("wall_timeout")]):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("print('done')")
        self.assertEqual(self.records()[-1]["event"], "reply_withheld_after_logging")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_terminal_exception_survives_feedback_or_log_failure(self):
        self.episode.terminate_on_step = True
        for target, name in ((self.episode.runtime, "feedback"), (Path, "open")):
            with self.subTest(name=name), patch.object(target, name, side_effect=OSError("audit unavailable")):
                with self.assertRaisesRegex(EpisodeStopped, "max_steps"):
                    self.call("step(1)")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_logging_failure_does_not_consume_new_terminal_report(self):
        report = self.episode.runtime.submit(Instruction(Opcode.SHAPE_HAND, ("missing",), ShapeGoal((1.,))))
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.call("pass")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))
        _, result = self.call("pass")
        self.assertEqual(result["runtime"]["instructions"][0]["call_id"], report.call_id)

    def test_capture_is_opt_in_and_restores_trace_on_error(self):
        capture = {}
        trace = sys.gettrace()
        with self.assertRaises(ZeroDivisionError):
            execute_program("print('before')\n1 / 0", self.episode.namespace,
                            time.perf_counter() + 2, capture=capture)
        self.assertEqual(capture["stdout"], "before\n")
        self.assertIs(sys.gettrace(), trace)
        self.assertEqual(stdout_preview("small"), ("small", 0))

    def test_motion_captures_once_at_final_step_for_both_methods_without_payload_in_log(self):
        for method in ("direct", "manipisa"):
            with self.subTest(method=method):
                self.episode = Episode(self.folder.name, method=method)
                reply, result = self.call("step(2)\nstep(3)")
                self.assertEqual(self.episode.steps, 5)
                self.assertEqual(self.episode.runtime.adapter.steps, 5)
                self.assertEqual(self.episode.visual_captures, 1)
                self.assertEqual(FEEDBACK_FORMAT, "compact-visual-v2")
                self.assertEqual(result["feedback_format"], FEEDBACK_FORMAT)
                self.assertEqual(result["visual_feedback"]["physics_step"], 5)
                self.assertEqual(result["visual_feedback"]["status"], "captured")
                self.assertNotIn("isError", reply)
                images = reply["content"][1:]
                self.assertEqual(len(images), 3)
                self.assertTrue(all(block["type"] == "image" for block in images))
                self.assertEqual([i["data"].removeprefix("mock-image-payload-") for i in images],
                                 result["visual_feedback"]["camera_ids"])
                record = self.records()[-1]
                self.assertEqual(record["visual_feedback"], result["visual_feedback"])
                self.assertEqual(record["image_count"], 3)
                self.assertNotIn("mock-image-payload-", json.dumps(record))

    def test_expired_budget_before_visual_capture_does_not_render(self):
        with patch.object(self.episode, "check_time", side_effect=EpisodeStopped("wall_timeout")):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)")
        self.assertEqual(self.episode.visual_captures, 0)
        self.assertTrue(self.records()[-1]["reply_withheld"])
        self.assertEqual(self.records()[-1]["visual_feedback"]["status"], "withheld")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_visual_capture_terminal_exception_is_not_an_ordinary_tool_error(self):
        with patch.object(self.episode, "capture_visual_feedback", side_effect=EpisodeStopped("wall_timeout")) as camera:
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)")
        camera.assert_called_once_with()
        self.assertTrue(self.records()[-1]["reply_withheld"])
        self.assertEqual(self.records()[-1]["image_count"], 0)
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_budget_exhausted_during_successful_capture_withholds_saved_images(self):
        capture = self.episode.capture_visual_feedback
        def finish_after_deadline():
            result = capture()
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            return result
        with patch.object(self.episode, "capture_visual_feedback", side_effect=finish_after_deadline):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)\nprint('moved')")
        record = self.records()[-1]
        self.assertTrue(record["reply_withheld"])
        self.assertEqual(record["visual_feedback"]["status"], "captured")
        self.assertEqual(record["image_count"], 3)
        self.assertEqual(record["stdout"], "moved\n")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_camera_failure_returns_structured_error_without_stale_images(self):
        self.call("step(1)")
        with patch.object(self.episode, "capture_visual_feedback", side_effect=RuntimeError("camera unavailable")):
            reply, result = self.call("step(1)\nprint('second movement')")
        self.assertTrue(reply["isError"])
        self.assertNotIn("error", result)
        self.assertEqual(result["physics_step"], 2)
        self.assertEqual(result["stdout"], "second movement\n")
        self.assertEqual(result["visual_feedback"], {"status": "error", "physics_step": 2,
                         "error": {"type": "RuntimeError", "message": "camera unavailable"}})
        self.assertEqual(len(reply["content"]), 1)
        self.assertEqual(self.records()[-1]["image_count"], 0)

    def test_camera_failure_does_not_replace_partial_program_failure(self):
        with patch.object(self.episode, "capture_visual_feedback", side_effect=OSError("image storage unavailable")):
            reply, result = self.call("step(2)\nprint('partial')\n1 / 0")
        self.assertTrue(reply["isError"])
        self.assertEqual(result["error"]["type"], "ZeroDivisionError")
        self.assertEqual(result["visual_feedback"]["error"]["type"], "OSError")
        self.assertEqual(result["physics_step"], 2)
        self.assertEqual(result["stdout"], "partial\n")
        self.assertEqual(len(reply["content"]), 1)

    def test_capture_failure_after_deadline_terminates_instead_of_requesting_retry(self):
        def fail_after_deadline():
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            raise RuntimeError("camera failed late")
        with patch.object(self.episode, "capture_visual_feedback", side_effect=fail_after_deadline):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)")
        record = self.records()[-1]
        self.assertTrue(record["reply_withheld"])
        self.assertEqual(record["visual_feedback"]["error"]["message"], "camera failed late")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_capture_must_not_advance_physics(self):
        capture = self.episode.capture_visual_feedback
        def accidental_motion():
            result = capture()
            self.episode.step(1)
            return result
        with patch.object(self.episode, "capture_visual_feedback", side_effect=accidental_motion):
            reply, result = self.call("step(1)")
        self.assertTrue(reply["isError"])
        self.assertEqual(result["physics_step"], 2)
        self.assertEqual(result["visual_feedback"]["physics_step"], 2)
        self.assertIn("must not advance physics", result["visual_feedback"]["error"]["message"])
        self.assertEqual(len(reply["content"]), 1)

    def test_deadline_crossed_while_logging_after_capture_withholds_all_images(self):
        with patch.object(self.episode, "check_time", side_effect=[None, None, None, EpisodeStopped("wall_timeout")]):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)")
        self.assertEqual(self.episode.visual_captures, 1)
        self.assertEqual(self.records()[0]["image_count"], 3)
        self.assertEqual(self.records()[-1]["event"], "reply_withheld_after_logging")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_late_log_failure_cannot_override_wall_timeout_after_capture(self):
        def fail_after_deadline(*args, **kwargs):
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            raise OSError("disk failed late")
        with patch.object(Path, "open", side_effect=fail_after_deadline):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.call("step(1)")
        self.assertEqual(self.episode.visual_captures, 1)
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))

    def test_capture_timeout_remains_terminal_when_audit_fails(self):
        with patch.object(self.episode, "capture_visual_feedback", side_effect=EpisodeStopped("wall_timeout")):
            with patch.object(self.episode.runtime, "feedback", side_effect=OSError("audit failed")):
                with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                    self.call("step(1)")
        self.assertFalse(hasattr(self.episode, "_feedback_reports"))


if __name__ == "__main__":
    unittest.main()
