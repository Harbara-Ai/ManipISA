"""Public-state echo tests; these neither simulate physics nor invoke a model."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from manipisa.evaluation.programs import EpisodeStopped
from manipisa_robodojo.adapter import CAMERAS
from manipisa_robodojo.episode import RoboDojoEpisode, FEEDBACK_FORMAT
from manipisa_robodojo.model import Policy


def observation():
    state = {}
    for side, x in (("left", -.3), ("right", .3)):
        state[f"{side}_arm_joint_state"] = np.zeros(6)
        state[f"{side}_ee_pose"] = np.array([x, 0., 1., 1., 0., 0., 0.])
        state[f"{side}_ee_joint_state"] = np.array([1.])
    return {"data_format_version": "v1.0", "env_idx": 0,
            "additional_info": {"frequency": 25}, "instruction": "Move the block.", "state": state,
            "vision": {camera: {"color": np.full((3, 4, 3), [210, 50, 10], dtype=np.uint8)}
                       for camera in CAMERAS}}


class Plant:
    def __init__(self):
        self.obs, self.actions = observation(), []

    def exchange(self, action):
        self.actions.append(deepcopy(action))
        for side in ("left", "right"):
            for suffix in ("ee_pose", "ee_joint_state"):
                key = f"{side}_{suffix}"
                self.obs["state"][key] = action[key].copy()
        return deepcopy(self.obs)


def payload(reply):
    return json.loads(reply["content"][0]["text"])


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.plant = Plant()
        self.episode = RoboDojoEpisode(self.plant.obs, self.plant.exchange,
                                      Path(self.temp.name) / "episode", wall_limit=60)
        self.episode.warmup()
        self.episode.begin()

    def execute(self, code):
        return self.episode.tool("execute_python", {"code": code})

    def records(self):
        return [json.loads(line) for line in (self.episode.out / "tool-feedback.jsonl").read_text().splitlines()]

    def test_motion_returns_exact_three_current_rgb_and_no_extra_action(self):
        self.plant.obs["vision"]["cam_left_wrist"]["color"][:] = [7, 8, 9]
        with patch.object(self.episode.runtime, "step", wraps=self.episode.runtime.step) as step:
            reply = self.execute("set_gripper('left', 0.2)\nstep(2)")
        self.assertEqual(step.call_args_list[0].kwargs, {"return_snapshot": False})
        self.assertEqual(len(self.plant.actions), 3)  # warmup + exactly two intervals
        body = payload(reply)
        self.assertEqual(body["feedback_format"], FEEDBACK_FORMAT)
        self.assertEqual(body["visual_feedback"]["camera_ids"], list(CAMERAS))
        images = [block for block in reply["content"] if block["type"] == "image"]
        self.assertEqual(len(images), 3)
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[1]["data"]))))
        np.testing.assert_array_equal(pixels[0, 0], [7, 8, 9])
        self.assertEqual(body["grippers"]["left"]["reported_control_target"], .2)
        self.assertFalse(body["grippers"]["left"]["measured_position_available"])
        self.assertEqual(body["evidence_availability"]["contact_measurement"], "unavailable")
        self.assertEqual(self.episode.visual_feedback_count, 1)
        self.assertEqual(len(list((self.episode.out / "observations").glob("*.png"))), 3)
        self.assertEqual(self.records()[0]["image_count"], 3)

    def test_terminal_dedup_preserves_explicit_queries_handles_and_active_reports(self):
        first = payload(self.execute("report = runtime.submit(Instruction(Opcode.MOVE, ('arm_a',), "
            "PoseGoal('tool_a', (.3, 0., 1.), (1., 0., 0., 0.)), dwell_s=0.))\nstep()"))
        self.assertEqual(first["runtime"]["instructions"][0]["status"], "SUCCEEDED")
        second = payload(self.execute("print(runtime.query(report.call_id).status)"))
        self.assertEqual(second["runtime"]["instructions"], [])
        self.assertEqual(second["runtime"]["omitted_unchanged_terminal_count"], 1)
        self.assertEqual(len(second["runtime"]["control_handles"]), 1)
        self.assertIn("SUCCEEDED", second["stdout"])
        self.assertEqual(len(self.episode.runtime.feedback()["instructions"]), 1)
        self.assertEqual(len(self.records()[1]["runtime"]["instructions"]), 1)
        self.assertEqual(second["visual_feedback"]["status"], "unchanged")
        third = payload(self.execute("active = runtime.submit(Instruction(Opcode.MOVE, ('arm_b',), "
            "PoseGoal('tool_b', (-.4, 0., 1.), (1., 0., 0., 0.))))"))
        fourth = payload(self.execute("print('same active state')"))
        self.assertEqual(third["runtime"]["instructions"][0]["status"], "RUNNING")
        self.assertEqual(fourth["runtime"]["instructions"][0]["status"], "RUNNING")

    def test_partial_error_returns_real_executed_state_images_and_stdout(self):
        reply = self.execute("step(1)\nprint('after one action')\nraise ValueError('partial failure')")
        body = payload(reply)
        self.assertTrue(reply["isError"])
        self.assertEqual(body["error"], {"type": "ValueError", "message": "partial failure"})
        self.assertEqual(body["official_action_intervals"], 2)
        self.assertEqual(body["confirmed_observation_intervals"], 2)
        self.assertIn("legacy alias", body["interval_count_semantics"])
        self.assertEqual(body["stdout"], "after one action\n")
        self.assertEqual(sum(c["type"] == "image" for c in reply["content"]), 3)
        self.assertEqual(self.records()[0]["action_intervals_before"], 1)
        self.assertEqual(self.records()[0]["official_action_intervals"], 2)

    def test_long_stdout_preview_and_full_audit_are_distinct(self):
        body = payload(self.execute("print('HEAD' + 'x' * 45000 + 'TAIL')"))
        self.assertLessEqual(len(body["stdout"]), 4000)
        self.assertTrue(body["stdout"].startswith("HEAD"))
        self.assertTrue(body["stdout"].endswith("TAIL\n"))
        self.assertGreater(body["stdout_omitted_chars"], 0)
        self.assertEqual(len(self.records()[0]["stdout"]), body["stdout_total_chars"])

    def test_unavailable_grasp_and_hidden_fields_cannot_become_success_feedback(self):
        self.plant.obs.update(hidden_score=1, objects={"secret-object": [1, 2, 3]})
        self.plant.obs["state"]["contact_force"] = 55
        self.plant.obs["vision"]["cam_head"]["depth"] = "hidden-depth"
        body = payload(self.execute("report = runtime.submit(Instruction(Opcode.SHAPE_HAND, ('hand_a',), "
            "ShapeGoal((0.,))))\nstep()"))
        self.assertEqual(body["runtime"]["instructions"][0]["reason"], "UNSUPPORTED_CAPABILITY")
        text = json.dumps(body)
        self.assertNotIn("secret-object", text)
        self.assertNotIn("hidden-depth", text)
        self.assertNotIn("hidden_score", text)
        self.assertIn("link6", body["frame_semantics"]["tool_a"])
        self.assertIn("not the grasp center", body["frame_semantics"]["tool_a"])

    def test_numerical_facade_rejects_file_access(self):
        reply = self.execute("np.load('/some/private/file')")
        self.assertTrue(reply["isError"])
        self.assertEqual(len(self.plant.actions), 1)
        reply = self.execute("print(np.linalg.norm(np.array([3., 4.])))")
        self.assertNotIn("isError", reply)
        self.assertEqual(payload(reply)["stdout"], "5.0\n")

    def test_no_camera_images_before_motion_or_after_validation_error(self):
        reply = self.execute("raise ValueError('before motion')")
        self.assertTrue(reply["isError"])
        self.assertEqual(sum(c["type"] == "image" for c in reply["content"]), 0)
        reply = self.execute("import os")
        self.assertTrue(reply["isError"])
        self.assertEqual(self.episode.queries, 0)

    def test_deadline_after_feedback_withholds_reply_and_baseline(self):
        original = self.episode.runtime.feedback
        def expire():
            value = original()
            self.episode.stop("wall_timeout")
            return value
        with patch.object(self.episode.runtime, "feedback", side_effect=expire):
            with self.assertRaises(EpisodeStopped):
                self.execute("report = runtime.submit(Instruction(Opcode.SHAPE_HAND, ('hand_a',), ShapeGoal((0.,))))")
        self.assertEqual(self.episode._feedback_reports, {})
        self.assertTrue(self.records()[0]["reply_withheld"])

    def test_deadline_during_logging_adds_withheld_correction(self):
        original = self.episode._append_log
        def expire(filename, record):
            original(filename, record)
            if filename == "tool-feedback.jsonl":
                self.episode.stop("wall_timeout")
        with patch.object(self.episode, "_append_log", side_effect=expire):
            with self.assertRaises(EpisodeStopped):
                self.execute("print('done')")
        self.assertEqual(self.records()[-1]["event"], "reply_withheld_after_logging")
        self.assertTrue(self.records()[-1]["reply_withheld"])

    def test_deadline_during_explicit_observation_cannot_yield_images(self):
        original = self.episode._capture_visual
        def expire(source):
            result = original(source)
            self.episode.stop("wall_timeout")
            return result
        with patch.object(self.episode, "_capture_visual", side_effect=expire):
            with self.assertRaises(EpisodeStopped):
                self.episode.tool("observe", {})
        self.assertTrue(self.records()[-1]["reply_withheld"])

    def test_camera_failure_after_motion_does_not_return_partial_images(self):
        original_save = Image.Image.save
        calls = []
        def fail_second(image, *args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise OSError("camera encoding failed")
            return original_save(image, *args, **kwargs)
        with patch.object(Image.Image, "save", fail_second):
            reply = self.execute("step()")
        self.assertTrue(reply["isError"])
        self.assertEqual(payload(reply)["visual_feedback"]["status"], "error")
        self.assertEqual(sum(c["type"] == "image" for c in reply["content"]), 0)
        self.assertEqual(self.episode.queries, 0)

    def test_read_api_and_unknown_tools_share_final_deadline_check(self):
        for name, arguments in (("read_api", {"name": "index"}), ("unknown", {})):
            with self.subTest(tool=name):
                original = self.episode._append_log
                def expire(filename, record):
                    original(filename, record)
                    self.episode.stop("wall_timeout")
                with patch.object(self.episode, "_append_log", side_effect=expire):
                    with self.assertRaises(EpisodeStopped):
                        self.episode.tool(name, arguments)
                self.episode.reason = self.episode.terminal_time = None


class PolicyEpisodeTests(unittest.TestCase):
    def test_termination_on_first_sent_action_is_unconfirmed(self):
        called = []
        with tempfile.TemporaryDirectory() as directory:
            model = Policy({"action_type": "ee", "env_cfg_type": "arx_x5",
                            "artifact_dir": directory, "action_timeout_s": 2},
                           runner=lambda episode: called.append(True))
            try:
                model.update_obs(observation())
                model.get_action()  # Client may execute and end, never updating obs.
                model.reset()
                summary = json.loads(next(Path(directory).glob("*/adapter-result.json")).read_text())
                self.assertEqual(summary["sent_actions"], 1)
                self.assertEqual(summary["confirmed_observation_intervals"], 0)
                self.assertEqual(summary["received_action_observations"], 0)
                self.assertEqual(summary["sent_actions_without_followup_observation"], 1)
                self.assertEqual(summary["last_sent_action_execution"], "unknown_without_followup_observation")
                self.assertTrue(summary["summary_finalized_at_reset"])
                self.assertEqual(called, [])
                self.assertEqual(model.sent_actions, 0)
                self.assertEqual(model.received_action_observations, 0)
            finally:
                model.reset()

    def test_final_executed_action_without_next_observation_is_not_lost(self):
        def runner(episode):
            episode.begin()
            episode.step(2)
        with tempfile.TemporaryDirectory() as directory:
            model = Policy({"action_type": "ee", "env_cfg_type": "arx_x5",
                            "artifact_dir": directory, "action_timeout_s": 2}, runner=runner)
            plant = Plant()
            try:
                model.update_obs(plant.obs)
                for _ in range(2):
                    model.update_obs(plant.exchange(model.get_action()[0]))
                plant.exchange(model.get_action()[0])  # Ends at client; no final update_obs.
                with self.assertRaises(RuntimeError):
                    model.get_action()  # Invalid duplicate requests do not count.
                model.reset()
                summary = json.loads(next(Path(directory).glob("*/adapter-result.json")).read_text())
                self.assertEqual(len(plant.actions), 3)
                self.assertEqual(summary["sent_actions"], 3)
                self.assertEqual(summary["confirmed_observation_intervals"], 2)
                self.assertEqual(summary["received_action_observations"], 2)
                self.assertEqual(summary["sent_actions_without_followup_observation"], 1)
                self.assertEqual(summary["last_sent_action_execution"], "unknown_without_followup_observation")
                self.assertNotIn("action_intervals", summary)
            finally:
                model.reset()

    def test_holds_after_agent_exit_are_counted_at_final_reset(self):
        def runner(episode):
            episode.begin()
        with tempfile.TemporaryDirectory() as directory:
            model = Policy({"action_type": "ee", "env_cfg_type": "arx_x5",
                            "artifact_dir": directory, "action_timeout_s": 2}, runner=runner)
            plant = Plant()
            try:
                model.update_obs(plant.obs)
                model.update_obs(plant.exchange(model.get_action()[0]))
                self.assertTrue(model._done.wait(2))
                model.update_obs(plant.exchange(model.get_action()[0]))  # Fallback hold acknowledged.
                model.get_action()  # Another hold, with unknown execution outcome.
                model.reset()
                summary = json.loads(next(Path(directory).glob("*/adapter-result.json")).read_text())
                self.assertEqual(summary["sent_actions"], 3)
                self.assertEqual(summary["confirmed_observation_intervals"], 1)
                self.assertEqual(summary["received_action_observations"], 2)
                self.assertEqual(summary["sent_actions_without_followup_observation"], 1)
                self.assertTrue(summary["summary_finalized_at_reset"])
            finally:
                model.reset()

    def test_two_episodes_use_new_host_and_clear_feedback_baseline(self):
        captured = []
        def runner(episode):
            episode.begin()
            captured.append(episode)
            episode.tool("execute_python", {"code": "set_gripper('left', .2)\nstep()"})
        with tempfile.TemporaryDirectory() as directory:
            model = Policy({"action_type": "ee", "env_cfg_type": "arx_x5",
                            "artifact_dir": directory, "action_timeout_s": 2}, runner=runner)
            try:
                for _ in range(2):
                    plant = Plant()
                    model.update_obs(plant.obs)
                    for _ in range(2):
                        model.update_obs(plant.exchange(model.get_action()[0]))
                    self.assertTrue(model._done.wait(2))
                    self.assertIsNone(model._error)
                    self.assertIsInstance(model._episode, RoboDojoEpisode)
                    self.assertEqual(model._episode.visual_feedback_count, 1)
                    model.on_trial_end({"official_score": 99})
                self.assertIsNot(captured[0], captured[1])
                results = [json.loads(p.read_text()) for p in Path(directory).glob("*/adapter-result.json")]
                self.assertEqual(len(results), 2)
                self.assertTrue(all(r["official_score"] is None for r in results))
            finally:
                model.reset()


if __name__ == "__main__":
    unittest.main()
