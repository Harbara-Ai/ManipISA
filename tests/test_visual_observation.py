"""Observation capture contracts using synthetic frames, without Isaac or a model."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from manipisa.evaluation.programs import EpisodeStopped
from manipisa.evaluation.simulation import SimulationEpisode


class VisualObservationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.trace = []
        self.camera_ids = ["cam_overhead", "cam_wrist_left", "cam_wrist_right"]
        self.frames = {}
        for index, name in enumerate(self.camera_ids):
            self.frames[name] = SimpleNamespace(
                rgb=np.full((3, 5, 3), 30 + index * 50, dtype=np.uint8),
                image_shape=(3, 5), camera_model="pinhole", intrinsic=np.eye(3),
                extrinsic_world_from_cam=np.eye(4), distortion_coefficients=None,
                fisheye_camera_matrix=None)
        self.episode = episode = SimulationEpisode.__new__(SimulationEpisode)
        episode.out = Path(folder.name)
        episode.steps, episode.queries, episode.visual_feedback_count = 17, 0, 0
        episode._observation_attempts = 0
        episode.dt = 1 / 60
        episode.started, episode.reason, episode.terminal_time = None, None, None
        episode.wall_limit = 60
        episode.rig = SimpleNamespace(
            camera_ids=tuple(self.camera_ids),
            sync_mounted_camera_poses=lambda: self.trace.append("sync"),
            capture=self.capture)
        episode.sim = SimpleNamespace(render=lambda: self.trace.append("render"), step=Mock())
        episode.robot, episode.objects, episode.ids = object(), {"ball": object()}, ["ball"]
        episode.read_robot = Mock(return_value={
            "qpos": np.array([0.1, 0.2]), "qvel": np.array([0.3, 0.4]),
            "joint_names": ["joint_a", "joint_b"], "history": ["old sample"]})
        episode.read_objects = Mock(return_value={"ball": {"pose_world": np.arange(7)}})
        episode.public_state = Mock(return_value={
            "physics_step": episode.steps, "robot": {"joint_names": ["joint_a", "joint_b"]},
            "bodies": {"wrist": [0, 0, 1]}, "object_geometry": {"ball": [1, 2, 3]},
            "contacts": {"example": {"valid": True}}, "history": ["full state sentinel"]})

    def capture(self, dt):
        self.assertEqual(dt, self.episode.dt)
        self.trace.append("capture")
        # The rig's public order must win over the frame mapping's insertion order.
        return {name: self.frames[name] for name in reversed(self.camera_ids) if name in self.frames}

    def assert_no_delivery(self):
        self.assertEqual(self.episode.queries, 0)
        self.assertEqual(self.episode.visual_feedback_count, 0)

    def test_compact_capture_is_synchronized_and_matches_saved_pngs(self):
        self.episode.public_state.side_effect = AssertionError("compact capture read full state")
        metadata, images = self.episode.capture_visual_feedback()
        state = metadata["state"]
        self.assertEqual(self.trace, ["sync", "render", "capture"])
        self.episode.sim.step.assert_not_called()
        self.assertEqual(self.episode.steps, 17)
        self.assertEqual(metadata["physics_step"], 17)
        self.assertEqual(metadata["camera_ids"], self.camera_ids)
        self.assertEqual(list(state["cameras"]), self.camera_ids)
        self.assertEqual(len(images), len(self.camera_ids))
        self.assertEqual(state["robot"], {"qpos": [0.1, 0.2], "qvel": [0.3, 0.4]})
        self.assertEqual(state["objects"]["ball"]["pose_world"], list(range(7)))
        for field in ("bodies", "object_geometry", "contacts", "history"):
            self.assertNotIn(field, state)
        for name, image in zip(self.camera_ids, images):
            camera = state["cameras"][name]
            self.assertEqual((image["type"], image["mimeType"]), ("image", "image/png"))
            encoded = base64.b64decode(image["data"], validate=True)
            self.assertEqual((self.episode.out / camera["image_path"]).read_bytes(), encoded)
            self.assertEqual(camera["sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(camera["physics_step"], 17)
            self.assertEqual(camera["image_shape"], [3, 5])
            with Image.open(io.BytesIO(encoded)) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), self.frames[name].rgb)
        saved = self.episode.out / "observations" / metadata["observation_id"] / "state.json"
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), state)
        self.assertEqual((self.episode.queries, self.episode.visual_feedback_count), (1, 1))

    def test_explicit_observe_keeps_full_state(self):
        content = self.episode.observe_content()
        self.episode.public_state.assert_called_once_with()
        self.episode.read_robot.assert_not_called()
        state = json.loads(content[0]["text"])
        self.assertEqual(state["history"], ["full state sentinel"])
        self.assertEqual(state["object_geometry"], {"ball": [1, 2, 3]})
        self.assertIn("contacts", state)
        self.assertEqual(len(content), 1 + len(self.camera_ids))
        self.assertEqual(self.episode.visual_feedback_count, 0)

    def test_missing_camera_fails_without_partial_delivery(self):
        del self.frames[self.camera_ids[-1]]
        with self.assertRaisesRegex(RuntimeError, "Incomplete camera set"):
            self.episode.capture_visual_feedback()
        self.assert_no_delivery()
        self.assertFalse((self.episode.out / "observations").exists())

    def test_invalid_last_rgb_fails_without_partial_delivery(self):
        invalid_frames = [np.zeros((3, 5), dtype=np.uint8),
                          np.zeros((3, 5, 2), dtype=np.uint8),
                          np.zeros((3, 5, 3), dtype=np.float32),
                          np.full((3, 5, 3), np.nan),
                          np.zeros((5, 3, 3), dtype=np.uint8)]
        for rgb in invalid_frames:
            with self.subTest(shape=rgb.shape, dtype=rgb.dtype):
                self.frames[self.camera_ids[-1]].rgb = rgb
                with self.assertRaisesRegex(RuntimeError, "RGB"):
                    self.episode.capture_visual_feedback()
                self.assert_no_delivery()
                self.assertFalse((self.episode.out / "observations").exists())

    def test_failed_disk_write_never_overwrites_previous_observation(self):
        first, _ = self.episode.capture_visual_feedback()
        first_folder = self.episode.out / "observations" / first["observation_id"]
        original = {path.name: path.read_bytes() for path in first_folder.iterdir()}
        write_bytes = Path.write_bytes

        def fail_second_camera(path, data):
            if path.name == self.camera_ids[1] + ".png":
                raise OSError("synthetic disk failure")
            return write_bytes(path, data)

        with patch.object(Path, "write_bytes", fail_second_camera):
            with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                self.episode.capture_visual_feedback()
        self.assertEqual((self.episode.queries, self.episode.visual_feedback_count), (1, 1))
        third, _ = self.episode.capture_visual_feedback()
        self.assertNotEqual(first["observation_id"], third["observation_id"])
        self.assertEqual({path.name: path.read_bytes() for path in first_folder.iterdir()}, original)
        self.assertEqual((self.episode.queries, self.episode.visual_feedback_count), (2, 2))

    def test_capture_crossing_deadline_does_not_deliver_images(self):
        self.episode.started = time.perf_counter()
        capture = self.episode.rig.capture

        def expire_during_capture(dt):
            frames = capture(dt)
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            return frames

        self.episode.rig.capture = expire_during_capture
        with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
            self.episode.capture_visual_feedback()
        self.assert_no_delivery()
        self.assertFalse((self.episode.out / "observations").exists())

    def test_disk_write_crossing_deadline_does_not_deliver_images(self):
        self.episode.started = time.perf_counter()
        write_text = Path.write_text

        def expire_during_write(path, *args, **kwargs):
            result = write_text(path, *args, **kwargs)
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            return result

        with patch.object(Path, "write_text", expire_during_write):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.episode.capture_visual_feedback()
        self.assert_no_delivery()

    def test_physics_step_change_is_rejected(self):
        capture = self.episode.rig.capture

        def changed_step(dt):
            frames = capture(dt)
            self.episode.steps += 1
            return frames

        self.episode.rig.capture = changed_step
        with self.assertRaisesRegex(RuntimeError, "Physics advanced"):
            self.episode.capture_visual_feedback()
        self.assert_no_delivery()
        self.assertFalse((self.episode.out / "observations").exists())

    def test_explicit_camera_failure_cannot_hide_wall_timeout(self):
        self.episode.started = time.perf_counter()
        self.episode.tool_calls = 0
        def late_camera_failure():
            self.episode.started = time.perf_counter() - self.episode.wall_limit - 1
            raise RuntimeError("camera failed after the deadline")
        with patch.object(self.episode, "observe_content", side_effect=late_camera_failure):
            with self.assertRaisesRegex(EpisodeStopped, "wall_timeout"):
                self.episode.tool("observe", {})
        self.assert_no_delivery()


if __name__ == "__main__":
    unittest.main()
