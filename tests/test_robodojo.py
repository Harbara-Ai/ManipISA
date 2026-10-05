"""Public-protocol/contract tests, not RoboDojo physics or leaderboard scores."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from manipisa import Runtime, Instruction, Opcode, PoseGoal, ShapeGoal, Status
from manipisa.adapters.robodojo import CAMERAS, RoboDojoAdapter, public_observation
from manipisa.evaluation.robodojo import RoboDojoModel
from manipisa.evaluation.native_agent import task_instructions


def observation():
    state = {}
    for side, x in (("left", -0.3), ("right", 0.3)):
        state[f"{side}_arm_joint_state"] = np.zeros(6)
        state[f"{side}_ee_pose"] = np.array([x, 0., 1., 1., 0., 0., 0.])
        state[f"{side}_ee_joint_state"] = np.array([1.])
    return {"data_format_version": "v1.0", "env_idx": 0,
            "additional_info": {"frequency": 25}, "instruction": "Move the block.", "state": state,
            "vision": {camera: {"color": np.full((3, 4, 3), [210, 50, 10], dtype=np.uint8)} for camera in CAMERAS}}


class Plant:
    """Deterministic pose echo to exercise protocol; does not model IK or physics."""
    def __init__(self):
        self.obs = observation()
        self.actions = []

    def exchange(self, action):
        self.actions.append(deepcopy(action))
        for side in ("left", "right"):
            self.obs["state"][f"{side}_ee_pose"] = action[f"{side}_ee_pose"].copy()
            self.obs["state"][f"{side}_ee_joint_state"] = action[f"{side}_ee_joint_state"].copy()
        return deepcopy(self.obs)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.plant = Plant()
        self.adapter = RoboDojoAdapter(self.plant.obs, self.plant.exchange)

    def warm_runtime(self):
        self.adapter.step()
        return Runtime(self.adapter)

    def test_public_boundary_and_rgb_order(self):
        obs = observation()
        obs.update(success=True, objects={"secret": [1, 2, 3]}, action={"future": 123})
        obs["state"]["task_score"] = 99
        obs["vision"]["oracle_camera"] = {"color": np.zeros((3, 4, 3), dtype=np.uint8)}
        clean = public_observation(obs)
        self.assertEqual(set(clean["vision"]), set(CAMERAS))
        self.assertNotIn("objects", clean)
        self.assertNotIn("action", clean)
        self.assertNotIn("task_score", clean["state"])
        np.testing.assert_array_equal(clean["vision"]["cam_head"]["color"][0, 0], [210, 50, 10])
        obs["vision"]["cam_head"]["color"][:] = 0
        self.assertEqual(clean["vision"]["cam_head"]["color"][0, 0, 0], 210)

    def test_bad_wire_values_fail(self):
        for field, value in (("left_ee_pose", [0] * 7), ("right_arm_joint_state", [float("nan")] * 6),
                             ("left_ee_joint_state", [1.1])):
            obs = observation()
            obs["state"][field] = value
            with self.assertRaises(ValueError):
                public_observation(obs)
        obs = observation()
        obs["vision"]["cam_head"]["color"] = b"jpeg"
        with self.assertRaises(ValueError):
            public_observation(obs)

    def test_first_frame_does_not_invent_velocity(self):
        runtime = Runtime(self.adapter)
        command = Instruction(Opcode.MOVE, ("arm_a",), PoseGoal("tool_a", (.3, 0., 1.), (1., 0., 0., 0.)))
        self.assertEqual(runtime.submit(command).reason, "EVIDENCE_UNAVAILABLE")
        self.adapter.step()
        self.assertTrue(self.adapter.observe().joints["arm_a"].valid)
        self.assertAlmostEqual(self.adapter.observe().timestamp, .04)

    def test_move_maps_right_a_and_only_completes_from_feedback(self):
        runtime = self.warm_runtime()
        command = Instruction(Opcode.MOVE, ("arm_a",), PoseGoal("tool_a", (.32, 0., 1.), (1., 0., 0., 0.)), dwell_s=.08)
        report = runtime.submit(command)
        self.assertEqual(report.status, Status.RUNNING)
        self.assertEqual(len(self.plant.actions), 1)
        runtime.step()
        self.assertEqual(runtime.query(report.call_id).status, Status.RUNNING)
        self.assertAlmostEqual(self.plant.actions[-1]["right_ee_pose"][0], .304)
        self.assertEqual(self.plant.actions[-1]["left_ee_pose"][0], -.3)
        for _ in range(15):
            runtime.step()
        completed = runtime.query(report.call_id)
        self.assertEqual(completed.status, Status.SUCCEEDED)
        self.assertIsNotNone(completed.control_handle)
        self.assertEqual(runtime.feedback()["state"]["state_source"], "robodojo_public_observation")
        self.assertEqual(runtime.submit(command).reason, "RESOURCE_CONFLICT")
        self.assertEqual(runtime.submit(command, replace_handle=completed.control_handle).status, Status.RUNNING)

    def test_stuck_robot_never_succeeds(self):
        adapter = RoboDojoAdapter(observation(), lambda action: observation())
        adapter.step()
        runtime = Runtime(adapter)
        report = runtime.submit(Instruction(Opcode.MOVE, ("arm_a",), PoseGoal("tool_a", (.5, 0., 1.), (1., 0., 0., 0.)), timeout_s=.12))
        for _ in range(4):
            runtime.step()
        self.assertEqual(runtime.query(report.call_id).reason, "TIMEOUT")

    def test_gripper_is_a_command_not_shape_evidence(self):
        runtime = self.warm_runtime()
        report = runtime.submit(Instruction(Opcode.SHAPE_HAND, ("hand_a",), ShapeGoal((0.,))))
        self.assertEqual(report.reason, "UNSUPPORTED_CAPABILITY")
        result = self.adapter.set_gripper("right", .25)
        self.assertFalse(result["physical_success_verified"])
        runtime.step()
        self.assertEqual(self.plant.actions[-1]["right_ee_joint_state"][0], .25)
        self.assertNotIn("hand_a", self.adapter.observe().joints)
        self.assertEqual(self.adapter.observe().contacts, {})

    def test_rotation_wxyz_and_quaternion_sign(self):
        self.adapter.step()
        self.plant.obs["state"]["right_ee_pose"][3:] *= -1
        # Changing quaternion sign describes the same physical rotation.
        self.adapter.exchange = lambda action: deepcopy(self.plant.obs)
        self.adapter.step()
        np.testing.assert_allclose(self.adapter.observe().frames["tool_a"].angular_velocity, [0., 0., 0.])

    def test_clock_change_is_rejected(self):
        self.adapter.exchange = lambda action: {**observation(), "additional_info": {"frequency": 30}}
        with self.assertRaisesRegex(ValueError, "frequency changed"):
            self.adapter.step()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = {"env_cfg_type": "arx_x5", "action_type": "ee", "eval_batch": False,
                    "artifact_dir": self.tmp.name, "agent_wall_limit_s": 3., "action_timeout_s": 3.}
        self.models = []

    def tearDown(self):
        for model in self.models:
            model.reset()
        self.tmp.cleanup()

    def model(self, runner):
        model = RoboDojoModel(self.cfg, runner=runner)
        self.models.append(model)
        return model

    def test_full_handshake_tools_and_episode_cleanup(self):
        captured = []
        def runner(episode):
            episode.begin()
            captured.append(task_instructions(episode))
            captured.append(episode.first_observation)
            reply = episode.tool("execute_python", {"code": "set_gripper('left', 0.2)\nstep(2)\nprint(state()['official_action_intervals'])"})
            captured.append(reply)
        model = self.model(runner)
        plant = Plant()
        model.prepare_case({"hidden_score": 1})
        model.update_obs(plant.obs)
        for _ in range(3):
            chunk = model.get_action()
            self.assertEqual(len(chunk), 1)
            self.assertEqual(set(chunk[0]), {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"})
            model.update_obs(plant.exchange(chunk[0]))
        self.assertTrue(model._done.wait(timeout=2))
        self.assertIsNone(model._error)
        self.assertIn("RoboDojo", captured[0])
        self.assertNotIn("hidden_score", str(captured))
        self.assertEqual(sum(c["type"] == "image" for c in captured[1]), 3)
        self.assertEqual(plant.actions[-1]["left_ee_joint_state"][0], .2)
        self.assertEqual(model._episode.steps, 3)
        # Agent completion is not a simulator success/termination signal.
        self.assertEqual(len(model.get_action()), 1)
        model.on_trial_end({"success": True})
        self.assertIsNone(model._thread)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*/adapter-result.json"))), 1)

    def test_duplicate_rpc_calls_cannot_advance_clock(self):
        model = self.model(lambda episode: None)
        with self.assertRaises(RuntimeError):
            model.get_action()
        model.update_obs(observation())
        with self.assertRaises(RuntimeError):
            model.update_obs(observation())
        model.get_action()
        with self.assertRaises(RuntimeError):
            model.get_action()

    def test_reset_interrupts_waiting_worker_and_starts_new_episode(self):
        def runner(episode):
            episode.begin()
            episode.step(100)
        model = self.model(runner)
        model.update_obs(observation())
        model.get_action()
        old_thread = model._thread
        model.reset()
        self.assertFalse(old_thread.is_alive())
        model.update_obs(observation())
        self.assertEqual(len(model.get_action()), 1)

    def test_worker_failure_is_visible(self):
        def runner(episode):
            raise ValueError("policy failed")
        model = self.model(runner)
        model.update_obs(observation())
        model.get_action()
        model.update_obs(observation())
        self.assertTrue(model._done.wait(timeout=2))
        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            model.get_action()

    def test_configuration_rejects_joint_and_batch_modes(self):
        for change in ({"action_type": "joint"}, {"eval_batch": True}, {"env_cfg_type": "ur5"}):
            with self.assertRaises(ValueError):
                RoboDojoModel({**self.cfg, **change})


if __name__ == "__main__":
    unittest.main()
