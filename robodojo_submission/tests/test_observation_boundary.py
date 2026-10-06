"""Public-boundary regression checks; no simulator or official scores involved."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manipisa import Instruction, Opcode, PoseGoal, Runtime, Status
from manipisa_robodojo.adapter import CAMERAS, RoboDojoAdapter, public_observation


def observation():
    state = {}
    for side, x in (("left", -.3), ("right", .3)):
        state[f"{side}_arm_joint_state"] = np.zeros(6)
        state[f"{side}_ee_pose"] = np.array([x, 0., 1., 1., 0., 0., 0.])
        state[f"{side}_ee_joint_state"] = np.array([1.])
    return {"data_format_version": "v1.0", "env_idx": 0,
            "additional_info": {"frequency": 25}, "instruction": "Move the block.",
            "state": state, "vision": {
                name: {"color": np.full((3, 4, 3), [210, 50, 10], dtype=np.uint8)}
                for name in CAMERAS}}


class Poison:
    def fail(self, *args, **kwargs):
        raise AssertionError("A hidden object was accessed")

    __array__ = __float__ = __deepcopy__ = __repr__ = __str__ = fail


class ObservationBoundaryTests(unittest.TestCase):
    def test_unknown_fields_are_not_traversed(self):
        obs = observation()
        poison = Poison()
        obs.update(friction=poison, contact=poison, object_gt=poison, evaluator_labels=poison)
        obs["additional_info"].update(friction_coefficient=poison, reward=poison)
        obs["state"].update(contact=poison, object_gt=poison, success=poison)
        obs["vision"]["hidden_camera"] = poison
        for camera in CAMERAS:
            obs["vision"][camera].update(depth=poison, intrinsics=poison, contact=poison)
        clean = public_observation(obs)
        self.assertEqual(set(clean), {"data_format_version", "env_idx", "additional_info",
                                     "instruction", "state", "vision"})
        self.assertEqual(clean["additional_info"], {"frequency": 25.})
        self.assertEqual(set(clean["state"]), set(observation()["state"]))
        self.assertEqual(set(clean["vision"]), set(CAMERAS))
        for camera in CAMERAS:
            self.assertEqual(set(clean["vision"][camera]), {"color"})

    def test_array_storage_and_dtype_metadata_do_not_leak(self):
        obs = observation()
        for key, value in obs["state"].items():
            obs["state"][key] = value.view(np.dtype(value.dtype, metadata={"secret": Poison()}))
        for camera in CAMERAS:
            pixels = obs["vision"][camera]["color"][:, ::-1]
            obs["vision"][camera]["color"] = pixels.view(np.dtype("uint8", metadata={"secret": Poison()}))
        clean = public_observation(obs)
        pairs = [(obs["state"][key], value) for key, value in clean["state"].items()]
        pairs += [(obs["vision"][name]["color"], clean["vision"][name]["color"]) for name in CAMERAS]
        for source, result in pairs:
            self.assertFalse(np.shares_memory(source, result))
            self.assertIsNone(result.base)
            self.assertIsNone(result.dtype.metadata)
            before = result.copy()
            source[...] = 0
            np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(clean["vision"][CAMERAS[0]]["color"][0, 0], [210, 50, 10])

    def test_protocol_numbers_do_not_invoke_object_conversion(self):
        object_array = np.empty(6, dtype=object)
        for i in range(6):
            object_array[i] = Poison()
        invalid = [object_array, [Poison()] * 6,
                   ["0"] * 6, [True] * 6, np.ones(6, dtype=complex),
                   np.ones(6, dtype=bool), np.ma.array(np.zeros(6)),
                   [10 ** 1000] * 6, np.zeros((1, 6)), np.full(6, np.nan), np.full(6, np.inf)]
        for value in invalid:
            with self.subTest(kind=type(value).__name__):
                obs = observation()
                obs["state"]["left_arm_joint_state"] = value
                with self.assertRaises(ValueError):
                    public_observation(obs)

    def test_scalar_and_required_field_validation(self):
        class ExtraString(str):
            hidden = Poison()

        class ExtraMapping(dict):
            __getitem__ = __contains__ = get = Poison.fail

        cases = [("env_idx", True), ("env_idx", 1), ("env_idx", np.int64(0)),
                 ("instruction", ExtraString("task")), ("instruction", " "),
                 ("state", {"hidden": Poison()}), ("state", ExtraMapping()),
                 ("data_format_version", "v2")]
        for key, value in cases:
            obs = observation()
            obs[key] = value
            with self.assertRaises(ValueError):
                public_observation(obs)
        for frequency in (True, "25", Poison(), 0., -1., float("inf"), float("nan"), 5e-324, 10 ** 1000):
            obs = observation()
            obs["additional_info"]["frequency"] = frequency
            with self.assertRaises(ValueError):
                public_observation(obs)
        for key in observation():
            obs = observation()
            del obs[key]
            with self.assertRaises(ValueError):
                public_observation(obs)

    def test_pose_gripper_and_camera_validation(self):
        for key, value in (("right_ee_pose", [0.] * 7),
                           ("left_ee_pose", [0., 0., 1., 2., 0., 0., 0.]),
                           ("left_ee_joint_state", [-.01]),
                           ("right_ee_joint_state", [1.01])):
            obs = observation()
            obs["state"][key] = value
            with self.assertRaises(ValueError):
                public_observation(obs)
        for image in (b"jpeg", np.zeros((3, 4, 4), dtype=np.uint8),
                      np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((3, 4, 3)),
                      np.ma.array(np.zeros((3, 4, 3), dtype=np.uint8))):
            obs = observation()
            obs["vision"][CAMERAS[0]]["color"] = image
            with self.assertRaises(ValueError):
                public_observation(obs)

    def test_plain_wire_lists_remain_supported(self):
        obs = observation()
        obs["state"] = {key: value.tolist() for key, value in obs["state"].items()}
        obs["additional_info"]["frequency"] = np.float32(25)
        clean = public_observation(obs)
        self.assertEqual(clean["state"]["right_ee_pose"].shape, (7,))

    def test_both_arms_closed_loop_preserves_link6_and_gripper_contracts(self):
        current = observation()
        actions = []

        def exchange(action):
            actions.append(deepcopy(action))
            for key, value in action.items():
                current["state"][key] = value.copy()
            current["state"]["object_gt"] = Poison()
            current["additional_info"]["friction"] = Poison()
            return current

        adapter = RoboDojoAdapter(current, exchange)
        adapter.step()
        runtime = Runtime(adapter)
        reports = []
        for actor, frame, target in (("arm_a", "tool_a", (.312, 0., 1.)),
                                     ("arm_b", "tool_b", (-.308, 0., 1.))):
            reports.append(runtime.submit(Instruction(Opcode.MOVE, (actor,),
                           PoseGoal(frame, target, (1., 0., 0., 0.)), dwell_s=.08)))
        adapter.set_gripper("left", .25)
        runtime.step()
        self.assertAlmostEqual(actions[-1]["right_ee_pose"][0], .304)
        self.assertAlmostEqual(actions[-1]["left_ee_pose"][0], -.304)
        np.testing.assert_array_equal(actions[-1]["right_ee_pose"][1:], [0., 1., 1., 0., 0., 0.])
        self.assertEqual(actions[-1]["left_ee_joint_state"].tolist(), [.25])
        for _ in range(12):
            runtime.step()
        self.assertTrue(all(runtime.query(report.call_id).status == Status.SUCCEEDED for report in reports))
        self.assertEqual(set(actions[-1]), {f"{side}_{suffix}" for side in ("left", "right")
                                           for suffix in ("ee_pose", "ee_joint_state")})
        self.assertEqual(adapter.observe().contacts, {})
        self.assertFalse(adapter.gripper_targets()["left"]["measured_position_available"])
        self.assertNotIn("object_gt", adapter.observation["state"])
        self.assertEqual(adapter.dt, .04)

    def test_exchange_revalidates_before_advancing_time(self):
        obs = observation()
        adapter = RoboDojoAdapter(obs, lambda action: obs)
        obs["state"]["right_ee_pose"][3:] = 0.
        with self.assertRaises(ValueError):
            adapter.step()
        self.assertEqual(adapter.steps, 0)
        self.assertEqual(adapter.observation["state"]["right_ee_pose"][3], 1.)

    def test_nontrivial_wxyz_rotation_preserves_other_arm(self):
        current = observation()

        def exchange(action):
            for key, value in action.items():
                current["state"][key] = value.copy()
            return current

        adapter = RoboDojoAdapter(current, exchange)
        adapter.step()
        runtime = Runtime(adapter)
        runtime.submit(Instruction(Opcode.MOVE, ("arm_b",),
                       PoseGoal("tool_b", (-.3, 0., 1.), (np.sqrt(.5), 0., 0., np.sqrt(.5)))))
        runtime.step()
        # 0.5 rad/s * 0.04 s = 0.02 rad around world z: q=(cos(.01),0,0,sin(.01)).
        np.testing.assert_allclose(adapter.observation["state"]["left_ee_pose"][3:],
                                   [np.cos(.01), 0., 0., np.sin(.01)], atol=1e-12)
        np.testing.assert_array_equal(adapter.observation["state"]["right_ee_pose"][3:], [1., 0., 0., 0.])
        np.testing.assert_allclose(adapter.observe().frames["tool_b"].angular_velocity, [0., 0., .5])


if __name__ == "__main__":
    unittest.main()
