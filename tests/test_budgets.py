"""Shared episode-budget contracts; no simulator or model is launched."""
import copy
import json
from pathlib import Path
import unittest

from manipisa.evaluation.budgets import EpisodeBudget
from manipisa.evaluation.catalog import inventory


ROOT = Path(__file__).resolve().parents[1]


class EpisodeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/ur5_wuji_codex.json").read_text(encoding="utf-8"))

    def test_real_configuration_gives_task27_and_host_deadlines(self):
        budget = EpisodeBudget.from_config(self.config)
        self.assertEqual(budget.expert_budget_scale, 3.0)
        self.assertEqual(budget.wall_limit_s, 1200)
        self.assertEqual(budget.physics_steps(2380), 7140)
        self.assertEqual(budget.tool_timeout_s, 1260)
        self.assertEqual(budget.process_timeout_s, 1800)
        self.assertAlmostEqual(budget.physics_steps(2380) * budget.physics_dt, 119.0)
        exported = budget.to_dict()
        self.assertEqual(exported["wall_limit_s"], 1200)
        self.assertEqual(exported["tool_timeout_s"], 1260)
        self.assertEqual(exported["process_timeout_s"], 1800)

    def test_cli_wall_override_controls_all_derived_deadlines(self):
        original = copy.deepcopy(self.config)
        for wall, tool, process in ((300, 360, 900), (90.25, 151, 691)):
            with self.subTest(wall=wall):
                budget = EpisodeBudget.from_config(self.config, wall_limit_s=wall)
                self.assertEqual(budget.wall_limit_s, wall)
                self.assertEqual(budget.tool_timeout_s, tool)
                self.assertEqual(budget.process_timeout_s, process)
                self.assertEqual(budget.physics_steps(2380), 7140)
                self.assertEqual(budget.to_dict()["wall_limit_s"], wall)
        self.assertEqual(self.config, original)

    def test_scale_and_stride_round_up_to_whole_policy_intervals(self):
        cases = ((2380, 1.5, 3, 3570), (5, 1.5, 3, 9),
                 (7, 2, 4, 16), (7, 2, 1, 14), (1, 0.5, 3, 3))
        for expert, scale, stride, expected in cases:
            with self.subTest(expert=expert, scale=scale, stride=stride):
                config = dict(self.config, expert_budget_scale=scale, policy_stride=stride)
                budget = EpisodeBudget.from_config(config)
                self.assertEqual(budget.physics_steps(expert), expected)

    def test_nonpositive_nonfinite_and_boolean_inputs_are_rejected(self):
        invalid = (0, -1, float("nan"), float("inf"), -float("inf"), True, False, "1200")
        for value in invalid:
            for field in ("physics_dt", "expert_budget_scale", "wall_time_limit_s"):
                with self.subTest(field=field, value=value):
                    config = copy.deepcopy(self.config)
                    if field == "wall_time_limit_s":
                        config["development_run"][field] = value
                    else:
                        config[field] = value
                    with self.assertRaises(ValueError):
                        EpisodeBudget.from_config(config)
            with self.subTest(field="CLI wall_limit_s", value=value):
                with self.assertRaises(ValueError):
                    EpisodeBudget.from_config(self.config, wall_limit_s=value)
            with self.subTest(field="expert_time_step", value=value):
                with self.assertRaises(ValueError):
                    EpisodeBudget.from_config(self.config).physics_steps(value)

    def test_stride_requires_a_positive_integer(self):
        for stride in (0, -1, True, False, 1.5, 3.0, float("nan"), float("inf"), "3"):
            with self.subTest(stride=stride):
                with self.assertRaises(ValueError):
                    EpisodeBudget.from_config(dict(self.config, policy_stride=stride))

    def test_inventory_with_explicit_budget_fields_agrees_for_every_task(self):
        budget = EpisodeBudget.from_config(self.config)
        data = inventory(ROOT / "Bench2Dex", physics_dt=budget.physics_dt,
                         budget_scale=budget.expert_budget_scale, stride=budget.policy_stride)
        self.assertEqual(data["task_count"], 26)
        for task in data["tasks"]:
            with self.subTest(task=task["task"]):
                expected = budget.physics_steps(task["expert_physics_steps"])
                self.assertEqual(task["physics_steps"], expected)
                self.assertEqual(task["physics_dt"], budget.physics_dt)
                self.assertAlmostEqual(task["simulation_budget_s"], expected * budget.physics_dt)
        ball = next(task for task in data["tasks"] if task["task"] == "27_ball_box_loading")
        self.assertEqual(ball["physics_steps"], 7140)
        self.assertAlmostEqual(ball["simulation_budget_s"], 119.0)


if __name__ == "__main__":
    unittest.main()
