import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest

from manipisa.evaluation.catalog import inventory, release_tasks
from manipisa.evaluation.programs import EpisodeStopped, execute_program, public_builtins
from manipisa.evaluation.scoring import ScoreConfig, score_episode
from manipisa.evaluation.telemetry import codex_costs

ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = ScoreConfig(600, 200000, 30)
        self.official = {"evaluation_valid": True, "stable_success": True,
                         "latched_stage_completion_rate": .8, "safety_hard_violation": False}
        self.cost = {"wall_time_s": 120, "native_total_tokens": 80000, "model_request_count": 15,
                     "tokens_complete": True, "requests_complete": True, "settled": True}

    def test_document_examples(self):
        for success, violation, expected in ((True, False, 87.6), (True, True, 30),
                                            (False, False, 11.52), (False, True, 6)):
            with self.subTest(success=success, violation=violation):
                self.official.update(stable_success=success, safety_hard_violation=violation)
                self.assertAlmostEqual(score_episode(self.official, self.cost, self.config)["Overall"], expected)

    def test_missing_data_is_not_free_even_when_gate_is_zero(self):
        self.official.update(stable_success=False, latched_stage_completion_rate=0)
        self.cost["native_total_tokens"] = None
        self.assertIsNone(score_episode(self.official, self.cost, self.config)["Overall"])

    def test_cost_saturation_and_true_zero(self):
        self.cost.update(wall_time_s=1200, native_total_tokens=500000, model_request_count=50)
        self.assertEqual(score_episode(self.official, self.cost, self.config)["Overall"], 60)
        self.cost.update(wall_time_s=0, native_total_tokens=0, model_request_count=0)
        self.assertEqual(score_episode(self.official, self.cost, self.config)["Overall"], 100)

    def test_invalid_cost_and_evaluator_do_not_produce_score(self):
        for field, value in (("wall_time_s", -1), ("wall_time_s", float("nan")),
                             ("native_total_tokens", True), ("model_request_count", 1.5),
                             ("settled", False), ("requests_complete", False)):
            cost = dict(self.cost, **{field: value})
            self.assertIsNone(score_episode(self.official, cost, self.config)["Overall"])
        self.official["evaluation_valid"] = False
        result = score_episode(self.official, self.cost, self.config)
        self.assertIsNone(result["A"])
        self.assertIsNone(result["Overall"])

    def test_unset_reference_cannot_rank(self):
        self.assertIsNone(score_episode(self.official, self.cost, ScoreConfig())["Overall"])
        for bad in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                ScoreConfig(wall_time_ref_s=bad)


class CatalogTests(unittest.TestCase):
    def test_exact_release_catalog_and_task_budget(self):
        data = inventory(ROOT / "Bench2Dex")
        self.assertEqual(data["task_count"], 26)
        self.assertEqual(data["excluded_extra_yaml"], ["86_short_jigsaw_puzzle.yaml"])
        ball = next(t for t in data["tasks"] if t["task"].startswith("27_"))
        self.assertEqual(ball["physics_steps"], 3570)
        self.assertAlmostEqual(ball["simulation_budget_s"], 59.5)
        self.assertTrue(all(t["robot_key"] == "multi_ur5_wuji_with_flange" for t in data["tasks"]))


class ProgramTests(unittest.TestCase):
    def test_persistent_computation_and_private_api_rejection(self):
        scope = {"__builtins__": public_builtins()}
        execute_program("x = sum(range(5))", scope, time.perf_counter() + 2)
        self.assertEqual(execute_program("print(x)", scope, time.perf_counter() + 2), "10\n")
        for code in ("import os", "x.__class__", "_hidden = 1"):
            with self.assertRaises(ValueError):
                execute_program(code, scope, time.perf_counter() + 2)

    def test_infinite_python_loop_reaches_wall_deadline(self):
        with self.assertRaises(EpisodeStopped):
            execute_program("while True:\n    pass", {"__builtins__": public_builtins()}, time.perf_counter() + .02)


class TelemetryTests(unittest.TestCase):
    def fixture(self):
        common = {"conversation.id": "episode-1", "model": "gpt-6.1-sol", "app.version": "0.160.0"}
        def event(name, second, **extra):
            return {**common, "event.name": name, "event.timestamp": f"2026-10-05T01:00:{second:02d}.000Z", **extra}
        return [event("codex.websocket_request", 0, success="true"),
                event("codex.sse_event", 1, **{"event.kind": "response.completed", "input_token_count": "9000", "output_token_count": "0"}),
                event("codex.websocket_request", 3, success="true"),
                event("codex.sse_event", 5, **{"event.kind": "response.completed", "input_token_count": "100", "output_token_count": "20", "cached_token_count": "60", "reasoning_token_count": "10"})]

    def measure(self, records):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            path.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 9100, "output_tokens": 20}}))
            return codex_costs(path, records, wall_time_s=10, process_completed=False,
                task_start_utc="2026-10-05T01:00:02+00:00", task_end_utc="2026-10-05T01:00:10+00:00")

    def test_prewarm_excluded_subsets_not_added(self):
        result = self.measure(self.fixture())
        self.assertEqual(result["native_total_tokens"], 120)
        self.assertEqual(result["model_request_count"], 1)
        self.assertTrue(result["settled"])

    def test_incomplete_failed_duplicate_and_http_are_not_guessed(self):
        fixture = self.fixture()
        variants = [fixture[:-1], fixture + [fixture[-1]],
                    fixture + [{"event.name": "codex.api_request", "endpoint": "/responses"}]]
        failed = copy.deepcopy(fixture)
        failed[2]["success"] = "false"
        variants.append(failed)
        for records in variants:
            result = self.measure(records)
            self.assertFalse(result["requests_complete"])
            self.assertIsNone(result["model_request_count"])
            self.assertFalse(result["tokens_complete"])

    def test_auxiliary_model_before_task_still_invalidates_complete_accounting(self):
        records = self.fixture() + [{"model": "codex-auto-review", "event.name": "codex.conversation_starts",
                                     "event.timestamp": "2026-10-05T01:00:00.000Z"}]
        result = self.measure(records)
        self.assertFalse(result["settled"])
        self.assertIsNone(result["native_total_tokens"])
        self.assertEqual(result["auxiliary_models"], ["codex-auto-review"])

    def test_setup_completion_after_task_delivery_is_excluded(self):
        records = self.fixture()
        records[1]["event.timestamp"] = "2026-10-05T01:00:02.020Z"
        result = self.measure(records)
        self.assertEqual(result["native_total_tokens"], 120)
        self.assertEqual(result["model_request_count"], 1)


class SummaryTests(unittest.TestCase):
    def test_unrun_episode_remains_in_coverage_and_blocks_complete_mean(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("benchmark_summary", ROOT / "scripts/summarize_benchmark.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            entries = [{"episode_id": "a", "task": "27", "method": "direct", "status": "completed", "report": "a.json"},
                       {"episode_id": "b", "task": "21", "method": "direct", "status": "planned"}]
            (run / "plan.json").write_text(json.dumps({"scope": "development", "episodes": entries}))
            (run / "a.json").write_text(json.dumps({"official": {"evaluation_valid": True, "stable_success": True,
                "safety_hard_violation": False, "latched_stage_completion_rate": 1}, "score": {"Overall": 80}}))
            result = module.summarize(run)["methods"]["direct"]
            self.assertEqual(result["planned"], 2)
            self.assertEqual(result["Overall"]["coverage"], .5)
            self.assertIsNone(result["Overall"]["value"])
            self.assertEqual(result["Overall"]["available_only_mean"], 80)


if __name__ == "__main__":
    unittest.main()
