import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pilot_suite', Path(__file__).resolve().parents[1] / 'scripts/run_benchmark_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class SuiteCleanupTests(unittest.TestCase):
    def test_scene_failure_without_model_allows_next_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(suite.wait_for_wsl_cleanup(Path(folder), timeout_s=0))

    def test_running_agent_must_finish_and_clean_before_next_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            metadata = directory / 'wsl-agent.json'
            metadata.write_text(json.dumps({'phase': 'running'}))
            def finish(_):
                metadata.write_text(json.dumps({'phase': 'finished', 'cleanup': {'complete': True}}))
            with patch.object(suite.time, 'sleep', side_effect=finish) as sleep:
                self.assertTrue(suite.wait_for_wsl_cleanup(directory, timeout_s=5))
                sleep.assert_called_once()

    def test_incomplete_or_missing_or_malformed_cleanup_blocks_next_episode(self):
        states = [None, 'invalid JSON', json.dumps({'phase': 'finished', 'cleanup': {'complete': False}})]
        for state in states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory / 'agent-launch.json').write_text('{}')
                if state is not None:
                    (directory / 'wsl-agent.json').write_text(state)
                self.assertFalse(suite.wait_for_wsl_cleanup(directory, timeout_s=0))


if __name__ == '__main__':
    unittest.main()
