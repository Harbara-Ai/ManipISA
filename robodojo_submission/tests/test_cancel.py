"""Cancellation must finish before the original adapter's 15-second reset limit."""
import subprocess
import sys
import time
import unittest

from manipisa_robodojo.agent import _stop
from manipisa_robodojo.model import Policy
from protocol_check import observation
import tempfile
import os
from pathlib import Path


class CancellationTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux signal lifecycle")
    def test_server_sigterm_closes_episode_before_reset(self):
        code = r'''
import asyncio, json, os, signal, sys, threading, types
from unittest.mock import patch
import run_wsl
events = []
class Episode:
    def stop(self, reason): events.append('episode_stop')
class Policy:
    def __init__(self, config):
        self._closed = threading.Event()
        self._episode = Episode()
    def reset(self):
        assert self._closed.is_set()
        events.append('reset')
class Server:
    def __init__(self, *args): pass
    async def start(self):
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
    async def stop(self): events.append('server_stop')
module = types.ModuleType('client_server.ws.model_server')
module.PolicyServer = Server
module.PolicyServerConfig = lambda **kwargs: kwargs
with patch.dict(sys.modules, {'client_server.ws.model_server': module}), patch('manipisa_robodojo.model.Policy', Policy):
    asyncio.run(run_wsl.serve({'model':'gpt-6-astra','reasoning_effort':'high','artifact_dir':'fixture'},19000))
assert events == ['episode_stop', 'server_stop', 'reset'], events
print('SIGTERM cleanup passed')
'''
        submission = Path(__file__).resolve().parents[1]
        completed = subprocess.run([sys.executable, "-B", "-c", code],
            env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(submission), str(submission.parent)))),
            capture_output=True, text=True, timeout=10, check=True)
        self.assertIn("SIGTERM cleanup passed", completed.stdout)

    def test_reset_interrupts_agent_waiting_for_next_observation(self):
        def runner(episode):
            episode.begin()
            episode.step(100)

        with tempfile.TemporaryDirectory() as directory:
            model = Policy({"action_type": "ee", "env_cfg_type": "arx_x5",
                            "artifact_dir": directory, "action_timeout_s": 60}, runner=runner)
            try:
                model.update_obs(observation())
                model.get_action()  # Warm-up observation interval.
                model.update_obs(observation())
                model.get_action()  # Agent now awaits the next observation.
                start = time.monotonic()
                model.reset()
                self.assertLess(time.monotonic() - start, 2)
                self.assertIsNone(model._thread)
            finally:
                model.reset()

    @unittest.skipUnless(sys.platform == "linux", "WSL process-group cleanup")
    def test_process_that_ignores_termination_is_killed_within_reset_budget(self):
        process = subprocess.Popen([sys.executable, "-B", "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            start = time.monotonic()
            _stop(process)
            self.assertLess(time.monotonic() - start, 5)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
