"""XPolicyLab policy interface for the existing ManipISA/Codex agent.

The worker runs the normal synchronous Runtime. Each step hands ONE action to
XPolicyLab and blocks until the next official observation; there is no shadow
simulator and reasoning never advances a synthetic physics clock.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from manipisa import Runtime
from manipisa.adapters.robodojo import (
    ACTORS, CAMERAS, ROBODOJO_REVISION, XPOLICYLAB_REVISION,
    RoboDojoAdapter, public_observation,
)
from manipisa.types import json_safe
from .programs import EpisodeStopped, execute_program, public_builtins


API_REFERENCE = """RoboDojo public-observation integration:
runtime.submit(Instruction(Opcode.MOVE, ('arm_a',), PoseGoal('tool_a', xyz, wxyz)))
arm_a/tool_a = right X5; arm_b/tool_b = left X5. Positions are meters in the
environment-origin frame; orientations are wxyz. The official environment
performs IK and joint interpolation. Use separate MOVE calls for the two arms.
runtime.query(call_id), runtime.cancel(call_id), runtime.update(call_id, instruction),
runtime.feedback(); terminal calls retain control handles. Use
runtime.submit(new_instruction, replace_handle=old_report.control_handle) for takeover.
step(n) executes n official action intervals (normally 25 Hz), updating both
runtime and RGB observations. state() and observe do not advance physics.
set_gripper('left' or 'right', value) queues the official normalized [0,1]
gripper target; step() applies it. This is a raw control command, NOT a
SHAPE_HAND/CONTROL_GRASP success. The upstream gripper state is a control
target. Actual gripper position, object ground truth, contact force, wrench,
depth, calibration and evaluator labels are NOT provided to the agent.
Only MOVE REACH/SUSTAIN contracts are supported. SHAPE_HAND, MAKE_CONTACT,
BREAK_CONTACT, CONTROL_GRASP and APPLY_WRENCH reject unavailable capabilities.
Arm joint and Cartesian velocities are finite differences at the observation
frequency, not instantaneous or high-rate safety guarantees. An initial hold
interval, counted by the official evaluator, obtains the second observation.
Cartesian command rates are bounded; joint-speed checks are retrospective.
The original three RGB images retain their original camera names and pixels.
"""


class RoboDojoEpisode:
    """Agent host whose only environment access is the public exchange callback."""
    def __init__(self, initial_obs, exchange, out, *, wall_limit=600.0):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.wall_limit = float(wall_limit)
        self.started = self.terminal_time = self.reason = None
        self.queries = self.tool_calls = 0
        self.adapter = RoboDojoAdapter(initial_obs, exchange)
        self.task = {"description": self.adapter.observation["instruction"]}
        self.runtime = None
        self.namespace = None

    def warmup(self):
        self.adapter.step()  # A real, evaluator-counted hold; no invented zero velocity.
        self.runtime = Runtime(self.adapter)
        import manipisa.types as types
        self.namespace = {"__builtins__": public_builtins(), "np": np, "math": math,
                          "step": self.step, "state": self.public_state,
                          "set_gripper": self.adapter.set_gripper,
                          "runtime": SimpleNamespace(**{name: getattr(self.runtime, name) for name in
                                     ("submit", "query", "cancel", "update", "feedback")})}
        for name in ("Instruction", "Opcode", "Mode", "PoseGoal", "ShapeGoal", "Status"):
            self.namespace[name] = getattr(types, name)

    @property
    def steps(self):
        return self.adapter.steps

    @property
    def first_observation(self):
        return self.observe_content()

    def begin(self):
        if self.started is not None:
            raise RuntimeError("Task already delivered")
        self.started = time.perf_counter()

    def stop(self, reason):
        if self.reason is None:
            self.reason, self.terminal_time = reason, time.perf_counter()

    def check_time(self):
        if self.reason is not None:
            raise EpisodeStopped(self.reason)
        if self.started is None:
            raise RuntimeError("Task has not been delivered")
        if time.perf_counter() - self.started >= self.wall_limit:
            self.stop("wall_timeout")
            raise EpisodeStopped(self.reason)

    def step(self, n=1):
        if type(n) is not int or n < 1:
            raise ValueError("step(n) requires a positive integer")
        for _ in range(n):
            self.check_time()
            self.runtime.step()

    def public_state(self):
        return {"state": self.adapter.observe().to_dict(), "official_action_intervals": self.steps,
                "observation_period_s": self.adapter.dt, "grippers": self.adapter.gripper_targets(),
                "cameras": list(CAMERAS), "capabilities": {"MOVE": True, "SHAPE_HAND": False,
                    "MAKE_CONTACT": False, "BREAK_CONTACT": False, "CONTROL_GRASP": False,
                    "APPLY_WRENCH": False, "raw_gripper_command": True},
                "velocity_source": "finite_difference_at_observation_frequency"}

    def observe_content(self):
        from PIL import Image
        content = [{"type": "text", "text": json.dumps(self.public_state())}]
        for name in CAMERAS:
            stream = io.BytesIO()
            Image.fromarray(self.adapter.observation["vision"][name]["color"]).save(stream, format="PNG")
            content.extend([{"type": "text", "text": f"RGB camera: {name}; action interval: {self.steps}"},
                            {"type": "image", "mimeType": "image/png",
                             "data": base64.b64encode(stream.getvalue()).decode("ascii")}])
        self.queries += 1
        return content

    def agent_instructions(self):
        return (f"Task: {self.task['description']}\nYou control RoboDojo's original dual ARX X5 robot "
                f"and grippers using ManipISA. Wall budget: {self.wall_limit} seconds.\n"
                "Solve from the supplied language, RGB and public robot observations. Physics advances only "
                "through step(n). Do not import modules, inspect private attributes, read files, use network, "
                "access evaluator state, reset, or teleport. Persistent Python tools provide np, math and "
                "the APIs below. Use observe for RGB and read_api('instructions') for dataclass definitions.\n"
                + API_REFERENCE)

    def tool(self, name, arguments):
        self.check_time()
        self.tool_calls += 1
        if name == "observe":
            return {"content": self.observe_content()}
        if name == "read_api":
            requested = arguments.get("name")
            if requested == "index":
                text = json.dumps(["robodojo", "instructions"])
            elif requested == "robodojo":
                text = API_REFERENCE
            elif requested == "instructions":
                text = (Path(__file__).parents[1] / "types.py").read_text(encoding="utf-8")
            else:
                raise ValueError("Unknown API document")
            return {"content": [{"type": "text", "text": text}]}
        if name != "execute_python":
            raise ValueError("Unknown tool")
        code = arguments["code"]
        with (self.out / "programs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"action_interval": self.steps, "code": code}) + "\n")
        output = execute_program(code, self.namespace, self.started + self.wall_limit)
        return {"content": [{"type": "text", "text": json.dumps(json_safe(
            {"stdout": output, "runtime": self.runtime.feedback(), "grippers": self.adapter.gripper_targets()}))}]}


class RoboDojoModel:
    """Duck-types XPolicyLab ModelTemplate; one live environment per server.

    runner is injectable for transport tests. Production uses the existing
    Codex CLI runner and its existing model/reasoning configuration.
    """
    def __init__(self, model_cfg, *, runner=None):
        if model_cfg.get("action_type") != "ee":
            raise ValueError("ManipISA RoboDojo requires action_type=ee")
        if model_cfg.get("env_cfg_type") != "arx_x5" or model_cfg.get("eval_batch", False):
            raise ValueError("Use env_cfg_type=arx_x5 and eval_batch=false")
        self.cfg = dict(model_cfg)
        self.wall_limit = float(model_cfg.get("agent_wall_limit_s", 600))
        self.action_timeout = float(model_cfg.get("action_timeout_s", 660))
        if not all(math.isfinite(v) and v > 0 for v in (self.wall_limit, self.action_timeout)):
            raise ValueError("Agent and action timeouts must be finite and positive")
        if runner is None:
            from .native_agent import run_native_agent
            runner = run_native_agent
        self.runner = runner
        self._thread = None
        self._episode = None
        self.reset()

    def reset(self):
        if self._thread is not None:
            self._closed.set()
            if self._episode is not None:
                self._episode.stop("environment_reset_or_end")
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise RuntimeError("Previous agent did not terminate; refusing to mix episodes")
        self._thread = self._episode = self._latest = self._error = None
        self._actions, self._observations = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._closed, self._done = threading.Event(), threading.Event()
        self._sent = False

    def prepare_case(self, case_meta=None):
        # Case metadata may contain evaluator internals. Do not forward it.
        pass

    def on_trial_end(self, result=None):
        # End/cleanup only; terminal labels are never agent inputs.
        self.reset()

    def update_obs(self, obs):
        clean = public_observation(obs)
        if self._latest is not None and not self._sent:
            raise RuntimeError("An observation requires a preceding get_action; duplicate observations cannot advance time")
        self._latest = clean
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, args=(clean,), daemon=True,
                                            name="manipisa-robodojo-agent")
            self._thread.start()
        elif not self._done.is_set():
            self._observations.put_nowait(clean)
        self._sent = False

    def _exchange(self, action):
        if self._closed.is_set():
            raise EpisodeStopped("environment_reset_or_end")
        self._actions.put_nowait(action)
        deadline = time.monotonic() + self.action_timeout
        while not self._closed.is_set():
            if self._episode is not None and self._episode.started is not None:
                self._episode.check_time()
            if time.monotonic() >= deadline:
                raise TimeoutError("No official observation after action")
            try:
                return self._observations.get(timeout=0.05)
            except queue.Empty:
                continue
        raise EpisodeStopped("environment_reset_or_end")

    def _run(self, initial):
        out = Path(self.cfg.get("artifact_dir", "artifacts/manipisa-robodojo")) / uuid4().hex
        try:
            self._episode = RoboDojoEpisode(initial, self._exchange, out, wall_limit=self.wall_limit)
            self._episode.warmup()
            self.runner(self._episode)
        except EpisodeStopped:
            pass
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if self._episode is not None:
                    self._episode.stop("agent_error" if self._error else "agent_stopped")
                    summary = {"reason": self._episode.reason, "error": self._error,
                               "action_intervals": self._episode.steps,
                               "observation": "robodojo_public_observation+RGB",
                               "supported_isa": ["MOVE"], "raw_gripper_command": True,
                               "robodojo_reference_revision": ROBODOJO_REVISION,
                               "xpolicylab_reference_revision": XPOLICYLAB_REVISION,
                               "official_score": None}
                    (out / "adapter-result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            finally:
                self._done.set()

    def get_action(self):
        if self._latest is None or self._sent:
            raise RuntimeError("get_action requires a new observation")
        deadline = time.monotonic() + self.action_timeout
        while True:
            if self._error:
                raise RuntimeError(self._error)
            try:
                action = self._actions.get(timeout=0.05)
                self._sent = True
                return [deepcopy(action)]  # Exactly one official interval per request.
            except queue.Empty:
                if self._done.is_set():
                    # An agent finishing is not task success. Keep the robot in
                    # place until the unchanged official evaluator terminates.
                    self._sent = True
                    return [RoboDojoAdapter.hold_action(self._latest)]
                if time.monotonic() >= deadline:
                    self._closed.set()
                    if self._episode is not None:
                        self._episode.stop("policy_action_timeout")
                    raise TimeoutError("Agent did not provide an action within action_timeout_s")

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Use eval_batch=false and one environment per server")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Use eval_batch=false and one environment per server")
