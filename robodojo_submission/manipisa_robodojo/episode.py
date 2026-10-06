"""RoboDojo public-observation episode and deadline-checked policy feedback.

This host owns no simulator, object state, reward function, or contact sensor.
Images are copied from the observation returned by the official action exchange.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

from manipisa import Runtime
import manipisa.types as types
from manipisa.evaluation.programs import EpisodeStopped
from manipisa.evaluation.robodojo import RoboDojoEpisode as BaseEpisode
from manipisa.evaluation.tool_feedback import stdout_preview
from manipisa.feedback import summarize_feedback
from manipisa.types import json_safe

from .adapter import CAMERAS, RoboDojoAdapter
from .programs import execute_program, public_builtins, restricted_math, restricted_np


FEEDBACK_FORMAT = "manipisa.robodojo.compact-visual.v1"
EVIDENCE_AVAILABILITY = {
    "contact_measurement": "unavailable",
    "wrench_measurement": "unavailable",
    "measured_gripper_position": "unavailable",
    "object_ground_truth": "unavailable",
    "official_reward_or_success": "unavailable",
}
FRAME_SEMANTICS = {
    "tool_a": "right ARX X5 link6 frame, not the grasp center",
    "tool_b": "left ARX X5 link6 frame, not the grasp center",
    "reference": "official environment-origin world frame; positions in meters; wxyz quaternions",
}
API_REFERENCE = """RoboDojo public-observation integration:
arm_a/tool_a = right X5; arm_b/tool_b = left X5. tool_a and tool_b denote the
official link6 frames, NOT the grasp centers. Do not apply a guessed grasp-center
offset. Positions are meters in the environment-origin world frame; orientation
is a wxyz quaternion. The official simulator owns IK and joint interpolation.
runtime.submit(Instruction(Opcode.MOVE, ('arm_a',), PoseGoal('tool_a', xyz, wxyz)))
runtime.query(call_id), runtime.cancel(call_id), runtime.update(call_id, instruction),
runtime.feedback(); terminal calls retain control handles. Hand off with
runtime.submit(new_instruction, replace_handle=old_report.control_handle).
Use separate MOVE instructions for the two arms. MOVE supports REACH/SUSTAIN.
step(n) requests n official action intervals, normally at 25 Hz, waiting for a
follow-up observation after each. confirmed_observation_intervals counts only
exchanges with a received observation, including warmup, not sent actions or
the official evaluator's executed-step total. The compatibility field
official_action_intervals has this same confirmed-observation meaning. If the
final action ends the environment without another observation, it is not counted.
Submitting a
command does not advance physics; state(), runtime.feedback() and observe do not
advance physics. After execute_python advances any intervals, its reply includes
the latest head, left-wrist and right-wrist RGB images once, in that order.
set_gripper('left' or 'right', value) queues a normalized opening target [0,1],
0 closed and 1 open; step() applies it. Reported gripper state is the previous
control target, NOT measured jaw position or contact force. Command submission
does not certify grasp success. SHAPE_HAND, MAKE_CONTACT, BREAK_CONTACT,
CONTROL_GRASP and APPLY_WRENCH lack public evidence and are unsupported.
Contact dictionaries contain no measured contact evidence; an empty dictionary
does not certify absence of contact. No object truth, friction coefficient,
wrench, reward, success labels, depth or camera calibration is supplied here.
Robot velocities are finite differences of successive public observations, not
instantaneous measurements. A real evaluator-counted initial hold obtains the
second observation. Cartesian target rates are bounded; joint-speed checks are
retrospective. Public RGB pixels retain their official camera names and colors.
Automatic runtime feedback keeps all active/changed reports and current control
handles, omitting only unchanged terminal reports already delivered. Explicit
runtime.feedback() and runtime.query() remain complete. Long stdout is previewed
with an explicit omission count; full text is retained only in the local audit.
np and math expose documented numerical operations only; file/network access,
imports and private attributes are outside this simulation policy interface.
"""


class RoboDojoEpisode(BaseEpisode):
    """Use optimized ManipISA contracts without extending the observation boundary."""

    def __init__(self, initial_obs, exchange, out, *, wall_limit=600.0):
        self.wall_limit = float(wall_limit)
        if not math.isfinite(self.wall_limit) or self.wall_limit <= 0:
            raise ValueError("wall_limit must be finite and positive")
        self.adapter = RoboDojoAdapter(initial_obs, exchange)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = self.terminal_time = self.reason = None
        self.queries = self.tool_calls = self.visual_feedback_count = 0
        self.task = {"description": self.adapter.observation["instruction"]}
        self.runtime = self.namespace = None
        self._feedback_reports = {}

    def warmup(self):
        if self.runtime is not None:
            raise RuntimeError("Episode already warmed up")
        self.adapter.step()  # One real, official evaluator-counted hold interval.
        self.runtime = Runtime(self.adapter)
        self.namespace = {
            "__builtins__": public_builtins(), "np": restricted_np(), "math": restricted_math(),
            "step": self.step, "state": self.public_state, "set_gripper": self.adapter.set_gripper,
            "runtime": SimpleNamespace(**{name: getattr(self.runtime, name) for name in
                ("submit", "query", "cancel", "update", "feedback")}),
        }
        for name in ("Instruction", "Opcode", "Mode", "PoseGoal", "ShapeGoal", "Status"):
            self.namespace[name] = getattr(types, name)

    def step(self, n=1):
        if type(n) is not int or n < 1:
            raise ValueError("step(n) requires a positive integer")
        for _ in range(n):
            self.check_time()
            self.runtime.step(return_snapshot=False)
            self.check_time()

    def public_state(self):
        return {**super().public_state(), **self._interval_counts(), "frame_semantics": dict(FRAME_SEMANTICS),
                "evidence_availability": dict(EVIDENCE_AVAILABILITY)}

    def _interval_counts(self, value=None):
        count = self.steps if value is None else value
        return {"confirmed_observation_intervals": count, "official_action_intervals": count,
                "interval_count_semantics": "confirmed follow-up observations consumed by the Runtime adapter, "
                    "including warmup; official_action_intervals is a legacy alias, not the evaluator's executed-step total"}

    def _append_log(self, filename, record):
        with (self.out / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")

    def _capture_visual(self, source):
        from PIL import Image

        self.check_time()
        try:
            step = self.steps
            observation_id = self.queries + 1
            blocks, stored = [], []
            image_dir = self.out / "observations"
            image_dir.mkdir(exist_ok=True)
            for camera in CAMERAS:
                stream = io.BytesIO()
                Image.fromarray(self.adapter.observation["vision"][camera]["color"]).save(stream, format="PNG")
                pixels = stream.getvalue()
                filename = f"{observation_id:06d}_{camera}.png"
                (image_dir / filename).write_bytes(pixels)
                stored.append({"camera": camera, "path": "observations/" + filename,
                               "png_sha256": hashlib.sha256(pixels).hexdigest()})
                blocks.append({"type": "image", "mimeType": "image/png",
                               "data": base64.b64encode(pixels).decode("ascii")})
            if self.steps != step:
                raise RuntimeError("Image capture must not advance the official environment")
            state = self.public_state()
            visual = {"status": "captured", "source": source, "observation_id": observation_id,
                      **self._interval_counts(step), "camera_ids": list(CAMERAS), "state": state}
            self._append_log("observations.jsonl", {**visual, "images": stored})
            self.queries += 1
            if source == "automatic":
                self.visual_feedback_count += 1
            return visual, blocks
        finally:
            # A slow camera, encoder, or failing disk must not yield an expired reply.
            self.check_time()

    def capture_visual_feedback(self):
        return self._capture_visual("automatic")

    def observe_content(self):
        visual, blocks = self._capture_visual("observe")
        content = [{"type": "text", "text": json.dumps(json_safe(
            {**visual["state"], "visual_feedback": {k: v for k, v in visual.items() if k != "state"}}),
            ensure_ascii=False, allow_nan=False)}, *blocks]
        self.check_time()
        return content

    def agent_instructions(self):
        return (f"Task: {self.task['description']}\nYou control RoboDojo's original dual ARX X5 robot "
                f"and grippers through ManipISA. Wall budget: {self.wall_limit:g} seconds.\n"
                "Use only the supplied task, RGB and public robot observations. Physics advances "
                "only through step(n). Do not inspect files, network, private attributes, evaluator "
                "state or scene internals, reset, or teleport. Use observe to request RGB without "
                "moving, and read_api('instructions') for instruction definitions.\n" + API_REFERENCE)

    def _execute_reply(self, code):
        before = self.steps
        capture, error, stopped = {}, None, None
        self._append_log("programs.jsonl", {"tool_call": self.tool_calls,
            **self._interval_counts(before), "code": code})
        try:
            execute_program(code, self.namespace, self.started + self.wall_limit, capture=capture)
        except EpisodeStopped as exc:
            stopped = exc
            error = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        images = []
        visual = {"status": "unchanged", **self._interval_counts()}
        if stopped is None and self.steps > before:
            try:
                visual, images = self.capture_visual_feedback()
            except EpisodeStopped as exc:
                stopped = exc
            except Exception as exc:
                visual = {"status": "error", **self._interval_counts(),
                          "error": {"type": type(exc).__name__, "message": str(exc)}}
        if stopped is not None:
            visual = {"status": "withheld", **self._interval_counts(), "reason": str(stopped)}
            images = []
        stdout = capture.get("stdout", "")
        preview, omitted = stdout_preview(stdout)
        full_runtime = self.runtime.feedback()
        grippers = self.adapter.gripper_targets()
        summary = summarize_feedback(full_runtime, previous_reports=self._feedback_reports)
        result = {"feedback_format": FEEDBACK_FORMAT, "stdout": preview,
            "stdout_total_chars": len(stdout), "stdout_omitted_chars": omitted,
            **self._interval_counts(), "runtime": summary, "grippers": grippers,
            "frame_semantics": dict(FRAME_SEMANTICS),
            "evidence_availability": dict(EVIDENCE_AVAILABILITY), "visual_feedback": visual}
        if error is not None:
            result["error"] = error
        text = json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False)
        reply = {"content": [{"type": "text", "text": text}, *images]}
        if error is not None or visual["status"] == "error":
            reply["isError"] = True
        record = {"stdout": stdout, "stdout_omitted_chars": omitted, "error": error,
            "runtime": full_runtime, "grippers": grippers, "visual_feedback": visual,
            "action_intervals_before": before, "confirmed_observation_intervals_before": before,
            **self._interval_counts(),
            "image_count": len(images), "reply_chars": len(text)}
        baseline = {report["call_id"]: report for report in full_runtime["instructions"]}
        return reply, record, baseline, stopped

    def _deliver(self, name, reply, record, baseline=None, stopped=None):
        record.update(tool_call=self.tool_calls, tool=name, feedback_format=FEEDBACK_FORMAT,
                      reply_withheld=stopped is not None)
        if stopped is None:
            try:
                self.check_time()
            except EpisodeStopped as exc:
                stopped = exc
                record["reply_withheld"] = True
        try:
            self._append_log("tool-feedback.jsonl", record)
            if stopped is not None:
                self.stop(str(stopped) or "episode_stopped")
                raise stopped
            try:
                self.check_time()
            except EpisodeStopped as exc:
                # Preserve the fact that logging itself prevented delivery.
                self._append_log("tool-feedback.jsonl", {
                    "tool_call": self.tool_calls, "tool": name, "reply_withheld": True,
                    "event": "reply_withheld_after_logging", "reason": str(exc)})
                raise
            if baseline is not None:
                self._feedback_reports = baseline
            return reply
        except BaseException:
            # A serialization/logging error cannot turn termination into a retry.
            if stopped is not None:
                self.stop(str(stopped) or "episode_stopped")
                raise stopped
            self.check_time()
            raise

    def tool(self, name, arguments):
        self.check_time()
        self.tool_calls += 1
        baseline, stopped = None, None
        try:
            if name == "execute_python":
                reply, record, baseline, stopped = self._execute_reply(arguments["code"])
            elif name == "observe":
                reply = {"content": self.observe_content()}
                record = {"observation_id": self.queries, "image_count": 3,
                          **self._interval_counts(), "public_state": self.public_state()}
            elif name == "read_api":
                requested = arguments.get("name")
                if requested == "index":
                    text = json.dumps(["robodojo", "instructions"])
                elif requested == "robodojo":
                    text = API_REFERENCE
                elif requested == "instructions":
                    text = Path(types.__file__).read_text(encoding="utf-8")
                else:
                    raise ValueError("Unknown API document")
                reply = {"content": [{"type": "text", "text": text}]}
                record = {"document": requested, "text": text, "image_count": 0}
            else:
                raise ValueError("Unknown tool")
        except (Exception, EpisodeStopped) as exc:
            if isinstance(exc, EpisodeStopped):
                stopped = exc
            error = {"type": type(exc).__name__, "message": str(exc)}
            reply = {"isError": True, "content": [{"type": "text", "text": json.dumps({"error": error})}]}
            record = {"error": error, "image_count": 0, **self._interval_counts()}
        return self._deliver(name, reply, record, baseline, stopped)
