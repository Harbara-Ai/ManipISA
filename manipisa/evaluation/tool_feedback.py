"""Compact policy replies and complete local records for benchmark Python tools."""
from __future__ import annotations

import json
from contextlib import nullcontext

from ..feedback import summarize_feedback
from .programs import EpisodeStopped, execute_program


FEEDBACK_FORMAT = "compact-visual-v2"
STDOUT_LIMIT = 4000
LOG_NAME = "tool-feedback.jsonl"


def stdout_preview(text):
    """Keep both ends of long output; never silently discard diagnostics."""
    if len(text) <= STDOUT_LIMIT:
        return text, 0
    marker = "\n... [stdout truncated; full output in tool-feedback.jsonl] ...\n"
    available = STDOUT_LIMIT - len(marker)
    head = available // 2
    tail = available - head
    return text[:head] + marker + text[-tail:], len(text) - available


def _timed(episode, label):
    profiler = getattr(episode, "profiler", None)
    return profiler.span(label) if profiler is not None else nullcontext()


def execute_with_feedback(episode, code, *, normalize=lambda value: value):
    """Execute once, record full diagnostics, then return a bounded policy reply.

    The episode owns the instruction baseline. Explicit runtime queries never
    advance it. A program that advances physics receives one fresh visual
    observation, without advancing physics during capture. Ordinary exceptions
    preserve partial effects; terminal exceptions still propagate so the host
    withholds the reply from the policy.
    """
    capture = {}
    start_step = episode.steps
    error = None
    stopped = None
    try:
        with _timed(episode, "tool.program"):
            execute_program(code, episode.namespace, episode.started + episode.wall_limit,
                            capture=capture)
    except EpisodeStopped as exc:
        stopped = exc
        error = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}

    try:
        images = []
        visual = {"status": "unchanged", "physics_step": episode.steps}
        if stopped is not None:
            visual = {"status": "withheld", "physics_step": episode.steps,
                      "reason": str(stopped)}
        elif episode.steps > start_step:
            try:
                episode.check_time()
            except EpisodeStopped as exc:
                stopped = exc
                visual = {"status": "withheld", "physics_step": episode.steps,
                          "reason": str(exc)}
            if stopped is None:
                capture_step = episode.steps
                try:
                    with _timed(episode, "feedback.visual"):
                        visual, images = episode.capture_visual_feedback()
                    # Capture must not move the robot or make the saved image's
                    # timestamp disagree with the current structured feedback.
                    if episode.steps != capture_step:
                        raise RuntimeError("Visual capture must not advance physics")
                except EpisodeStopped as exc:
                    stopped = exc
                    images = []
                    visual = {"status": "withheld", "physics_step": episode.steps,
                              "reason": str(exc)}
                except Exception as exc:
                    images = []
                    visual = {"status": "error", "physics_step": episode.steps,
                              "error": {"type": type(exc).__name__, "message": str(exc)}}
                # Check even after an ordinary camera failure: an expired
                # capture cannot turn into another model-visible retry request.
                if stopped is None:
                    try:
                        episode.check_time()
                    except EpisodeStopped as exc:
                        stopped = exc
        stdout = capture["stdout"]
        preview, omitted = stdout_preview(stdout)
        result = {"feedback_format": FEEDBACK_FORMAT, "stdout": preview,
                  "stdout_total_chars": len(stdout), "stdout_omitted_chars": omitted,
                  "physics_step": episode.steps, "visual_feedback": visual}
        if error is not None:
            result["error"] = error
        full_runtime = None
        if episode.method == "manipisa":
            with _timed(episode, "feedback.full"):
                full_runtime = normalize(episode.runtime.feedback())
            # The complete snapshot is already required for the audit log; reuse it
            # rather than serializing the runtime a second time for the policy.
            with _timed(episode, "feedback.compact"):
                result["runtime"] = summarize_feedback(full_runtime,
                    previous_reports=getattr(episode, "_feedback_reports", None))
        with _timed(episode, "feedback.encode"):
            result = normalize(result)
            text = json.dumps(result, ensure_ascii=False)
        record = {"feedback_format": FEEDBACK_FORMAT, "tool_call": episode.tool_calls,
                  "physics_step_before": start_step, "physics_step": episode.steps,
                  "stdout": stdout, "stdout_omitted_chars": omitted, "error": error,
                  "runtime": full_runtime, "reply_withheld": stopped is not None,
                  "visual_feedback": result["visual_feedback"],
                  "image_count": len(images), "reply_chars": len(text)}
        # Include serialization/logging overhead in the same wall budget. If this
        # crosses the deadline, retain the full record and withhold the response.
        if stopped is None:
            try:
                episode.check_time()
            except EpisodeStopped as exc:
                stopped = exc
                record["reply_withheld"] = True
        with _timed(episode, "feedback.log"):
            with (episode.out / LOG_NAME).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if stopped is not None:
            raise stopped
        try:
            episode.check_time()
        except EpisodeStopped as exc:
            stopped = exc
            # Disk I/O itself may cross the deadline. Append a correction instead
            # of claiming that the preceding candidate reply was delivered.
            with (episode.out / LOG_NAME).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"feedback_format": FEEDBACK_FORMAT,
                    "tool_call": episode.tool_calls, "event": "reply_withheld_after_logging",
                    "error": {"type": type(exc).__name__, "message": str(exc)}}) + "\n")
            raise
        if full_runtime is not None:
            episode._feedback_reports = {r["call_id"]: r for r in full_runtime["instructions"]}
        reply = {"content": [{"type": "text", "text": text}, *images]}
        if error is not None or visual["status"] == "error":
            reply["isError"] = True
        return reply
    except Exception:
        # Feedback encoding or disk failure may itself consume the remaining
        # budget. Preserve its ordinary exception only while time remains.
        if stopped is None:
            try:
                episode.check_time()
            except EpisodeStopped as exc:
                stopped = exc
        raise
    finally:
        # Terminal boundaries take precedence even if serialization or audit
        # storage fails; never turn EpisodeStopped into an ordinary tool reply.
        if stopped is not None:
            raise stopped
