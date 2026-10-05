"""Native Codex events: retain missing/ambiguous costs instead of guessing."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def otel_records(payload: dict) -> list[dict]:
    records = []
    for resource in payload.get("resourceLogs", []):
        for scope in resource.get("scopeLogs", []):
            for record in scope.get("logRecords", []):
                attrs = {item["key"]: next(iter(item["value"].values()), None)
                         for item in record.get("attributes", [])}
                # Account identity and prompt/tool text are not scoring evidence.
                attrs = {k: v for k, v in attrs.items()
                         if not k.startswith("user.") and k not in ("prompt", "output", "tool.output")}
                records.append(attrs)
    return records


def codex_costs(events_path: Path, records: list[dict], *, wall_time_s: float, process_completed: bool,
                task_start_utc: str | None = None, task_end_utc: str | None = None) -> dict:
    events, parse_errors = [], 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1
    turns = [e for e in events if e.get("type") == "turn.completed"]
    usage = turns[0].get("usage", {}) if len(turns) == 1 else {}
    def count(name):
        value = usage.get(name)
        return value if type(value) is int and value >= 0 else None
    inputs, outputs = count("input_tokens"), count("output_tokens")
    tokens = inputs + outputs if inputs is not None and outputs is not None else None
    request_events = [r for r in records if r.get("event.name") == "codex.websocket_request" or
                      (r.get("event.name") == "codex.api_request" and r.get("endpoint") == "/responses")]
    result = {"wall_time_s": wall_time_s, "native_total_tokens": None,
            "input_tokens": inputs, "output_tokens": outputs,
            "cached_input_tokens": count("cached_input_tokens"),
            "reasoning_output_tokens": count("reasoning_output_tokens"),
            "model_request_count": None, "observed_transport_request_events": len(request_events),
            "tokens_complete": False,
            "requests_complete": False, "settled": False,
            "measurement_source": "codex exec turn.completed usage + local OTel",
            "limitation": "Task-window inference accounting not verified; missing values remain N/A",
            "raw_turn_usage_including_setup": usage, "raw_turn_total_tokens": tokens, "event_parse_errors": parse_errors}
    # Auto-review can initialize before task delivery, then make auxiliary
    # inference calls without exporting their full usage. Its prewarm event
    # alone must not make a reviewed episode look fully metered.
    auxiliary_models = sorted({r["model"] for r in records if r.get("model") and r["model"] != "gpt-6.1-sol"})
    if auxiliary_models:
        result["auxiliary_models"] = auxiliary_models
        result["limitation"] = "Auxiliary model usage is not fully exported: " + ", ".join(auxiliary_models)
        return result
    if task_start_utc is None or task_end_utc is None:
        return result
    start, end = datetime.fromisoformat(task_start_utc), datetime.fromisoformat(task_end_utc)
    if end < start:
        return result
    def timestamp(record):
        try:
            return datetime.fromisoformat(record["event.timestamp"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError):
            return None
    # Validated transport: one serial native CLI conversation, websocket sends
    # plus response.completed usage. Startup/prewarm events precede task delivery.
    # HTTP fallbacks and failed or ambiguous streams are retained as incomplete.
    # Pair the whole serial stream first. A streamed start_task call can return
    # before that setup request's response.completed event; filtering finishes
    # by the task timestamp would accidentally count this setup completion.
    sends = list(request_events)
    completions = [r for r in records if r.get("event.name") == "codex.sse_event" and
                   r.get("event.kind") == "response.completed"]
    result["task_window_request_events"] = sum(timestamp(r) is not None and start <= timestamp(r) <= end for r in sends)
    has_http = any(r.get("event.name") == "codex.api_request" for r in request_events)
    if has_http or not sends or parse_errors or len(completions) != len(sends):
        return result
    if any(timestamp(r) is None for r in sends + completions):
        return result
    send_times = [timestamp(r) for r in sends]
    finish_times = [timestamp(r) for r in completions]
    if len(set(send_times)) != len(send_times) or len(set(finish_times)) != len(finish_times):
        return result
    sends.sort(key=timestamp)
    completions.sort(key=timestamp)
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_output_tokens": 0}
    mappings = {"input_tokens": "input_token_count", "output_tokens": "output_token_count",
                "cached_input_tokens": "cached_token_count", "reasoning_output_tokens": "reasoning_token_count"}
    task_request_count = 0
    for i, (send, completed) in enumerate(zip(sends, completions)):
        if send.get("success") not in (True, "true") or timestamp(completed) < timestamp(send):
            return result
        if i + 1 < len(sends) and timestamp(completed) > timestamp(sends[i + 1]):
            return result
        if send.get("conversation.id") != completed.get("conversation.id"):
            return result
        if completed.get("app.version") != "0.160.0" or completed.get("model") != "gpt-6.1-sol":
            return result
        if not start <= timestamp(send) <= end:
            continue
        task_request_count += 1
        for target, source in mappings.items():
            value = completed.get(source)
            if isinstance(value, bool) or not str(value).isdigit():
                return result
            totals[target] += int(value)
    result.update(totals)
    result.update(native_total_tokens=totals["input_tokens"] + totals["output_tokens"],
                  model_request_count=task_request_count, tokens_complete=True, requests_complete=True, settled=True,
                  limitation=None, measurement_source="Codex 0.160.0 serial websocket sends and paired response.completed usage within task window")
    return result
