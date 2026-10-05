"""Lossless contact grouping and stateless report selection for model feedback."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
import numbers

from .types import TERMINAL, json_safe


SCHEMA_VERSION = "manipisa.feedback.summary.v1"


def _record(value):
    return json_safe(asdict(value) if is_dataclass(value) else deepcopy(value))


def _zero(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value) and value == 0)


def _zero_vector(value):
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(_zero(v) for v in value)


def _empty_contact(record):
    """Compress only explicitly known empty contacts, never missing evidence."""
    required = {"actor", "target", "timestamp", "present", "normal_force", "force_on_target_w",
                "torque_on_target_world_origin_w", "points", "separation_m", "valid",
                "wrench_valid", "source", "detail"}
    stamp = record.get("timestamp")
    return (required <= record.keys() and record["present"] is False
            and record["valid"] is True and record["wrench_valid"] is True
            and isinstance(stamp, numbers.Real) and not isinstance(stamp, bool) and math.isfinite(stamp)
            and all(isinstance(record[k], str) and record[k] for k in ("actor", "target", "source"))
            and _zero(record["normal_force"]) and _zero_vector(record["force_on_target_w"])
            and _zero_vector(record["torque_on_target_world_origin_w"])
            and record["points"] == [] and record["separation_m"] is None and record["detail"] == "")


def compact_contacts(contacts):
    """Accept dataclass contacts or serialized historical records; retain every ID."""
    explicit, groups = {}, {}
    for contact_id, value in contacts.items():
        record = _record(value)
        if not _empty_contact(record):
            explicit[contact_id] = record
            continue
        # Compare the entire record, including any additional metadata. This
        # prevents different actors, sources, timestamps or future fields merging.
        key = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key not in groups:
            groups[key] = {"ids": [], "record": record}
        groups[key]["ids"].append(contact_id)
    compressed = []
    for group in groups.values():
        if len(group["ids"]) < 2:
            explicit[group["ids"][0]] = group["record"]
        else:
            compressed.append(group)
    return explicit, compressed


def expand_contacts(state):
    """Reconstruct a complete contact dictionary from a summary state."""
    result = deepcopy(state.get("contacts", {}))
    for group in state.get("empty_contact_groups", []):
        for contact_id in group["ids"]:
            if contact_id in result:
                raise ValueError("Repeated contact ID in summary")
            result[contact_id] = deepcopy(group["record"])
    return result


def summarize_feedback(feedback, *, previous_reports=None):
    """Summarize full feedback dictionaries, including saved historical logs.

    ``previous_reports`` maps call IDs to full serialized reports from the last
    delivered feedback. It is caller-owned and never modified. Merge returned
    ``instructions`` into that map after delivery; reads do not consume a cursor.
    Current state and the entire current handle collection are always included.
    """
    previous = {} if previous_reports is None else previous_reports
    result = {key: deepcopy(value) for key, value in feedback.items() if key not in ("state", "instructions")}
    state = {key: deepcopy(value) for key, value in feedback["state"].items() if key != "contacts"}
    state["contacts"], state["empty_contact_groups"] = compact_contacts(feedback["state"].get("contacts", {}))
    state["contact_encoding"] = "explicit_plus_identical_empty_groups"
    state["contacts_complete"] = True
    selected, omitted = [], 0
    for value in feedback["instructions"]:
        report = _record(value)
        if report.get("status") in TERMINAL and previous.get(report["call_id"]) == report:
            omitted += 1
        else:
            selected.append(report)
    result.update(schema_version=SCHEMA_VERSION, state=state, instructions=selected,
                  instructions_scope="active_and_changed_terminal", active_instructions_complete=True,
                  omitted_unchanged_terminal_count=omitted, control_handles_complete=True)
    return result
