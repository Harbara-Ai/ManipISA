"""Summary feedback preserves control decisions without consuming read cursors."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from manipisa import Runtime, Status, Mode, ContactState, ContactPoint
from manipisa.feedback import SCHEMA_VERSION, compact_contacts, expand_contacts, summarize_feedback
from manipisa.types import StateSnapshot
from test_runtime import TestAdapter, shape, move, finish


def empty_contact(**changes):
    value = ContactState("hand", "ball", 1.0, False, 0.0,
                         (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), wrench_valid=True)
    return replace(value, **changes)


def serialized_contact(**changes):
    return StateSnapshot(1.0, 0, contacts={"contact": empty_contact(**changes)}).to_dict()["contacts"]["contact"]


class ContactSummaryTests(unittest.TestCase):
    def test_empty_contacts_group_losslessly_and_keep_ids(self):
        original = {"thumb": serialized_contact(), "index": serialized_contact(), "middle": serialized_contact()}
        before = deepcopy(original)
        explicit, groups = compact_contacts(original)
        self.assertEqual(explicit, {})
        self.assertEqual(groups, [{"ids": ["thumb", "index", "middle"], "record": original["thumb"]}])
        state = {"contacts": explicit, "empty_contact_groups": groups}
        self.assertEqual(expand_contacts(state), original)
        self.assertEqual(original, before)
        groups[0]["record"]["actor"] = "modified"
        self.assertEqual(original, before)

    def test_empty_singleton_stays_explicit_without_group_overhead(self):
        original = {"thumb": serialized_contact()}
        self.assertEqual(compact_contacts(original), (original, []))

    def test_invalid_unknown_present_and_nonzero_records_are_never_grouped(self):
        point = ContactPoint((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0)
        variants = (
            {"valid": False}, {"wrench_valid": False}, {"present": True},
            {"normal_force": 1e-12}, {"normal_force": -1e-12},
            {"force_on_target_w": (0.0, 1e-12, 0.0)},
            {"torque_on_target_world_origin_w": (0.0, 0.0, 1e-12)},
            {"normal_force": float("nan")}, {"timestamp": float("nan")},
            {"detail": "missing friction data"}, {"points": (point,)}, {"separation_m": 0.0},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                original = {"left": serialized_contact(**changes), "right": serialized_contact(**changes)}
                explicit, groups = compact_contacts(original)
                self.assertEqual(explicit, original)
                self.assertEqual(groups, [])
                json.dumps(explicit, allow_nan=False)

    def test_missing_unknown_or_malformed_evidence_remains_explicit(self):
        variants = []
        for field in ("valid", "wrench_valid", "present", "timestamp", "source"):
            missing = serialized_contact()
            del missing[field]
            variants.append(missing)
            unknown = serialized_contact()
            unknown[field] = None
            variants.append(unknown)
        malformed = serialized_contact()
        malformed["normal_force"] = False
        variants.append(malformed)
        for record in variants:
            with self.subTest(record=record):
                original = {"a": record, "b": deepcopy(record)}
                self.assertEqual(compact_contacts(original), (original, []))

    def test_grouping_retains_all_metadata_including_future_fields(self):
        variants = [serialized_contact()]
        for field, value in (("actor", "other_hand"), ("target", "other_ball"),
                             ("timestamp", 2.0), ("source", "other_sensor")):
            record = serialized_contact()
            record[field] = value
            variants.append(record)
        extra = serialized_contact()
        extra["future_metadata"] = {"calibration": "different"}
        variants.append(extra)
        contacts = {f"{i}_{side}": deepcopy(record) for i, record in enumerate(variants) for side in ("a", "b")}
        explicit, groups = compact_contacts(contacts)
        self.assertEqual(len(groups), len(variants))
        self.assertEqual(expand_contacts({"contacts": explicit, "empty_contact_groups": groups}), contacts)


class RuntimeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TestAdapter()
        self.runtime = Runtime(self.adapter)

    def baseline(self):
        return {r["call_id"]: r for r in self.runtime.feedback()["instructions"]}

    def test_current_state_and_active_report_are_self_contained(self):
        call = self.runtime.submit(shape(target=0.4))
        baseline = self.baseline()
        summary = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual([r["call_id"] for r in summary["instructions"]], [call.call_id])
        self.assertTrue(summary["active_instructions_complete"])
        self.assertTrue(summary["state"]["contacts_complete"])
        full = self.runtime.feedback()
        for key, value in full["state"].items():
            if key != "contacts":
                self.assertEqual(summary["state"][key], value)
        self.runtime.step()
        later = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertEqual(later["state"]["timestamp"], self.adapter.now)
        self.assertNotEqual(later["state"]["joints"], summary["state"]["joints"])

    def test_terminal_changes_and_new_rejections_are_reported(self):
        call = self.runtime.submit(shape())
        baseline = self.baseline()
        terminal = finish(self.runtime, call.call_id)
        summary = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertEqual(summary["instructions"], [terminal.to_dict()])
        terminal_baseline = self.baseline()
        summary = self.runtime.feedback_summary(previous_reports=terminal_baseline)
        self.assertEqual(summary["instructions"], [])
        self.assertEqual(summary["omitted_unchanged_terminal_count"], 1)
        rejected = self.runtime.submit(shape())  # retained handle still owns the resource
        summary = self.runtime.feedback_summary(previous_reports=terminal_baseline)
        self.assertEqual(summary["instructions"], [rejected.to_dict()])
        self.assertEqual(rejected.status, Status.REJECTED)
        # Any terminal report change, not only its status, must be delivered.
        self.runtime._rejections[rejected.call_id] = replace(rejected, detail="updated diagnostic")
        terminal_baseline[rejected.call_id] = rejected.to_dict()
        summary = self.runtime.feedback_summary(previous_reports=terminal_baseline)
        self.assertEqual(summary["instructions"][0]["detail"], "updated diagnostic")

    def test_active_report_is_retained_even_when_identical_to_baseline(self):
        call = self.runtime.submit(move(target=0.0, mode=Mode.SUSTAIN, dwell_s=0.0))
        self.assertEqual(call.status, Status.RUNNING)
        for _ in range(20):
            if self.runtime.query(call.call_id).status == Status.ACTIVE:
                break
            self.runtime.step()
        report = self.runtime.query(call.call_id)
        self.assertEqual(report.status, Status.ACTIVE)
        baseline = self.baseline()
        summary = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertEqual(summary["instructions"], [report.to_dict()])
        self.assertEqual(summary["omitted_unchanged_terminal_count"], 0)

    def test_query_full_feedback_and_summary_do_not_consume_baseline(self):
        call = self.runtime.submit(shape())
        baseline = self.baseline()
        finish(self.runtime, call.call_id)
        before = deepcopy(baseline)
        expected = self.runtime.feedback_summary(previous_reports=baseline)
        self.runtime.query(call.call_id)
        self.runtime.feedback()
        self.runtime.feedback_summary()
        self.assertEqual(self.runtime.feedback_summary(previous_reports=baseline), expected)
        self.assertEqual(baseline, before)

    def test_all_handle_states_and_handoff_removal_are_visible(self):
        call = self.runtime.submit(shape())
        terminal = finish(self.runtime, call.call_id)
        baseline = self.baseline()
        handle = terminal.control_handle
        for state in ("HOLDING_POSITION", "EXPIRED", "FAULTED"):
            self.runtime._holds[handle].state = state
            summary = self.runtime.feedback_summary(previous_reports=baseline)
            self.assertTrue(summary["control_handles_complete"])
            self.assertEqual(summary["instructions"], [])
            self.assertEqual(summary["control_handles"][handle], self.runtime.feedback()["control_handles"][handle])
            self.assertEqual(summary["control_handles"][handle]["owner"], call.call_id)
        # Successful explicit takeover removes the previous handle from the complete set.
        replacement = self.runtime.submit(shape(target=0.2), replace_handle=handle)
        self.assertEqual(replacement.status, Status.RUNNING)
        summary = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertNotIn(handle, summary["control_handles"])
        self.assertEqual(summary["instructions"][0]["call_id"], replacement.call_id)

    def test_no_observation_step_or_full_snapshot_materialization(self):
        self.runtime._snapshot = replace(self.runtime._snapshot,
            contacts={"thumb": empty_contact(), "index": empty_contact()})
        self.runtime._state_error = "sensor read failed"
        with patch.object(self.adapter, "observe", side_effect=AssertionError("unexpected observation")), \
                patch.object(self.adapter, "step", side_effect=AssertionError("unexpected physics")), \
                patch.object(StateSnapshot, "to_dict", side_effect=AssertionError("full snapshot serialization")):
            summary = self.runtime.feedback_summary()
        self.assertFalse(summary["state_valid"])
        self.assertEqual(summary["state_error"], "sensor read failed")
        self.assertEqual(len(summary["state"]["empty_contact_groups"]), 1)
        summary["state"]["joints"]["hand"]["position"][0] = 99
        self.assertEqual(self.runtime._snapshot.joints["hand"].position, (0.0,))

    def test_saved_dictionary_helper_matches_runtime_and_is_nonmutating(self):
        call = self.runtime.submit(shape())
        finish(self.runtime, call.call_id)
        self.runtime._snapshot = replace(self.runtime._snapshot,
            contacts={"thumb": empty_contact(), "index": empty_contact(),
                      "unknown": empty_contact(wrench_valid=False)})
        saved = json.loads(json.dumps(self.runtime.feedback(), allow_nan=False))
        original = deepcopy(saved)
        baseline = {r["call_id"]: r for r in saved["instructions"]}
        offline = summarize_feedback(saved, previous_reports=baseline)
        live = self.runtime.feedback_summary(previous_reports=baseline)
        self.assertEqual(offline, live)
        self.assertEqual(expand_contacts(offline["state"]), saved["state"]["contacts"])
        self.assertEqual(saved, original)
        json.dumps(live, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
