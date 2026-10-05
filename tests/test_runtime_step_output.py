"""Skipping a discarded snapshot must preserve every control/evidence update."""
import unittest
from unittest.mock import patch

from manipisa import Runtime
from test_runtime import TestAdapter, shape


class StepOutputTests(unittest.TestCase):
    def test_default_snapshot_remains_isolated(self):
        runtime = Runtime(TestAdapter())
        result = runtime.step()
        self.assertIsNot(result, runtime._snapshot)
        result.joints.clear()
        self.assertTrue(runtime._snapshot.joints)

    def test_host_path_skips_only_return_copy_and_matches_control_trace(self):
        reference, candidate = Runtime(TestAdapter()), Runtime(TestAdapter())
        r = reference.submit(shape())
        c = candidate.submit(shape())
        for _ in range(15):
            reference.step()
            # No terminal transition in a tick needs a deepcopy; this assertion
            # also guards against accidentally materializing the return anyway.
            with patch("manipisa.runtime.deepcopy", side_effect=AssertionError("unused copy")):
                self.assertIsNone(candidate.step(return_snapshot=False))
            self.assertEqual(reference._snapshot, candidate._snapshot)
            self.assertEqual(reference.adapter.commands, candidate.adapter.commands)
            self.assertEqual(reference.adapter.steps, candidate.adapter.steps)
            old, new = reference.query(r.call_id).to_dict(), candidate.query(c.call_id).to_dict()
            for field in ("status", "timestamp", "reason", "detail", "goal", "evidence", "active_since"):
                self.assertEqual(old[field], new[field])
            self.assertEqual(bool(old["control_handle"]), bool(new["control_handle"]))


if __name__ == "__main__":
    unittest.main()
