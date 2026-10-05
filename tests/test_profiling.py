"""Host-only profiling regressions; no simulator, GPU, or model is launched."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from manipisa.profiling import TimingProfiler, attach_episode_profiler


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.clock_patch = patch("manipisa.profiling.perf_counter", self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.profiler = TimingProfiler()
        self.addCleanup(self.profiler.restore)

    def test_nested_exclusive_and_max_are_aggregated(self):
        with self.profiler.span("parent"):
            self.clock.advance(1)
            with self.profiler.span("child"):
                self.clock.advance(2)
                with self.profiler.span("leaf"):
                    self.clock.advance(3)
            self.clock.advance(4)
            with self.profiler.span("child"):
                self.clock.advance(1)
        snapshot = self.profiler.snapshot()
        stats = snapshot["timings"]
        self.assertEqual(stats["parent"], dict(count=1, inclusive_seconds=11.0,
                         exclusive_seconds=5.0, max_seconds=11.0, failures=0))
        self.assertEqual(stats["child"], dict(count=2, inclusive_seconds=6.0,
                         exclusive_seconds=3.0, max_seconds=5.0, failures=0))
        self.assertEqual(stats["leaf"]["exclusive_seconds"], 3.0)
        self.assertEqual(snapshot["root_seconds"], 11.0)
        self.assertEqual(snapshot["active_spans"], 0)
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)

    def test_exception_is_unchanged_and_failed_parent_is_counted(self):
        error = KeyboardInterrupt("interrupted")
        with self.assertRaises(KeyboardInterrupt) as caught:
            with self.profiler.span("parent"):
                with self.profiler.span("child"):
                    self.clock.advance(2)
                    raise error
        self.assertIs(caught.exception, error)
        stats = self.profiler.snapshot()["timings"]
        self.assertEqual(stats["child"]["failures"], 1)
        self.assertEqual(stats["parent"]["failures"], 1)
        self.assertEqual(self.profiler.snapshot()["active_spans"], 0)
        self.profiler.reset()
        self.assertEqual(self.profiler.snapshot()["timings"], {})

    def test_disabled_profiler_does_not_read_clock_or_object_or_label(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError("Disabled profiling inspected an object")
        profiler = TimingProfiler(enabled=False)
        profiler._clock = Mock(side_effect=AssertionError("Read clock"))
        classifier = Mock(side_effect=AssertionError("Classified label"))
        with profiler.span("ignored"):
            value = object()
        self.assertFalse(profiler.wrap(Untouchable(), "step", classifier))
        self.assertIs(attach_episode_profiler(Untouchable(), profiler), profiler)
        profiler._clock.assert_not_called()
        classifier.assert_not_called()
        self.assertIsNotNone(value)
        self.assertEqual(profiler.snapshot()["timings"], {})

    def test_inherited_descriptor_restores_without_instance_shadow(self):
        class Target:
            def method(self, value, *, scale=1):
                return value * scale
        obj = Target()
        descriptor = Target.__dict__["method"]
        self.assertTrue(self.profiler.wrap(obj, "method", "target.method"))
        self.assertFalse(self.profiler.wrap(obj, "method", "another"))
        self.assertEqual(obj.method(3, scale=4), 12)
        self.assertEqual(self.profiler.snapshot()["timings"]["target.method"]["count"], 1)
        self.profiler.restore()
        self.assertNotIn("method", vars(obj))
        self.assertIs(Target.__dict__["method"], descriptor)
        self.assertEqual(obj.method(3, scale=2), 6)
        self.profiler.restore()

    def test_plain_instance_function_restores_exact_object_on_exception(self):
        error = ValueError("original error")
        def operation(*args, **kwargs):
            self.assertEqual(args, (1, 2))
            self.assertEqual(kwargs, {"item": 3})
            raise error
        obj = SimpleNamespace(operation=operation)
        self.profiler.wrap(obj, "operation", "operation")
        try:
            with self.assertRaises(ValueError) as caught:
                obj.operation(1, 2, item=3)
            self.assertIs(caught.exception, error)
        finally:
            self.profiler.restore()
        self.assertIs(obj.operation, operation)
        self.assertEqual(self.profiler.snapshot()["timings"]["operation"]["failures"], 1)

    def test_classmethod_and_staticmethod_descriptors_are_restored(self):
        class Target:
            @classmethod
            def cls_name(cls):
                return cls.__name__

            @staticmethod
            def add_one(value):
                return value + 1
        obj = Target()
        for name in ("cls_name", "add_one"):
            self.assertTrue(self.profiler.wrap(obj, name, name))
        self.assertEqual(obj.cls_name(), "Target")
        self.assertEqual(obj.add_one(4), 5)
        self.profiler.restore()
        self.assertNotIn("cls_name", vars(obj))
        self.assertNotIn("add_one", vars(obj))
        self.assertIsInstance(Target.__dict__["cls_name"], classmethod)
        self.assertIsInstance(Target.__dict__["add_one"], staticmethod)

    def test_slot_callable_is_restored_and_read_only_method_is_skipped(self):
        class Slotted:
            __slots__ = ("operation",)

            def read_only(self):
                return 7
        obj = Slotted()
        original = lambda: 5
        obj.operation = original
        self.assertTrue(self.profiler.wrap(obj, "operation", "operation"))
        self.assertFalse(self.profiler.wrap(obj, "read_only", "read_only"))
        self.assertEqual(obj.operation(), 5)
        self.profiler.restore()
        self.assertIs(obj.operation, original)
        self.assertEqual(obj.read_only(), 7)

    def test_external_replacement_is_not_overwritten_by_restore(self):
        obj = SimpleNamespace(method=lambda: "old")
        self.profiler.wrap(obj, "method", "method")
        obj.method = replacement = lambda: "new"
        self.profiler.restore()
        self.assertIs(obj.method, replacement)

    def test_second_profiler_cannot_double_wrap_an_owned_method(self):
        obj = SimpleNamespace(method=lambda: 3)
        other = TimingProfiler()
        self.profiler.wrap(obj, "method", "first")
        self.assertFalse(other.wrap(obj, "method", "second"))
        self.assertEqual(obj.method(), 3)
        self.assertEqual(other.snapshot()["timings"], {})

    def test_reset_is_rejected_while_active_and_retains_bindings_afterwards(self):
        obj = SimpleNamespace(method=lambda: 3)
        self.profiler.wrap(obj, "method", "method")
        with self.profiler.span("active"):
            with self.assertRaisesRegex(RuntimeError, "span is active"):
                self.profiler.reset()
            self.assertEqual(self.profiler.snapshot()["active_spans"], 1)
        self.profiler.reset()
        self.assertEqual(obj.method(), 3)
        self.assertEqual(set(self.profiler.snapshot()["timings"]), {"method"})

    def test_snapshots_are_detached_and_storage_is_bounded(self):
        profiler = TimingProfiler(max_labels=2)
        for i in range(50):
            with profiler.span(f"unintended_dynamic_{i}"):
                self.clock.advance(1)
        first = profiler.snapshot()
        self.assertLessEqual(len(first["timings"]), 3)
        self.assertEqual(sum(v["count"] for v in first["timings"].values()), 50)
        self.assertIn("profiling.other", first["timings"])
        first["timings"]["profiling.other"]["count"] = -100
        self.assertGreater(profiler.snapshot()["timings"]["profiling.other"]["count"], 0)

    def test_attach_is_inert_deduplicates_robots_and_classifies_contact_cache(self):
        class Contacts:
            def __init__(self):
                self._timestamp = None
                self.reads = 0

            def observe(self, timestamp):
                if self._timestamp != timestamp:
                    self.reads += 1
                    self._timestamp = timestamp
                return {"timestamp": timestamp}

        class Robot:
            def __init__(self):
                self.writes = self.updates = 0

            def write_data_to_sim(self):
                self.writes += 1

            def update(self, dt):
                self.updates += 1
                return dt

        robot, contacts = Robot(), Contacts()
        calls = []
        sim = SimpleNamespace(step=lambda **kw: calls.append(("physics", kw)))
        adapter = SimpleNamespace(contact_source=contacts, _robots={1: robot, 2: robot})
        adapter.observe = lambda: contacts.observe(0.0)
        adapter.command = lambda value: value
        adapter.hold = lambda value: value
        def adapter_step():
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(1 / 60)
        adapter.step = adapter_step
        runtime = SimpleNamespace(step=adapter_step, _assess=lambda: None, _prepare=lambda: None)
        episode = SimpleNamespace(runtime=runtime, adapter=adapter, sim=sim, robot=robot,
                                  tracker=SimpleNamespace(update=lambda: None),
                                  read_objects=lambda objects, ids: (objects, ids),
                                  read_robot=lambda r: r)
        original_reader = episode.read_robot
        self.assertIs(attach_episode_profiler(episode, self.profiler), self.profiler)
        attach_episode_profiler(episode, self.profiler)
        self.assertEqual((robot.writes, robot.updates, contacts.reads), (0, 0, 0))
        self.assertEqual(calls, [])
        self.assertEqual(self.profiler.snapshot()["timings"], {})
        self.assertEqual(adapter.observe(), {"timestamp": 0.0})
        self.assertEqual(contacts.observe(timestamp=0.0), {"timestamp": 0.0})
        self.assertEqual(contacts.observe(1.0), {"timestamp": 1.0})
        adapter.step()
        self.assertIs(episode.read_robot(robot), robot)
        self.assertEqual(episode.read_objects("objects", "ids"), ("objects", "ids"))
        stats = self.profiler.snapshot()["timings"]
        self.assertEqual(stats["contacts.read"]["count"], 2)
        self.assertEqual(stats["contacts.cache_hit"]["count"], 1)
        self.assertEqual(stats["robot.write"]["count"], 1)
        self.assertEqual(stats["robot.update"]["count"], 1)
        self.assertEqual(stats["physics.step"]["count"], 1)
        self.assertEqual(calls, [("physics", {"render": False})])
        self.assertEqual((robot.writes, robot.updates, contacts.reads), (1, 1, 2))
        self.profiler.restore()
        self.assertIs(episode.read_robot, original_reader)
        self.assertNotIn("observe", vars(contacts))
        self.assertNotIn("update", vars(robot))


if __name__ == "__main__":
    unittest.main()
