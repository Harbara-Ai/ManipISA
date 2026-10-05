"""Optional, bounded host-side timings for simulation diagnosis.

These are perf_counter durations, not GPU kernel timings. Existing GPU-to-CPU
synchronization may make a timed call wait for earlier GPU work. No additional
CUDA synchronization, observation, or physics step is introduced here.

Inclusive timings overlap their children; exclusive timings subtract measured
child spans in the same thread. Neither may be added to unrelated agent/network
timings to obtain an episode wall-time breakdown.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from threading import RLock, local
from time import perf_counter
from typing import Callable


@dataclass
class _Frame:
    label: str
    started: float
    children: float = 0.0


@dataclass
class _Binding:
    obj: object
    name: str
    wrapper: Callable
    had_instance_value: bool
    instance_value: object


class TimingProfiler:
    """Aggregate fixed labels without retaining one record per call or step.

    ``wrap`` accepts a label string or a classifier called with the original
    call arguments, before the wrapped operation. Classifiers must be read-only
    and return a fixed label vocabulary. The label count is capped as a final
    safeguard; additional labels share ``profiling.other``.
    """

    def __init__(self, enabled=True, *, max_labels=128):
        if type(max_labels) is not int or max_labels < 1:
            raise ValueError("max_labels must be a positive integer")
        self.enabled = bool(enabled)
        self.max_labels = max_labels
        self._clock = perf_counter
        self._local = local()
        self._lock = RLock()
        self._stats = {}
        self._bindings = {}
        self._active = 0
        self._root_seconds = 0.0

    def span(self, label):
        """Time a block, or return a no-op without reading the clock if disabled."""
        if not self.enabled:
            return nullcontext()
        return self._measure(label)

    @contextmanager
    def _measure(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError("Timing labels must be nonempty strings")
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        with self._lock:
            if label not in self._stats:
                if len(self._stats) >= self.max_labels:
                    label = "profiling.other"
                self._stats.setdefault(label, {
                    "count": 0, "inclusive_seconds": 0.0,
                    "exclusive_seconds": 0.0, "max_seconds": 0.0,
                    "failures": 0,
                })
            self._active += 1
        frame = _Frame(label, self._clock())
        stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = max(0.0, self._clock() - frame.started)
            stack.pop()
            if stack:
                stack[-1].children += elapsed
            with self._lock:
                stats = self._stats[frame.label]
                stats["count"] += 1
                stats["inclusive_seconds"] += elapsed
                stats["exclusive_seconds"] += max(0.0, elapsed - frame.children)
                stats["max_seconds"] = max(stats["max_seconds"], elapsed)
                stats["failures"] += int(failed)
                self._active -= 1
                if not stack:
                    self._root_seconds += elapsed

    def wrap(self, obj, method, label):
        """Instrument one instance method/function attribute; return whether bound.

        Disabled, absent, read-only, non-callable, and already profiled bindings
        return False. Inherited descriptors are restored by deleting the
        instance override; existing plain function attributes are restored as
        the exact original object. Calls and arguments are forwarded unchanged.
        """
        if not self.enabled or obj is None:
            return False
        key = (id(obj), method)
        if key in self._bindings:
            return False
        original = getattr(obj, method, None)
        if not callable(original) or getattr(original, "_timing_profiler_wrapper", False):
            return False
        instance_dict = getattr(obj, "__dict__", {})
        had_instance_value = method in instance_dict
        instance_value = instance_dict.get(method)
        # Slot-backed callable attributes have no __dict__ entry, but must be
        # restored by assignment rather than removing their populated slot.
        for cls in type(obj).__mro__:
            descriptor = vars(cls).get(method)
            if descriptor is not None and hasattr(descriptor, "__set__"):
                had_instance_value, instance_value = True, original
                break

        @wraps(original)
        def measured(*args, **kwargs):
            selected = label(*args, **kwargs) if callable(label) else label
            with self.span(selected):
                return original(*args, **kwargs)

        measured._timing_profiler_wrapper = True
        try:
            setattr(obj, method, measured)
        except (AttributeError, TypeError):
            return False
        self._bindings[key] = _Binding(obj, method, measured, had_instance_value, instance_value)
        return True

    def restore(self):
        """Remove owned wrappers, preserving any later external replacement."""
        for binding in reversed(tuple(self._bindings.values())):
            if getattr(binding.obj, binding.name, None) is not binding.wrapper:
                continue
            if binding.had_instance_value:
                setattr(binding.obj, binding.name, binding.instance_value)
            else:
                delattr(binding.obj, binding.name)
        self._bindings.clear()

    def reset(self):
        """Clear aggregates while retaining wrappers; reject an in-flight reset."""
        with self._lock:
            if self._active:
                raise RuntimeError("Cannot reset timings while a span is active")
            self._stats.clear()
            self._root_seconds = 0.0

    def snapshot(self):
        """Return detached JSON-serializable aggregates, including open-span count."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "clock": "time.perf_counter",
                "scope": "host durations; no added CUDA synchronization",
                "limitations": (
                    "Inclusive spans overlap. Exclusive spans subtract measured children "
                    "in the same thread. Existing synchronization may charge earlier GPU "
                    "work to the waiting host call; these are not GPU kernel durations. "
                    "Uninstrumented work and open spans are not included."
                ),
                "active_spans": self._active,
                "root_seconds": self._root_seconds,
                "timings": {label: dict(values) for label, values in sorted(self._stats.items())},
            }


def attach_episode_profiler(episode, profiler):
    """Bind diagnosis points after episode initialization without doing any work.

    Optional or read-only methods are skipped. Shared articulations are bound
    once, including the episode robot used by DIRECT. Contact cache hits are
    classified using the existing timestamp; classification never reads PhysX.
    """
    if not profiler.enabled:
        return profiler
    runtime = getattr(episode, "runtime", None)
    adapter = getattr(episode, "adapter", None)
    for method, label in (("step", "runtime.step"), ("_assess", "runtime.assess"),
                          ("_prepare", "runtime.prepare")):
        profiler.wrap(runtime, method, label)
    for method in ("observe", "command", "hold", "step"):
        profiler.wrap(adapter, method, f"adapter.{method}")
    profiler.wrap(getattr(episode, "sim", None), "step", "physics.step")

    source = getattr(adapter, "contact_source", None)
    if source is not None:
        def contact_label(*args, **kwargs):
            timestamp = args[0] if args else kwargs.get("timestamp")
            return ("contacts.cache_hit" if timestamp == getattr(source, "_timestamp", None)
                    else "contacts.read")
        profiler.wrap(source, "observe", contact_label)

    robots = {id(robot): robot for robot in getattr(adapter, "_robots", {}).values()}
    robot = getattr(episode, "robot", None)
    if robot is not None:
        robots[id(robot)] = robot
    for robot in robots.values():
        profiler.wrap(robot, "write_data_to_sim", "robot.write")
        profiler.wrap(robot, "update", "robot.update")
    profiler.wrap(getattr(episode, "tracker", None), "update", "evaluator.update")
    for method in ("read_objects", "read_robot"):
        profiler.wrap(episode, method, f"evaluator.{method}")
    return profiler
