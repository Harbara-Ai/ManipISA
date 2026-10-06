"""Strict public observation boundary around the unchanged ManipISA adapter.

The official evaluator controls the original dual X5 scene and IK. In this
protocol tool_a/tool_b are right/left link6 poses, not fingertip TCPs. No extra
grasp offset is applied. Gripper values remain control targets, not evidence of
contact or measured opening. This validator is not a Python security sandbox.
"""
from __future__ import annotations

import math

import numpy as np

from manipisa.adapters.robodojo import (
    ACTORS, CAMERAS, ROBODOJO_REVISION, SIDES, XPOLICYLAB_REVISION,
    RoboDojoAdapter as _CoreAdapter,
    public_observation as _core_public_observation,
)


_REAL_TYPES = frozenset((int, float, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64,
                         np.float16, np.float32, np.float64, np.longdouble))


def _mapping(value, name):
    # A decoded protocol dictionary needs no custom lookup/iteration hooks.
    if type(value) is not dict:
        raise ValueError(f"{name}: expected a plain protocol dictionary")
    return value


def _field(value, key, name):
    if key not in value:
        raise ValueError(f"Missing public observation field: {name}")
    return value[key]


def _vector(value, size, name):
    if type(value) is np.ndarray:
        if value.shape != (size,) or value.dtype.kind not in "fiu":
            raise ValueError(f"{name}: expected {size} real numeric coordinates")
        # Dtype metadata can contain arbitrary nested objects. Strip it before
        # the core validator copies numeric storage into an owned output array.
        return value.view(np.dtype(value.dtype.str))
    if type(value) not in (list, tuple) or len(value) != size:
        raise ValueError(f"{name}: expected {size} real numeric coordinates")
    if any(type(item) not in _REAL_TYPES for item in value):
        raise ValueError(f"{name}: expected real numeric coordinates")
    return value


def public_observation(observation):
    """Return only copied public RGB/robot/instruction fields from decoded data.

    Unknown fields are never traversed, serialized or deep-copied, including
    nested friction, object poses, contact readings and evaluator labels. The
    core validator still owns finite-value, quaternion and gripper-range checks.
    """
    obs = _mapping(observation, "observation")
    version = _field(obs, "data_format_version", "data_format_version")
    if type(version) is not str or version != "v1.0":
        raise ValueError("Expected RoboDojo data_format_version v1.0")
    env_idx = _field(obs, "env_idx", "env_idx")
    if type(env_idx) is not int or env_idx != 0:
        raise ValueError("This integration requires one environment, env_idx=0")
    info = _mapping(_field(obs, "additional_info", "additional_info"), "additional_info")
    frequency = _field(info, "frequency", "additional_info.frequency")
    if type(frequency) not in _REAL_TYPES:
        raise ValueError("Observation frequency must be a real number")
    try:
        frequency = float(frequency)
    except OverflowError as exc:
        raise ValueError("Invalid observation frequency") from exc
    if not math.isfinite(frequency) or frequency <= 0 or not math.isfinite(1.0 / frequency):
        raise ValueError("Invalid observation frequency")
    instruction = _field(obs, "instruction", "instruction")
    if type(instruction) is not str or not instruction.strip():
        raise ValueError("Expected a nonempty plain language instruction")

    source_state = _mapping(_field(obs, "state", "state"), "state")
    state = {}
    for side in SIDES:
        for suffix, size in (("arm_joint_state", 6), ("ee_pose", 7), ("ee_joint_state", 1)):
            key = f"{side}_{suffix}"
            state[key] = _vector(_field(source_state, key, f"state.{key}"), size, key)
    source_vision = _mapping(_field(obs, "vision", "vision"), "vision")
    vision = {}
    for name in CAMERAS:
        camera = _mapping(_field(source_vision, name, f"vision.{name}"), name)
        pixels = _field(camera, "color", f"vision.{name}.color")
        if type(pixels) is not np.ndarray or pixels.dtype != np.dtype("uint8"):
            raise ValueError(f"{name}: expected decoded uint8 RGB pixels")
        vision[name] = {"color": pixels.view(np.uint8)}
    clean = {"data_format_version": version, "env_idx": env_idx,
             "additional_info": {"frequency": frequency}, "instruction": instruction,
             "state": state, "vision": vision}
    try:
        return _core_public_observation(clean)
    except (OverflowError, TypeError) as exc:
        raise ValueError("Invalid numeric public observation") from exc


class RoboDojoAdapter(_CoreAdapter):
    """Reuse core motion contracts with the same boundary at every observation."""

    def __init__(self, initial_observation, exchange, **kwargs):
        if not callable(exchange):
            raise TypeError("exchange must be callable")

        def public_exchange(action):
            return public_observation(exchange(action))

        super().__init__(public_observation(initial_observation), public_exchange, **kwargs)


__all__ = ["ACTORS", "CAMERAS", "SIDES", "ROBODOJO_REVISION", "XPOLICYLAB_REVISION",
           "RoboDojoAdapter", "public_observation"]
