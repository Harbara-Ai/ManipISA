"""Numerical tools for the public-observation submission.

This reduces the research tool's API surface; it is not an OS security sandbox.
No simulator, evaluator, filesystem or network object is given to agent code.
"""
from __future__ import annotations

import ast
import math
from types import SimpleNamespace

import numpy as np

from manipisa.evaluation.programs import (
    EpisodeStopped, execute_program as _execute_program, public_builtins,
)


NUMPY_FUNCTIONS = (
    "array", "asarray", "zeros", "ones", "full", "eye", "arange", "linspace",
    "zeros_like", "ones_like", "copy", "concatenate", "stack", "hstack", "vstack",
    "reshape", "transpose", "squeeze", "expand_dims", "clip", "where",
    "abs", "sqrt", "square", "sin", "cos", "tan", "arcsin", "arccos", "arctan2",
    "deg2rad", "rad2deg", "sign", "floor", "ceil", "round", "minimum", "maximum",
    "sum", "mean", "min", "max", "argmin", "argmax", "dot", "cross", "outer",
    "isfinite", "isnan", "all", "any", "allclose", "isclose", "array_equal",
    "float32", "float64", "int32", "int64", "uint8",
)
MATH_FUNCTIONS = (
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot",
    "degrees", "radians", "floor", "ceil", "trunc", "fabs", "copysign",
    "isfinite", "isnan", "isclose", "exp", "log", "log2", "log10", "pow",
)
# ndarray methods and string interpolation can bypass a module-only allowlist.
# format/format_map can traverse __private__ attributes inside literal strings.
BLOCKED_ATTRIBUTES = frozenset({
    "tofile", "dump", "dumps", "ctypes", "format", "format_map",
    "load", "loads", "loadtxt", "genfromtxt", "fromfile", "save", "savez",
    "savez_compressed", "savetxt", "memmap", "open_memmap",
})


def restricted_np():
    values = {name: getattr(np, name) for name in NUMPY_FUNCTIONS}
    values.update(pi=float(np.pi), inf=float(np.inf), nan=float(np.nan))
    values["linalg"] = SimpleNamespace(**{
        name: getattr(np.linalg, name) for name in ("norm", "inv", "solve", "det", "svd")
    })
    return SimpleNamespace(**values)


def restricted_math():
    return SimpleNamespace(**{name: getattr(math, name)
                              for name in (*MATH_FUNCTIONS, "pi", "e", "tau", "inf", "nan")})


def validate_program(code):
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    if len(code) > 100_000:
        raise ValueError("Program exceeds 100000 characters")
    tree = ast.parse(code, filename="<agent-program>")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are outside the submission API")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are outside the submission API")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES:
                raise ValueError(f"Attribute {node.attr!r} is outside the submission API")


def execute_program(code, namespace, deadline, *, capture=None):
    if capture is not None:
        capture["stdout"] = ""
    validate_program(code)
    return _execute_program(code, namespace, deadline, capture=capture)
