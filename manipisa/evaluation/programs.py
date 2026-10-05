"""Common bounded Python tool for the two experiment interfaces.

This enforces the experiment's public API, not an OS security sandbox. The
runner is for trusted research model outputs in simulation only.
"""
import ast
from contextlib import redirect_stdout
import io
import sys
import time


class EpisodeStopped(BaseException):
    pass


def execute_program(code, namespace, deadline, *, capture=None):
    # The benchmark response layer needs full stdout even if execution raises
    # after advancing physics. Keep the existing return/exception API intact.
    if capture is not None:
        capture["stdout"] = ""
    tree = ast.parse(code, filename="<agent-program>")
    if len(code) > 100_000:
        raise ValueError("Program exceeds 100000 characters")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Use the preloaded np, torch, math, and documented controller/types; imports are disabled")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are outside the experiment API")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are outside the experiment API")
    def check(frame, event, arg):
        # Trace agent-authored loops, not every line in Isaac/torch/metrics.
        # The host's step() separately checks the deadline each physics step.
        if frame.f_code.co_filename != "<agent-program>":
            return None
        if time.perf_counter() >= deadline:
            raise EpisodeStopped("wall_timeout")
        return check
    output = io.StringIO()
    previous = sys.gettrace()
    try:
        sys.settrace(check)
        with redirect_stdout(output):
            exec(compile(tree, "<agent-program>", "exec"), namespace)
    finally:
        sys.settrace(previous)
        if capture is not None:
            capture["stdout"] = output.getvalue()
    return output.getvalue()[-30000:]


def public_builtins():
    names = ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
             "len", "list", "map", "max", "min", "print", "range", "reversed", "round",
             "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError")
    import builtins
    return {name: getattr(builtins, name) for name in names}
