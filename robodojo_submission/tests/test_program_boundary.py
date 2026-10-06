import tempfile
import time
import unittest
from pathlib import Path

from manipisa_robodojo.programs import (
    execute_program, public_builtins, restricted_math, restricted_np,
)


class NumericalBoundaryTests(unittest.TestCase):
    def namespace(self):
        return {"__builtins__": public_builtins(), "np": restricted_np(),
                "math": restricted_math()}

    def test_pose_math_and_persistent_values_work(self):
        namespace = self.namespace()
        execute_program("target=np.array([0.3, 0.4, 0.0]); distance=np.linalg.norm(target)",
                        namespace, time.perf_counter() + 3)
        output = execute_program("print(distance, math.isclose(distance, 0.5))", namespace,
                                 time.perf_counter() + 3)
        self.assertEqual(output.strip(), "0.5 True")

    def test_file_and_private_introspection_routes_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evaluator-secret.txt"
            target.write_text("hidden score and material state")
            programs = [
                f"print(np.loadtxt({str(target)!r}, dtype=str))",
                f"print(np.fromfile({str(target)!r}))",
                f"np.array([1]).tofile({str(target)!r})",
                f"np.array([1]).dump({str(target)!r})",
                "print(np.array([1]).ctypes)",
                "print('{x.__dict__}'.format(x=np))",
                "print('{x.__dict__}'.format_map({'x': np}))",
                "import os",
                "print(np.__dict__)",
                "print(np.lib)",
                "print(open('anything'))",
            ]
            for code in programs:
                with self.subTest(code=code), self.assertRaises((ValueError, AttributeError, NameError)):
                    execute_program(code, self.namespace(), time.perf_counter() + 3)
                self.assertEqual(target.read_text(), "hidden score and material state")

    def test_failed_validation_preserves_capture_contract(self):
        capture = {}
        with self.assertRaises(ValueError):
            execute_program("import os", self.namespace(), time.perf_counter() + 3,
                            capture=capture)
        self.assertEqual(capture, {"stdout": ""})


if __name__ == "__main__":
    unittest.main()
