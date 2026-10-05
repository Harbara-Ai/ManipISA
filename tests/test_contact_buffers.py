"""PhysX buffer decoding regression tests; no Isaac app or GPU is required."""
import unittest

import numpy as np

from manipisa.adapters.physx_contacts import ContactBinding, PhysXContactSource


class TensorView:
    """Torch-like buffer view that records exactly what reaches .cpu()."""
    def __init__(self, data, name, transfers):
        self.data = data
        self.name = name
        self.transfers = transfers

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return "torch." + str(self.data.dtype)

    def reshape(self, *shape):
        return TensorView(self.data.reshape(*shape), self.name, self.transfers)

    def __getitem__(self, key):
        return TensorView(self.data[key], self.name, self.transfers)

    def detach(self):
        return self

    def cpu(self):
        self.transfers.append((self.name, self.shape))
        return self

    def numpy(self):
        # Deliberately alias the backing array, as a real CPU tensor does.
        return self.data


class PaddedView:
    def __init__(self, *, tensors=True, normal_count=1, normal_start=2,
                 friction_count=1, friction_start=5):
        self.capacity = 8
        self.tensors = tensors
        self.transfers = []
        self.calls = []
        self.normal_count, self.normal_start = normal_count, normal_start
        self.friction_count, self.friction_start = friction_count, friction_start
        self.friction_error = None
        self.arrays = {name: np.full((self.capacity, width), np.nan, dtype=np.float32)
                       for name, width in (("force", 1), ("points", 3), ("normals", 3),
                                           ("distances", 1), ("friction", 3), ("fp", 3))}
        # Data outside each valid range is intentionally unusable.
        if 0 <= normal_start < self.capacity and normal_count > 0:
            self.arrays["force"][normal_start] = 2.
            self.arrays["points"][normal_start] = (0., 1., 0.)
            self.arrays["normals"][normal_start] = (-1., 0., 0.)
            self.arrays["distances"][normal_start] = -.001
        if 0 <= friction_start < self.capacity and friction_count > 0:
            self.arrays["friction"][friction_start] = (0., -3., 0.)
            self.arrays["fp"][friction_start] = (1., 0., 0.)
        self.aggregate = np.array([[[-2., 0., 0.]]] if normal_count else [[[0., 0., 0.]]], dtype=np.float32)
        # Normal/friction getters share these buffers in the installed PhysX API.
        self.count = np.zeros((1, 1), dtype=np.int32)
        self.start = np.zeros((1, 1), dtype=np.int32)

    def wrap(self, data, name):
        return TensorView(data, name, self.transfers) if self.tensors else data

    def get_contact_data(self, dt):
        self.calls.append("normal")
        self.count[:] = self.normal_count
        self.start[:] = self.normal_start
        return (*[self.wrap(self.arrays[k], k) for k in ("force", "points", "normals", "distances")],
                self.wrap(self.count, "count"), self.wrap(self.start, "start"))

    def get_contact_force_matrix(self, dt):
        self.calls.append("aggregate")
        return self.wrap(self.aggregate, "aggregate")

    def get_friction_data(self, dt):
        self.calls.append("friction")
        if self.friction_error is not None:
            raise RuntimeError(self.friction_error)
        self.count[:] = self.friction_count
        self.start[:] = self.friction_start
        return (self.wrap(self.arrays["friction"], "friction"), self.wrap(self.arrays["fp"], "fp"),
                self.wrap(self.count, "count"), self.wrap(self.start, "start"))


class ContactBufferTests(unittest.TestCase):
    def source(self, view):
        source = PhysXContactSource.__new__(PhysXContactSource)
        source.bindings = {"pair": ContactBinding("hand", "ball", "/sensor", "/ball")}
        source.dt = .01
        source.capacity = view.capacity
        source._timestamp, source._cache = None, {}
        source._views = {"pair": view}
        return source

    def test_padding_and_nonzero_offsets_preserve_force_moment_and_points(self):
        for tensors in (False, True):
            with self.subTest(tensors=tensors):
                view = PaddedView(tensors=tensors)
                contact = self.source(view).observe(.1)["pair"]
                self.assertTrue(contact.valid)
                self.assertTrue(contact.wrench_valid)
                self.assertTrue(contact.present)
                self.assertEqual(contact.normal_force, 2.)
                np.testing.assert_allclose(contact.force_on_target_w, (2., 3., 0.))
                np.testing.assert_allclose(contact.torque_on_target_world_origin_w, (0., 0., 1.))
                self.assertEqual(len(contact.points), 1)
                self.assertEqual(contact.points[0].position_w, (0., 1., 0.))
                self.assertEqual(contact.points[0].normal_on_target_w, (1., 0., 0.))
                self.assertEqual(view.calls, ["normal", "aggregate", "friction"])

    def test_tensor_transfers_only_include_valid_patch_slices(self):
        view = PaddedView()
        self.source(view).observe(.1)
        actual = {name: shape for name, shape in view.transfers}
        for name in ("force", "distances"):
            self.assertEqual(actual[name], (1,))
        for name in ("points", "normals", "friction", "fp"):
            self.assertEqual(actual[name], (1, 3))
        self.assertEqual(len(view.transfers), 11)

    def test_empty_slices_skip_only_raw_patch_transfers(self):
        view = PaddedView(normal_count=0, normal_start=8, friction_count=0, friction_start=8)
        contact = self.source(view).observe(.1)["pair"]
        self.assertTrue(contact.valid)
        self.assertTrue(contact.wrench_valid)
        self.assertFalse(contact.present)
        self.assertEqual(contact.points, ())
        self.assertEqual(contact.force_on_target_w, (0., 0., 0.))
        self.assertEqual(view.calls, ["normal", "aggregate", "friction"])
        self.assertEqual([name for name, _ in view.transfers], ["count", "start", "aggregate", "count", "start"])

    def test_empty_normal_keeps_aggregate_inconsistency_detection(self):
        view = PaddedView(normal_count=0, friction_count=0)
        view.aggregate[0, 0, 0] = 1.
        contact = self.source(view).observe(.1)["pair"]
        self.assertFalse(contact.valid)
        self.assertIn("disagree with aggregate", contact.detail)

    def test_empty_normal_keeps_friction_failure_unknown(self):
        view = PaddedView(normal_count=0, friction_count=0)
        view.friction_error = "not available"
        contact = self.source(view).observe(.1)["pair"]
        self.assertTrue(contact.valid)
        self.assertFalse(contact.wrench_valid)
        self.assertIn("not available", contact.detail)
        self.assertEqual(view.calls, ["normal", "aggregate", "friction"])

    def test_friction_can_remain_nonempty_when_normal_is_empty(self):
        contact = self.source(PaddedView(normal_count=0)).observe(.1)["pair"]
        self.assertTrue(contact.valid)
        self.assertTrue(contact.wrench_valid)
        np.testing.assert_allclose(contact.force_on_target_w, (0., 3., 0.))
        np.testing.assert_allclose(contact.torque_on_target_world_origin_w, (0., 0., 3.))

    def test_shared_indices_are_consumed_before_friction_overwrites_them(self):
        view = PaddedView(normal_start=2, friction_start=5)
        contact = self.source(view).observe(.1)["pair"]
        self.assertEqual(view.start[0, 0], 5)
        self.assertTrue(contact.valid)
        self.assertTrue(contact.wrench_valid)
        self.assertEqual(contact.points[0].position_w, (0., 1., 0.))
        np.testing.assert_allclose(contact.torque_on_target_world_origin_w, (0., 0., 1.))

    def test_original_capacity_rejections_are_preserved(self):
        for count, start in ((-1, 0), (8, 0), (1, -1), (2, 7), (0, 9)):
            with self.subTest(count=count, start=start):
                view = PaddedView(normal_count=count, normal_start=start)
                contact = self.source(view).observe(.1)["pair"]
                self.assertFalse(contact.valid)
                self.assertIn("capacity reached or invalid indices", contact.detail)
        view = PaddedView(friction_count=8, friction_start=0)
        contact = self.source(view).observe(.1)["pair"]
        self.assertTrue(contact.valid)
        self.assertFalse(contact.wrench_valid)
        self.assertIn("capacity reached or invalid indices", contact.detail)

    def test_copied_cpu_slices_do_not_alias_reusable_buffers(self):
        for tensors in (False, True):
            with self.subTest(tensors=tensors):
                view = PaddedView(tensors=tensors)
                value = view.wrap(view.arrays["points"], "points")
                copied = PhysXContactSource._numpy_slice(value, slice(2, 3), width=3)
                view.arrays["points"][:] = 99.
                np.testing.assert_array_equal(copied, [[0., 1., 0.]])

    def test_empty_dtype_and_shape_are_preserved_without_tensor_transfer(self):
        for dtype in (np.float32, np.float64):
            transfers = []
            value = TensorView(np.zeros((8, 3), dtype=dtype), "points", transfers)
            result = PhysXContactSource._numpy_slice(value, slice(8, 8), width=3)
            self.assertEqual(result.shape, (0, 3))
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(transfers, [])

    def test_negative_and_nonfinite_active_forces_remain_invalid(self):
        for value in (-.001, np.nan, np.inf):
            with self.subTest(value=value):
                view = PaddedView()
                view.arrays["force"][2] = value
                contact = self.source(view).observe(.1)["pair"]
                self.assertFalse(contact.valid)
                self.assertIn("Invalid raw contact", contact.detail)

    def test_same_timestamp_cache_does_not_repeat_getters(self):
        view = PaddedView()
        source = self.source(view)
        first = source.observe(.1)
        calls = list(view.calls)
        self.assertEqual(source.observe(.1), first)
        self.assertEqual(view.calls, calls)
        source.observe(.2)
        self.assertEqual(len(view.calls), 2 * len(calls))


if __name__ == "__main__":
    unittest.main()
