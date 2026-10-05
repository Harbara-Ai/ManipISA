"""Structured PhysX contacts without TacMap/tactile images.

One exact sensor rigid body / exact target rigid body per binding. Contact reports
must be enabled before simulation reset. Read raw normal AND friction patches:
average contact positions cannot recover the interaction moment.
"""
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from manipisa.types import ContactState, ContactPoint


@dataclass(frozen=True)
class ContactBinding:
    actor: str
    target: str
    sensor_path: str
    target_path: str
    # Must compute a conservative surface gap from current collider geometry.
    # Returns (gap metres, observation timestamp); no force-zero inference.
    separation: Callable[[float], tuple[float, float]] | None = None
    # Calibrated conservative lower bound, supplied by the scene/robot adapter.
    # The Agent cannot increase friction simply by asking for a stronger grasp.
    friction_lower_bound: float | None = None


class PhysXContactSource:
    def __init__(self, sim, bindings: dict[str, ContactBinding], *, max_contact_points=512):
        import omni.physics.tensors as tensors
        self.bindings = dict(bindings)
        self.dt = float(sim.get_physics_dt())
        self._simulation_view = tensors.create_simulation_view("torch")
        self._simulation_view.set_subspace_roots("/")
        self._views, self._cache, self._timestamp = {}, {}, None
        self.capacity = max_contact_points
        for name, binding in self.bindings.items():
            if binding.friction_lower_bound is not None and (not math.isfinite(binding.friction_lower_bound) or binding.friction_lower_bound <= 0):
                raise ValueError("Friction calibration must be finite and positive")
            if any(c in binding.sensor_path+binding.target_path for c in "*?[]") or binding.sensor_path == binding.target_path:
                raise ValueError("Contact bindings require distinct exact rigid-body paths")
            view = self._simulation_view.create_rigid_contact_view(binding.sensor_path,
                filter_patterns=[binding.target_path], max_contact_data_count=max_contact_points)
            if view.sensor_count != 1 or view.filter_count != 1:
                raise ValueError(f"Contact pair {name} did not resolve to exactly one sensor/filter")
            self._views[name] = view

    def pair_key(self, name):
        b = self.bindings[name]
        return "|".join(sorted((b.sensor_path, b.target_path)))

    def target_key(self, name):
        return self.bindings[name].target_path

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.array(value, copy=True)

    @classmethod
    def _numpy_slice(cls, value, indices, *, width=None):
        """Copy only the valid patch range; empty ranges need no device read.

        PhysX supplies contiguous (capacity, 1)/(capacity, 3) float buffers,
        so these reshapes and basic slices are views before the CPU transfer.
        Keep _numpy's final copy for CPU tensors backed by reusable buffers.
        """
        if not hasattr(value, "reshape"):
            value = np.asarray(value)
        shaped = value.reshape(-1) if width is None else value.reshape(-1, width)
        selected = shaped[indices]
        if selected.shape[0] == 0:
            # dtype/shape are host metadata, including for CUDA tensors.
            dtype = np.dtype(str(selected.dtype).removeprefix("torch."))
            return np.empty(tuple(selected.shape), dtype=dtype)
        return cls._numpy(selected)

    def _buffer(self, count, start):
        count, start = int(self._numpy(count).reshape(-1)[0]), int(self._numpy(start).reshape(-1)[0])
        if count < 0 or count >= self.capacity or start < 0 or start+count > self.capacity:
            raise ValueError("Contact buffer capacity reached or invalid indices")
        return slice(start, start+count)

    def observe(self, timestamp):
        if timestamp == self._timestamp:
            return dict(self._cache)
        result = {}
        for name, binding in self.bindings.items():
            try:
                view = self._views[name]
                force, points, normals, distances, count, start = view.get_contact_data(dt=self.dt)
                indices = self._buffer(count, start)
                f = self._numpy_slice(force, indices)
                p = self._numpy_slice(points, indices, width=3)
                n = self._numpy_slice(normals, indices, width=3)
                d = self._numpy_slice(distances, indices)
                if not np.isfinite(np.r_[f,p.ravel(),n.ravel(),d]).all() or (f < -1e-6).any():
                    raise ValueError("Invalid raw contact points/forces")
                normal_vectors = f[:,None] * n
                aggregate = self._numpy(view.get_contact_force_matrix(dt=self.dt)).reshape(-1,3)[0]
                if not np.allclose(normal_vectors.sum(axis=0), aggregate, atol=1e-3, rtol=1e-3):
                    raise ValueError("Raw points disagree with aggregate normal force (possible truncation)")
                on_target = -normal_vectors
                total_force = on_target.sum(axis=0)
                moment = np.cross(p, on_target).sum(axis=0)
                wrench_valid, detail = True, ""
                try:
                    friction, fp, fc, fs = view.get_friction_data(dt=self.dt)
                    fi = self._buffer(fc, fs)
                    tangential = -self._numpy_slice(friction, fi, width=3)
                    positions = self._numpy_slice(fp, fi, width=3)
                    if not np.isfinite(np.r_[tangential.ravel(),positions.ravel()]).all():
                        raise ValueError("Invalid friction patches")
                    total_force += tangential.sum(axis=0)
                    moment += np.cross(positions, tangential).sum(axis=0)
                except Exception as exc:
                    wrench_valid, detail = False, f"Full wrench unavailable: {exc}"
                gap = None
                if binding.separation is not None:
                    value, stamp = binding.separation(timestamp)
                    if math.isfinite(value) and math.isfinite(stamp) and -1e-9 <= timestamp-stamp <= self.dt+1e-9:
                        gap = float(value)
                present = bool((d <= 1e-5).any() or f.sum() > 1e-6)
                result[name] = ContactState(binding.actor, binding.target, timestamp, present, float(f.sum()),
                    tuple(total_force), tuple(moment), tuple(ContactPoint(tuple(pi), tuple(-ni), float(fi)) for pi,ni,fi in zip(p,n,f)),
                    gap, True, wrench_valid, "PhysX.raw_contact_and_friction_patches", detail)
            except Exception as exc:
                result[name] = ContactState(binding.actor, binding.target, timestamp, False, 0., (0.,0.,0.), (0.,0.,0.),
                                            valid=False, source="PhysX.raw_contact", detail=str(exc))
        self._cache, self._timestamp = result, timestamp
        return dict(result)


def sphere_sphere_separation(first_position, first_radius, second_position, second_radius):
    """Binding helper for actual spherical colliders; positions must be read live."""
    def observe(timestamp):
        a, b = np.asarray(first_position()), np.asarray(second_position())
        return float(np.linalg.norm(a-b)-first_radius-second_radius), timestamp
    return observe


def axis_aligned_box_separation(first_bounds, second_bounds):
    """Conservative gap from current *collider-enclosing* world AABBs.

    Positive AABB gap proves separation; overlapping boxes do not prove contact.
    Callbacks must return actual current lower/upper bounds, not visual bounds.
    """
    def observe(timestamp):
        al, ah = map(np.asarray, first_bounds())
        bl, bh = map(np.asarray, second_bounds())
        gap = np.maximum(bl-ah, al-bh)
        return float(np.linalg.norm(np.maximum(gap, 0))) if (gap > 0).any() else float(np.max(gap)), timestamp
    return observe
