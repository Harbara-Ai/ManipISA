"""Optional RGB return channel. No tactile processing or task-evaluator input."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class RGBFrame:
    camera_id: str
    timestamp: float
    rgb: np.ndarray | None
    valid: bool
    error: str | None = None
    intrinsic: list | None = None
    world_from_camera: list | None = None
    camera_model: str = "pinhole"
    distortion_coefficients: list | None = None
    # Bench2Dex CameraFrame extrinsics use its optical camera convention.
    frame_convention: str = "Bench2Dex.CameraFrame"


class Bench2DexRGBSource:
    """Wrap an initialized CameraRig. The caller owns its creation and cleanup."""
    def __init__(self, sim, camera_rig):
        self.sim, self.rig = sim, camera_rig
        self.camera_ids = tuple(camera_rig.camera_ids)

    def capture(self, dt):
        # Match run_policy: synchronize mounted cameras, render, then extract.
        self.rig.sync_mounted_camera_poses()
        self.sim.render()
        return self.rig.capture(dt)


class RGBFeedback:
    def __init__(self, source, *, enabled=False, period_s=0.10, max_age_s=0.25):
        if not all(math.isfinite(x) and x > 0 for x in (period_s, max_age_s)):
            raise ValueError("RGB periods must be finite and positive")
        self.source, self.enabled = source, bool(enabled)
        self.period_s, self.max_age_s = period_s, max_age_s
        self._last_capture = None
        self._frames: dict[str, RGBFrame] = {}
        self.capture_count = 0
        ids = tuple(source.camera_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Camera IDs must be nonempty and unique")

    def set_enabled(self, enabled):
        if bool(enabled) != self.enabled:
            self.enabled = bool(enabled)
            self._last_capture = None
            self._frames.clear()

    def sample(self, timestamp):
        if not self.enabled:
            return
        if not math.isfinite(timestamp):
            raise ValueError("Invalid RGB sample time")
        if self._last_capture is not None and timestamp < self._last_capture:
            self._frames.clear()
            self._last_capture = None
        if self._last_capture is not None and timestamp - self._last_capture < self.period_s - 1e-9:
            return
        dt = self.period_s if self._last_capture is None else timestamp - self._last_capture
        self._last_capture = timestamp
        self.capture_count += 1
        try:
            raw = self.source.capture(dt)
            capture_error = None
        except Exception as exc:
            raw, capture_error = {}, str(exc)
        frames = {}
        for camera_id in self.source.camera_ids:
            try:
                frame = raw[camera_id]
                image = np.asarray(frame.rgb)
                if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0 or image.dtype != np.uint8:
                    raise ValueError("Expected nonempty HWC uint8 RGB")
                image = image.copy()
                image.setflags(write=False)
                def calibration(name, shape=None):
                    value = getattr(frame, name, None)
                    if value is None:
                        return None
                    value = np.asarray(value)
                    if not np.isfinite(value).all() or (shape and value.shape != shape):
                        raise ValueError(f"Invalid camera calibration: {name}")
                    return value.tolist()
                frames[camera_id] = RGBFrame(camera_id, timestamp, image, True,
                    intrinsic=calibration("intrinsic", (3, 3)),
                    world_from_camera=calibration("extrinsic_world_from_cam", (4, 4)),
                    camera_model=getattr(frame, "camera_model", "pinhole"),
                    distortion_coefficients=calibration("distortion_coefficients"))
            except Exception as exc:
                frames[camera_id] = RGBFrame(camera_id, timestamp, None, False,
                                             capture_error or f"Missing or invalid frame: {exc}")
        self._frames = frames

    def frames(self, now=None):
        """Raw image attachments; timestamps and validity must travel with pixels."""
        if not self.enabled:
            return {}
        if now is None:
            now = self._last_capture
        return {key: (frame if -1e-9 <= now - frame.timestamp <= self.max_age_s else
                      replace(frame, valid=False, error="Stale or future-dated frame"))
                for key, frame in self._frames.items()}

    def metadata(self, now):
        result = {}
        if self.enabled:
            for camera_id in self.source.camera_ids:
                frame = self._frames.get(camera_id)
                if frame is None:
                    result[camera_id] = {"valid": False, "error": "No frame captured", "timestamp": None}
                    continue
                age = now - frame.timestamp
                fresh = -1e-9 <= age <= self.max_age_s
                result[camera_id] = {
                    "timestamp": frame.timestamp, "age_s": age, "valid": frame.valid and fresh,
                    "error": frame.error if fresh else "Stale or future-dated frame",
                    "shape": list(frame.rgb.shape) if frame.rgb is not None else None,
                    "encoding": "rgb8", "intrinsic": frame.intrinsic,
                    "world_from_camera": frame.world_from_camera, "camera_model": frame.camera_model,
                    "distortion_coefficients": frame.distortion_coefficients,
                    "frame_convention": frame.frame_convention,
                }
        return {"enabled": self.enabled, "capture_count": self.capture_count, "frames": result}
