"""The project's 30/30/40 score; never infer missing evaluator/usage data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Mapping


def _number(value, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        return None
    if integer and value != int(value):
        return None
    return int(value) if integer else float(value)


def _bit(value):
    if isinstance(value, bool):
        return int(value)
    number = _number(value, integer=True)
    return number if number in (0, 1) else None


@dataclass(frozen=True)
class ScoreConfig:
    """References are shared across methods, calibrated outside test data."""
    wall_time_ref_s: float | None = None
    native_tokens_ref: int | None = None
    model_requests_ref: int | None = None
    partial_progress_weight: float = 0.25
    version: str = "manipisa-overall-v0.1"

    def __post_init__(self):
        for name in ("wall_time_ref_s", "native_tokens_ref", "model_requests_ref"):
            value = getattr(self, name)
            if value is not None and _number(value, positive=True) is None:
                raise ValueError(f"{name} must be finite and positive, or null")
        weight = _number(self.partial_progress_weight)
        if weight is None or weight > 1:
            raise ValueError("partial_progress_weight must be in [0, 1]")


def score_episode(official: Mapping, costs: Mapping, config: ScoreConfig) -> dict:
    """Consume independent Bench2Dex fields and an explicitly settled ledger.

    Required costs: wall_time_s, native_total_tokens, model_request_count,
    tokens_complete, requests_complete, settled. A partial value remains a
    diagnostic, not a replacement for the missing quantity.
    """
    issues = []
    success = _bit(official.get("stable_success"))
    progress = _number(official.get("latched_stage_completion_rate"))
    violation = _bit(official.get("safety_hard_violation"))
    if progress is not None and progress > 1:
        progress = None
    for name, value in (("stable_success", success),
                        ("latched_stage_completion_rate", progress),
                        ("safety_hard_violation", violation)):
        if value is None:
            issues.append(f"invalid_or_missing:{name}")
    if official.get("evaluation_valid") is not True:
        issues.append("evaluation_not_verified")
        success = progress = violation = None
    completion = (success + (1 - success) * config.partial_progress_weight * progress
                  if success is not None and progress is not None else None)
    reliability = success * (1 - violation) if success is not None and violation is not None else None
    wall = _number(costs.get("wall_time_s"))
    tokens = _number(costs.get("native_total_tokens"), integer=True)
    requests = _number(costs.get("model_request_count"), integer=True)
    if costs.get("tokens_complete") is not True:
        tokens = None
    if costs.get("requests_complete") is not True:
        requests = None
    efficiencies = {}
    for name, value, ref in (("wall_time", wall, config.wall_time_ref_s),
                             ("native_tokens", tokens, config.native_tokens_ref),
                             ("model_requests", requests, config.model_requests_ref)):
        if value is None:
            issues.append(f"invalid_or_missing:{name}")
        if ref is None:
            issues.append(f"uncalibrated:{name}")
        efficiencies[name] = 1 - min(value / ref, 1.) if value is not None and ref is not None else None
    if costs.get("settled") is not True:
        issues.append("accounting_not_settled")
    values = list(efficiencies.values())
    efficiency = sum(w * v for w, v in zip((.5, .4, .1), values)) if all(v is not None for v in values) else None
    effective = (completion * (1 - violation) * efficiency
                 if completion is not None and violation is not None and efficiency is not None else None)
    overall = (30 * completion + 30 * reliability + 40 * effective) if not issues else None
    return {"score_version": config.version, "score_config": asdict(config),
            "Overall": overall, "A": completion, "R": reliability, "E": efficiency,
            "C": effective, "efficiencies": efficiencies, "issues": issues,
            "stable_success": success, "LSCR": progress,
            "SafeSR": reliability, "hard_violation": violation,
            "wall_time_s": wall, "native_total_tokens": tokens, "model_request_count": requests}
