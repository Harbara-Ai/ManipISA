"""One validated budget definition shared by catalog, runner and simulator."""
from dataclasses import asdict, dataclass
import math


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def tool_timeout_seconds(wall_limit_s):
    return math.ceil(_positive(wall_limit_s, "wall_time_limit_s") + 60)


@dataclass(frozen=True)
class EpisodeBudget:
    physics_dt: float
    expert_budget_scale: float
    policy_stride: int
    wall_limit_s: float

    @classmethod
    def from_config(cls, config, *, wall_limit_s=None):
        stride = config["policy_stride"]
        if type(stride) is not int or stride < 1:
            raise ValueError("policy_stride must be a positive integer")
        wall = config["development_run"]["wall_time_limit_s"] if wall_limit_s is None else wall_limit_s
        return cls(_positive(config["physics_dt"], "physics_dt"),
                   _positive(config["expert_budget_scale"], "expert_budget_scale"), stride,
                   _positive(wall, "wall_time_limit_s"))

    def physics_steps(self, expert_steps):
        expert = _positive(expert_steps, "expert_time_step")
        return math.ceil(expert * self.expert_budget_scale / self.policy_stride) * self.policy_stride

    @property
    def tool_timeout_s(self):
        # A tool cannot legitimately run longer than the whole active episode.
        return tool_timeout_seconds(self.wall_limit_s)

    @property
    def process_timeout_s(self):
        # Scene/model startup and process-tree cleanup are outside task wall time.
        return math.ceil(self.wall_limit_s + 600)

    def to_dict(self):
        return {**asdict(self), "tool_timeout_s": self.tool_timeout_s,
                "process_timeout_s": self.process_timeout_s}
