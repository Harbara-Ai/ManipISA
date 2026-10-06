"""Compose unchanged ManipISA contracts with the isolated public RoboDojo host."""
import json
from pathlib import Path
import queue
import threading
from uuid import uuid4

from manipisa.evaluation.programs import EpisodeStopped
from manipisa.evaluation.robodojo import RoboDojoModel
from .adapter import public_observation, ROBODOJO_REVISION, XPOLICYLAB_REVISION
from .agent import CodexRunner
from .episode import RoboDojoEpisode, FEEDBACK_FORMAT, FRAME_SEMANTICS
from .provenance import verify_core_tree


class Policy(RoboDojoModel):
    def __init__(self, model_cfg, *, runner=None):
        config = dict(model_cfg)
        if config.get("bench_name", "RoboDojo") != "RoboDojo":
            raise ValueError("This add-on is for RoboDojo simulation only")
        self.core_provenance = verify_core_tree()
        super().__init__(config, runner=runner if runner is not None else CodexRunner(config))

    def reset(self):
        # Keep counters alive until the worker has joined and its final audit is
        # written. The official deployment loop may reset immediately after its
        # final take_action, without supplying that action's next observation.
        if getattr(self, "_thread", None) is not None:
            self._closed.set()
            if self._episode is not None:
                self._episode.stop("environment_reset_or_end")
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise RuntimeError("Previous agent did not terminate; refusing to mix episodes")
            if self._episode is not None:
                self._write_summary(self._episode, final=True)
        self._thread = self._episode = self._latest = self._error = None
        self._actions, self._observations = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._closed, self._done = threading.Event(), threading.Event()
        self._sent = False
        self.sent_actions = self.received_action_observations = 0

    def get_action(self):
        actions = super().get_action()
        # This includes fallback holds after agent completion. A returned action
        # is not evidence that take_action subsequently executed it.
        self.sent_actions += len(actions)
        return actions

    def update_obs(self, obs):
        # Use the submission allowlist before retaining or dispatching any input.
        clean = public_observation(obs)
        if self._latest is not None and not self._sent:
            raise RuntimeError("An observation requires a preceding get_action; duplicate observations cannot advance time")
        follows_action = self._latest is not None and self._sent
        self._latest = clean
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, args=(clean,), daemon=True,
                                            name="manipisa-robodojo-agent")
            self._thread.start()
        elif not self._done.is_set():
            self._observations.put_nowait(clean)
        if follows_action:
            self.received_action_observations += 1
        self._sent = False

    def _write_summary(self, episode, *, final=False):
        unconfirmed = self.sent_actions - self.received_action_observations
        summary = {"reason": episode.reason, "error": self._error,
            "summary_finalized_at_reset": final,
            "confirmed_observation_intervals": episode.steps,
            "sent_actions": self.sent_actions,
            "received_action_observations": self.received_action_observations,
            "sent_actions_without_followup_observation": unconfirmed,
            "last_sent_action_execution": ("unknown_without_followup_observation" if unconfirmed
                else "followup_observation_received" if self.sent_actions else "no_actions_sent"),
            "interval_count_semantics": (
                "confirmed_observation_intervals counts action exchanges whose follow-up observation "
                "was consumed by the Runtime adapter, including warmup. sent_actions counts all actions "
                "returned to the official client, including fallback holds; sending does not prove execution. "
                "received_action_observations also includes follow-up observations after the agent stops. "
                "The final sent action may execute and terminate the environment without another observation. "
                "These counters are not the official evaluator's executed-step total."
            ),
            "observation": "robodojo_public_observation+RGB",
            "supported_isa": ["MOVE"], "raw_gripper_command": True,
            "frame_semantics": FRAME_SEMANTICS,
            "core_provenance": self.core_provenance,
            "tool_feedback_format": FEEDBACK_FORMAT,
            "rgb_observations": {"total": episode.queries,
                "automatic_after_execution": episode.visual_feedback_count},
            "robodojo_reference_revision": ROBODOJO_REVISION,
            "xpolicylab_reference_revision": XPOLICYLAB_REVISION,
            "official_score": None}
        (episode.out / "adapter-result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _run(self, initial):
        default_out = Path(__file__).resolve().parents[1] / "results" / "policy"
        out = Path(self.cfg.get("artifact_dir", default_out)) / uuid4().hex
        try:
            self._episode = RoboDojoEpisode(initial, self._exchange, out, wall_limit=self.wall_limit)
            self._episode.warmup()
            self.runner(self._episode)
        except EpisodeStopped as exc:
            if self._episode is not None:
                self._episode.stop(str(exc) or "episode_stopped")
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if self._episode is not None:
                    self._episode.stop("agent_error" if self._error else "agent_stopped")
                    self._write_summary(self._episode)
            finally:
                self._done.set()
