"""Synchronous runtime: the host advances physics with step(), independent of Agent calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from uuid import uuid4

from .adapters.base import Adapter
from .evidence import evaluate_goal
from .feedback import summarize_feedback
from .types import (
    AdapterError, Evidence, ExecutionReport, Instruction, Mode, PreparedTask,
    StateSnapshot, Status, TERMINAL, Truth, validate_instruction,
    ContactGoal, GraspGoal, WrenchGoal, json_safe,
)


class _PreparationRejected(AdapterError):
    """Carry the checked public evidence into a rejected submission report."""
    def __init__(self, reason, detail, *, evidence, goal):
        super().__init__(reason, detail)
        self.evidence = evidence
        self.goal = goal


@dataclass
class _Call:
    instruction: Instruction
    prepared: PreparedTask
    started: float
    report: ExecutionReport
    satisfied_since: float | None = None
    commanded: bool = False


@dataclass
class _Hold:
    handle: str
    prepared: PreparedTask
    owner: str
    created_at: float
    expires_at: float
    state: str = "HOLDING_POSITION"
    detail: str = "Position control retained; this does not certify a grasp"


class Runtime:
    def __init__(self, adapter: Adapter, rgb=None):
        self.adapter = adapter
        self.rgb = rgb
        self._calls: dict[str, _Call] = {}
        self._rejections: dict[str, ExecutionReport] = {}
        self._holds: dict[str, _Hold] = {}
        self._state_error: str | None = None
        self._snapshot = adapter.observe()
        self._check_clock(self._snapshot)

    def _check_clock(self, state):
        if not math.isfinite(state.timestamp):
            raise AdapterError("EVIDENCE_UNAVAILABLE", "Invalid simulation timestamp")
        if hasattr(self, "_last_time") and state.timestamp < self._last_time - 1e-9:
            raise AdapterError("CLOCK_RESET", "Create a new Runtime after resetting simulation")
        self._last_time = state.timestamp

    def _observe(self):
        try:
            state = self.adapter.observe()
            self._check_clock(state)
        except Exception as exc:
            self._state_error = str(exc)
            raise
        self._state_error = None
        self._snapshot = state
        return state

    def _predicates(self, names, command, state):
        result = {}
        for name in names:
            e = state.predicates.get(name, Evidence(Truth.UNKNOWN, state.timestamp, "runtime", "Missing predicate"))
            if not isinstance(e.truth, Truth) or not e.fresh(state.timestamp, command.max_observation_age_s):
                e = Evidence(Truth.UNKNOWN, e.timestamp, e.source, "Stale or future-dated predicate")
            result[name] = e
        return result

    def _adapter_evidence(self, command, prepared, state, entering=False):
        if not hasattr(self.adapter, "check"):
            return {}
        evidence = self.adapter.check(command, prepared, state, entering=entering)
        result = {}
        for key, e in evidence.items():
            if not isinstance(e, Evidence) or not isinstance(e.truth, Truth) or not e.fresh(state.timestamp, command.max_observation_age_s):
                e = Evidence(Truth.UNKNOWN, state.timestamp, "runtime", "Invalid adapter evidence")
            result[key] = e
        return result

    def _joint_speed_evidence(self, command, state):
        result = {}
        for actor in command.actors:
            joint = state.joints.get(actor)
            if joint is None or not joint.valid or not joint.velocity or not all(math.isfinite(v) for v in joint.velocity):
                result[f"joint_speed:{actor}"] = Evidence(Truth.UNKNOWN, state.timestamp, "runtime", "Joint speed unavailable")
                continue
            speed = max(abs(v) for v in joint.velocity)
            evidence = Evidence(Truth.SATISFIED if speed <= command.max_joint_speed + 1e-6 else Truth.VIOLATED,
                                joint.timestamp, joint.source, f"Observed max joint speed={speed}")
            if not evidence.fresh(state.timestamp, command.max_observation_age_s):
                evidence = Evidence(Truth.UNKNOWN, joint.timestamp, joint.source, "Joint speed observation stale")
            result[f"joint_speed:{actor}"] = evidence
        return result

    @staticmethod
    def _conflict(a, b):
        return bool(a.resources & b.resources or a.coupling_keys & b.coupling_keys)

    def _require_call_id(self, value, operation):
        if not isinstance(value, str):
            hint = "Use report.call_id" if isinstance(value, ExecutionReport) else (
                "Use report['call_id'] for a serialized report" if isinstance(value, dict)
                else "Pass the call_id string from the submission report")
            raise TypeError(f"{operation}(call_id) expects a call_id string; received {type(value).__name__}. {hint}.")
        if value in self._holds or value.startswith("hold_"):
            raise ValueError(f"{operation}(call_id) received a control_handle. Use the owning report.call_id; "
                             "control_handle is only for submit(..., replace_handle=...).")
        if value not in self._calls and value not in self._rejections:
            raise KeyError(f"Unknown call_id {value!r}; use a call_id returned by this Runtime.submit().")
        return value

    def _require_replace_handle(self, value):
        if not isinstance(value, str):
            hint = "Use report.control_handle" if isinstance(value, ExecutionReport) else (
                "Use report['control_handle'] for a serialized report" if isinstance(value, dict)
                else "Pass a retained control_handle string")
            raise AdapterError("INVALID_ARGUMENT", f"replace_handle expects a control_handle string; "
                               f"received {type(value).__name__}. {hint}.")
        if value in self._calls or value in self._rejections:
            report = self.query(value)
            if report.control_handle is not None:
                detail = f"Use runtime.query(call_id).control_handle ({report.control_handle!r})."
            else:
                detail = f"This call is {report.status} and has no retained control_handle. No call was canceled."
            raise AdapterError("RESOURCE_CONFLICT", "replace_handle received a call_id. " + detail)
        if value not in self._holds:
            raise AdapterError("RESOURCE_CONFLICT", f"Unknown or already replaced control_handle {value!r}; "
                               "inspect runtime.feedback()['control_handles'] for current ownership.")

    def _prepare(self, instruction, replace_handle=None, exclude_call=None):
        validate_instruction(instruction)
        if replace_handle is not None:
            self._require_replace_handle(replace_handle)
        state = self._observe()
        if instruction.env_id != state.env_id:
            raise AdapterError("UNSUPPORTED_CAPABILITY", "Adapter and instruction env_id differ")
        try:
            prepared = self.adapter.prepare(instruction, state)
        except AdapterError as exc:
            if exc.reason == "UNREACHABLE":
                raise AdapterError(exc.reason, "Requested goal rejected by adapter before execution: " + str(exc)) from exc
            raise
        if not prepared.resources:
            raise AdapterError("BACKEND_ERROR", "Prepared task has no physical resources")
        if replace_handle is not None:
            old = self._holds.get(replace_handle)
            if old is None or old.prepared.resources != prepared.resources or old.prepared.coupling_keys != prepared.coupling_keys:
                raise AdapterError("RESOURCE_CONFLICT", "Handoff must replace exactly the retained resource set")
        for call_id, call in self._calls.items():
            if call_id != exclude_call and call.report.status not in TERMINAL and self._conflict(prepared, call.prepared):
                raise AdapterError("RESOURCE_CONFLICT", f"Resources or coupling domain owned by {call_id}")
        for handle, hold in self._holds.items():
            if handle != replace_handle and self._conflict(prepared, hold.prepared):
                raise AdapterError("RESOURCE_CONFLICT", f"Resources retained by {handle}; explicit handoff required")
        # Recheck immediately before taking control; preparation is not authorization to move.
        state = self._observe()
        predicates = self._predicates((*instruction.entry_guard, *instruction.invariants,
                                      *prepared.mandatory_invariants), instruction, state)
        predicates.update(self._joint_speed_evidence(instruction, state))
        predicates.update(self._adapter_evidence(instruction, prepared, state, entering=True))
        goal = evaluate_goal(instruction, state)
        if goal.evidence.truth == Truth.UNKNOWN:
            raise _PreparationRejected("EVIDENCE_UNAVAILABLE", "Required goal observation is unavailable",
                                       evidence=predicates, goal=goal)
        if any(e.truth != Truth.SATISFIED for e in predicates.values()):
            detail = "; ".join(f"{k}={v.truth}: {v.detail}" for k, v in predicates.items()
                               if v.truth != Truth.SATISFIED)
            raise _PreparationRejected("PRECONDITION_FAILED", "Current-state preconditions not satisfied: " + detail,
                                       evidence=predicates, goal=goal)
        return prepared, state

    def submit(self, instruction: Instruction, *, replace_handle: str | None = None) -> ExecutionReport:
        call_id = uuid4().hex
        try:
            prepared, state = self._prepare(instruction, replace_handle)
        except Exception as exc:
            fallback = "INVALID_ARGUMENT" if isinstance(exc, (ValueError, TypeError)) else "BACKEND_ERROR"
            report = ExecutionReport(call_id, instruction.operation, Status.REJECTED, self._snapshot.timestamp,
                                     getattr(exc, "reason", fallback), str(exc),
                                     goal=getattr(exc, "goal", None), evidence=getattr(exc, "evidence", {}))
            self._rejections[call_id] = report
            return deepcopy(report)
        report = ExecutionReport(call_id, instruction.operation, Status.RUNNING, state.timestamp)
        self._calls[call_id] = _Call(instruction, prepared, state.timestamp, report)
        if replace_handle is not None:
            del self._holds[replace_handle]
        return deepcopy(report)

    def query(self, call_id: str) -> ExecutionReport:
        self._require_call_id(call_id, "query")
        if call_id in self._rejections:
            return deepcopy(self._rejections[call_id])
        return deepcopy(self._calls[call_id].report)

    def _finish(self, call, status, state, reason=None, detail=""):
        if call.report.status in TERMINAL:
            return
        handle = "hold_" + call.report.call_id
        hold = _Hold(handle, call.prepared, call.report.call_id, state.timestamp,
                     state.timestamp + call.instruction.hold_for_s)
        try:
            self.adapter.hold(call.prepared)
        except Exception as exc:
            hold.state = "FAULTED"
            hold.detail = f"Position hold could not be established: {exc}"
            detail = (detail + "; " + hold.detail).strip("; ")
            if status == Status.SUCCEEDED:
                status, reason = Status.FAILED, "CONTINUATION_FAILED"
        self._holds[handle] = hold
        if isinstance(call.instruction.goal, (ContactGoal, GraspGoal, WrenchGoal)) and hold.state == "HOLDING_POSITION":
            hold.detail = "Last bounded position targets retained; no continuing grasp or wrench certification"
        call.report = replace(call.report, status=status, timestamp=state.timestamp,
                              reason=reason, detail=detail, control_handle=handle)

    def cancel(self, call_id: str) -> ExecutionReport:
        self._require_call_id(call_id, "cancel")
        call = self._calls.get(call_id)
        if call is None or call.report.status in TERMINAL:
            return self.query(call_id)
        state = self._observe()
        # A pending violation takes precedence over cancellation.
        self._assess(call, state)
        if call.report.status not in TERMINAL:
            self._finish(call, Status.CANCELED, state, "CANCELED", "Caller canceled; position control retained")
        return self.query(call_id)

    def update(self, call_id: str, instruction: Instruction) -> dict:
        try:
            self._require_call_id(call_id, "update")
            if not isinstance(instruction, Instruction):
                raise TypeError("update(call_id, instruction) expects an Instruction instance as its second argument.")
        except (TypeError, ValueError, KeyError) as exc:
            return {"accepted": False, "reason": "INVALID_ARGUMENT", "detail": str(exc)}
        call = self._calls.get(call_id)
        if call is None or call.report.status in TERMINAL:
            return {"accepted": False, "reason": "TERMINAL_CALL", "detail": "Submit a new instruction"}
        self._assess(call, self._observe())
        if call.report.status in TERMINAL:
            return {"accepted": False, "reason": "TERMINAL_CALL", "detail": "Existing contract already ended"}
        try:
            old = call.instruction
            if isinstance(old.goal, (ContactGoal, GraspGoal, WrenchGoal)):
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Interaction updates require a new call and explicit handoff; acquisition anchors cannot be reset")
            if (instruction.operation, instruction.actors, instruction.mode, instruction.env_id) != (
                    old.operation, old.actors, old.mode, old.env_id):
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Update cannot change operation, actors, mode or environment")
            # This slice supports goal-only updates; obligations and deadlines are unchanged.
            if replace(instruction, goal=old.goal) != old:
                raise AdapterError("UNSUPPORTED_CAPABILITY", "Only goal updates are implemented")
            prepared, state = self._prepare(instruction, exclude_call=call_id)
            if prepared.resources != call.prepared.resources or prepared.coupling_keys != call.prepared.coupling_keys:
                raise AdapterError("RESOURCE_CONFLICT", "Update changes resource ownership")
            if call.report.status == Status.ACTIVE and evaluate_goal(instruction, state).evidence.truth != Truth.SATISFIED:
                raise AdapterError("PRECONDITION_FAILED", "Updated SUSTAIN goal must already hold")
        except (AdapterError, ValueError, TypeError) as exc:
            result = {"accepted": False, "reason": getattr(exc, "reason", "INVALID_ARGUMENT"), "detail": str(exc)}
            if isinstance(exc, _PreparationRejected):
                result["evidence"] = {key: json_safe(asdict(value)) for key, value in exc.evidence.items()}
                result["goal"] = json_safe(asdict(exc.goal))
            return result
        call.instruction, call.prepared, call.satisfied_since = instruction, prepared, None
        return {"accepted": True, "reason": None, "detail": "Original time budget preserved"}

    def _assess(self, call, state):
        if call.report.status in TERMINAL:
            return
        command = call.instruction
        names = (*command.invariants, *call.prepared.mandatory_invariants)
        if call.report.status == Status.ACTIVE:
            names += command.active_invariants
        if not call.commanded:
            names += command.entry_guard
        evidence = self._predicates(names, command, state)
        evidence.update(self._joint_speed_evidence(command, state))
        evidence.update(self._adapter_evidence(command, call.prepared, state, entering=not call.commanded))
        goal = evaluate_goal(command, state)
        call.report = replace(call.report, timestamp=state.timestamp, goal=goal, evidence=evidence)
        if any(e.truth == Truth.VIOLATED for e in evidence.values()):
            failure = next(e for e in evidence.values() if e.truth == Truth.VIOLATED)
            return self._finish(call, Status.FAILED, state, failure.failure_reason, "An execution invariant was violated: " + failure.detail)
        if any(e.truth == Truth.UNKNOWN for e in evidence.values()) or goal.evidence.truth == Truth.UNKNOWN:
            call.satisfied_since = None
            return self._finish(call, Status.FAILED, state, "EVIDENCE_UNAVAILABLE", "Required evidence became unknown")
        elapsed = state.timestamp - call.started
        if elapsed > command.timeout_s + 1e-9:
            return self._finish(call, Status.FAILED, state, "TIMEOUT", "Total execution budget exceeded")
        if command.mode == Mode.SUSTAIN and call.report.status == Status.RUNNING and elapsed > command.activation_timeout_s + 1e-9:
            return self._finish(call, Status.FAILED, state, "ACTIVATION_TIMEOUT", "SUSTAIN was not activated in time")
        if call.report.status == Status.ACTIVE:
            if goal.evidence.truth != Truth.SATISFIED:
                reason = "GRASP_INVALID" if isinstance(command.goal, GraspGoal) else "CONSTRAINT_VIOLATED"
                return self._finish(call, Status.FAILED, state, reason, "Active SUSTAIN target was lost")
            if hasattr(self.adapter, "relation"):
                call.report = replace(call.report, relation=self.adapter.relation(command, call.prepared, state))
        elif goal.evidence.truth == Truth.SATISFIED:
            if call.satisfied_since is None:
                call.satisfied_since = goal.evidence.timestamp
            # Reusing a cached observation cannot manufacture dwell time.
            if goal.evidence.timestamp - call.satisfied_since + 1e-9 >= command.dwell_s:
                if hasattr(self.adapter, "relation"):
                    call.report = replace(call.report, relation=self.adapter.relation(command, call.prepared, state))
                if command.mode == Mode.REACH:
                    return self._finish(call, Status.SUCCEEDED, state)
                active_evidence = self._predicates(command.active_invariants, command, state)
                if any(e.truth != Truth.SATISFIED for e in active_evidence.values()):
                    call.report = replace(call.report, evidence={**evidence, **active_evidence})
                    reason = "CONSTRAINT_VIOLATED" if any(e.truth == Truth.VIOLATED for e in active_evidence.values()) else "EVIDENCE_UNAVAILABLE"
                    return self._finish(call, Status.FAILED, state, reason, "Activation constraints not satisfied")
                call.report = replace(call.report, status=Status.ACTIVE, active_since=state.timestamp,
                                      evidence={**evidence, **active_evidence})
        else:
            call.satisfied_since = None
        if call.report.status == Status.ACTIVE and command.sustain_s is not None:
            if state.timestamp - call.report.active_since + 1e-9 >= command.sustain_s:
                self._finish(call, Status.SUCCEEDED, state)

    def step(self, *, return_snapshot: bool = True) -> StateSnapshot | None:
        """Advance one tick; hosts may skip the otherwise unused return copy.

        Observations, contract checks, commands and retained-control checks are
        identical in both paths. The default still returns an isolated snapshot.
        """
        try:
            state = self._observe()
        except Exception as exc:
            for call in self._calls.values():
                self._finish(call, Status.FAILED, self._snapshot, getattr(exc, "reason", "BACKEND_ERROR"), str(exc))
            raise
        for call in self._calls.values():
            self._assess(call, state)
        for call in self._calls.values():
            if call.report.status not in TERMINAL:
                try:
                    self.adapter.command(call.instruction, call.prepared)
                    call.commanded = True
                except Exception as exc:
                    self._finish(call, Status.FAILED, state, getattr(exc, "reason", "BACKEND_ERROR"), str(exc))
        try:
            self.adapter.step()
            state = self._observe()
        except Exception as exc:
            self._state_error = str(exc)
            for call in self._calls.values():
                self._finish(call, Status.FAILED, state, getattr(exc, "reason", "BACKEND_ERROR"), str(exc))
            # Do not claim that a failed or reset simulator produced a new valid snapshot.
            raise
        for call in self._calls.values():
            self._assess(call, state)
        for hold in self._holds.values():
            if hold.state == "HOLDING_POSITION":
                owner = self._calls[hold.owner]
                e = self._predicates(hold.prepared.mandatory_invariants, owner.instruction, state)
                if any(v.truth != Truth.SATISFIED for v in e.values()):
                    hold.state, hold.detail = "FAULTED", "Retained position control lost required state evidence"
                elif state.timestamp >= hold.expires_at:
                    hold.state, hold.detail = "EXPIRED", "Hold interval ended; FAULT_AND_HOLD retains resources pending explicit handoff"
        if self.rgb is not None:
            self.rgb.sample(state.timestamp)
        return deepcopy(state) if return_snapshot else None

    def set_rgb_enabled(self, enabled: bool):
        if self.rgb is None and enabled:
            raise AdapterError("UNSUPPORTED_CAPABILITY", "No RGB source was attached")
        if self.rgb is not None:
            self.rgb.set_enabled(enabled)

    def feedback(self) -> dict:
        """Structured feedback; RGB pixel arrays are obtained separately from rgb.frames()."""
        return {
            "state": self._snapshot.to_dict(),
            "state_valid": self._state_error is None,
            "state_error": self._state_error,
            "instructions": [self.query(k).to_dict() for k in (*self._calls, *self._rejections)],
            "control_handles": {key: {"owner": h.owner, "state": h.state, "created_at": h.created_at,
                "expires_at": h.expires_at, "resources": sorted(h.prepared.resources), "detail": h.detail}
                for key,h in self._holds.items()},
            "rgb": self.rgb.metadata(self._snapshot.timestamp) if self.rgb is not None else {"enabled": False, "frames": {}},
        }

    def feedback_summary(self, *, previous_reports=None) -> dict:
        """Current state plus active/changed reports, without consuming a cursor.

        No observation or physics is triggered. Explicit ``query`` and full
        ``feedback`` calls remain independent of the caller-owned report baseline.
        """
        # Serialize non-contact state once; do not materialize a complete
        # StateSnapshot.to_dict() before compacting the contact collection.
        state = json_safe(asdict(replace(self._snapshot, contacts={})))
        state["contacts"] = self._snapshot.contacts
        return summarize_feedback({
            "state": state,
            "state_valid": self._state_error is None,
            "state_error": self._state_error,
            "instructions": [call.report for call in self._calls.values()] + list(self._rejections.values()),
            "control_handles": {key: {"owner": h.owner, "state": h.state, "created_at": h.created_at,
                "expires_at": h.expires_at, "resources": sorted(h.prepared.resources), "detail": h.detail}
                for key, h in self._holds.items()},
            "rgb": self.rgb.metadata(self._snapshot.timestamp) if self.rgb is not None else {"enabled": False, "frames": {}},
        }, previous_reports=previous_reports)

    def images(self):
        """Optional RGB attachments; separate from the JSON-serializable feedback."""
        return self.rgb.frames(self._snapshot.timestamp) if self.rgb is not None else {}
