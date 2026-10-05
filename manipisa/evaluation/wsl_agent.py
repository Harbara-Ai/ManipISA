"""Own one Linux Codex process group for a Windows simulation host.

The first stdin line supplies command/cwd/env. Only child stdout is forwarded;
supervision evidence is written atomically without command arguments or secrets.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


HEARTBEAT_MAX_AGE_S = 20.0
POLL_S = 0.1
TERM_GRACE_S = 2.0
KILL_GRACE_S = 2.0
ALLOWED_ENV = {"OTEL_BLRP_SCHEDULE_DELAY", "PYTHONUTF8"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    value["updated_utc"] = now()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_info(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_members(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_info(path.name)
            if info is not None and info["pgid"] == pgid:
                members.append(info)
    return members


def clean_group(process, pgid, leader_start_ticks):
    """Clean only the session created for this child, even if its leader exited."""
    result = {"pgid": pgid, "term_sent": False, "kill_sent": False}

    def live_members():
        members = group_members(pgid)
        if pgid <= 1 or pgid == os.getpgrp():
            raise RuntimeError("Refusing to signal the supervisor process group")
        if any(m["sid"] != pgid or (m["pid"] == pgid and
                m["start_ticks"] != leader_start_ticks) for m in members):
            raise RuntimeError("Process group ownership changed")
        return [m for m in members if m["state"] not in ("Z", "X")]

    for sig, grace, field in ((signal.SIGTERM, TERM_GRACE_S, "term_sent"),
                              (signal.SIGKILL, KILL_GRACE_S, "kill_sent")):
        if not live_members():
            break
        try:
            os.killpg(pgid, sig)
            result[field] = True
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while live_members() and time.monotonic() < deadline:
            time.sleep(POLL_S)
    process.wait(timeout=KILL_GRACE_S)
    result["live_pids_after"] = [m["pid"] for m in live_members()]
    # Subreaper adoption lets us reap grandchildren after the direct child.
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        reaped += 1
    result["adopted_children_reaped"] = reaped
    result["complete"] = not result["live_pids_after"]
    return result


def heartbeat_fresh(path):
    try:
        return time.time() - path.stat().st_mtime <= HEARTBEAT_MAX_AGE_S
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    metadata = {"phase": "starting", "started_utc": now(), "supervisor_pid": os.getpid(),
                "platform": platform.platform(), "distro": os.environ.get("WSL_DISTRO_NAME"),
                "uid": os.getuid(), "python": sys.executable,
                "python_version": platform.python_version(), "heartbeat_max_age_s": HEARTBEAT_MAX_AGE_S}
    process = None
    relay = None
    pgid = leader_start_ticks = None
    received_signal = []
    exit_code = 70

    def on_signal(signum, _frame):
        received_signal.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        # Linux subreaping is local to this supervisor and its descendants.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot enable child subreaping")
        metadata["subreaper"] = True
        save(args.metadata, metadata)
        spec = json.loads(sys.stdin.readline())
        command, cwd, overrides = spec["command"], spec["cwd"], spec.get("env", {})
        if not isinstance(command, list) or not command or not all(
                isinstance(arg, str) and "\0" not in arg for arg in command):
            raise ValueError("Expected a command argument list")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError("Expected an existing absolute Linux working directory")
        executable = Path(command[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("Expected an absolute Linux executable path")
        if not isinstance(overrides, dict) or not all(k in ALLOWED_ENV and isinstance(v, str)
                for k, v in overrides.items()):
            raise ValueError("Unsupported environment override")
        env = dict(os.environ)
        env.update(overrides)
        relay_spec = spec.get("relay")
        if relay_spec is not None:
            from wsl_relay import FileRelay
            if not isinstance(relay_spec, dict) or not all(isinstance(relay_spec.get(k), str)
                    and relay_spec[k] for k in ("spool", "token", "endpoint")):
                raise ValueError("Invalid shared-file relay configuration")
            if not Path(relay_spec["spool"]).is_absolute():
                raise ValueError("Relay spool must be an absolute Linux path")
            relay = FileRelay(relay_spec["spool"], relay_spec["token"], args.heartbeat,
                              args.stop, heartbeat_max_age_s=HEARTBEAT_MAX_AGE_S).start()
            bypass = []
            for value in (env.get("NO_PROXY", ""), env.get("no_proxy", ""),
                          "localhost,127.0.0.1,127.0.0.2"):
                for host in value.split(","):
                    host = host.strip()
                    if host and host not in bypass:
                        bypass.append(host)
            env["NO_PROXY"] = env["no_proxy"] = ",".join(bypass)
            command = [arg.replace(relay_spec["endpoint"], relay.endpoint) for arg in command]
            redacted = [arg.replace(relay_spec["token"], "<local-capability>") for arg in command]
            command_path = args.metadata.parent / "agent-command.json"
            temporary = command_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(redacted, indent=2), encoding="utf-8")
            temporary.replace(command_path)
            metadata.update(transport="shared_files", relay_endpoint=relay.endpoint,
                            relay_spool=relay_spec["spool"])
        metadata.update(cwd=cwd, executable=str(executable), executable_realpath=str(executable.resolve()))
        digest = hashlib.sha256()
        with executable.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        metadata["executable_sha256"] = digest.hexdigest()
        version = subprocess.run([str(executable), "--version"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=env, cwd=cwd, timeout=10, check=False)
        metadata["executable_version"] = version.stdout.strip()[:256]
        metadata["version_exit_code"] = version.returncode
        if args.stop.exists():
            metadata["exit_reason"] = "stop_requested_before_launch"
            exit_code = 0
            return exit_code
        if not heartbeat_fresh(args.heartbeat):
            metadata["exit_reason"] = "host_heartbeat_expired_before_launch"
            exit_code = 74
            return exit_code
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        pgid = process.pid
        info = process_info(process.pid)
        if info is None or info["pgid"] != pgid or info["sid"] != pgid:
            raise RuntimeError("Child did not enter its dedicated Linux session")
        leader_start_ticks = info["start_ticks"]
        metadata.update(phase="running", pid=process.pid, pgid=pgid, sid=info["sid"],
                        leader_start_ticks=leader_start_ticks)
        save(args.metadata, metadata)
        while True:
            code = process.poll()
            if code is not None:
                metadata["exit_reason"] = "child_exited"
                exit_code = code if code >= 0 else 128 - code
                break
            if received_signal:
                metadata["exit_reason"] = "supervisor_signal"
                metadata["signal"] = received_signal[0]
                exit_code = 128 + received_signal[0]
                break
            if args.stop.exists():
                metadata["exit_reason"] = "stop_requested"
                exit_code = 0
                break
            if not heartbeat_fresh(args.heartbeat):
                metadata["exit_reason"] = "host_heartbeat_expired"
                exit_code = 74
                break
            time.sleep(POLL_S)
    except Exception as exc:
        metadata.update(exit_reason="supervisor_error", error_type=type(exc).__name__)
        print("WSL agent supervisor failed: " + type(exc).__name__, file=sys.stderr, flush=True)
        exit_code = 70
    finally:
        if process is not None:
            try:
                metadata["cleanup"] = clean_group(process, pgid, leader_start_ticks)
                if not metadata["cleanup"]["complete"]:
                    exit_code = 70
            except Exception as exc:
                metadata["cleanup"] = {"complete": False, "error_type": type(exc).__name__}
                exit_code = 70
            metadata["child_exit_code"] = process.returncode
        if relay is not None:
            relay.close()
        metadata.update(phase="finished", finished_utc=now(), supervisor_exit_code=exit_code)
        save(args.metadata, metadata)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
