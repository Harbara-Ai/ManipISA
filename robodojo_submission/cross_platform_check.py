"""Windows official WS client -> WSL official policy server; no Codex or physics."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit


ADDON = Path(__file__).resolve().parent
WORK = ADDON


def prepare_paths(args):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    project = ADDON.parent
    paths = [ADDON, project, args.robodojo, args.robodojo / "XPolicyLab"]
    paths.extend(args.python_deps)
    sys.path[:0] = [str(path) for path in paths]
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = os.environ.get(key, "") + ",localhost,127.0.0.1,::1,[::1]"


def save(path, report):
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def windows_path(path):
    path = Path(path).resolve()
    if len(path.parts) < 4 or path.parts[1] != "mnt" or len(path.parts[2]) != 1:
        raise ValueError("Cross-platform script/results must reside on a mounted Windows drive")
    return path.parts[2].upper() + ":\\" + "\\".join(path.parts[3:])


def client_check(url, output, args):
    """Invoked only by the WSL parent in the existing Windows Python environment."""
    if os.name != "nt":
        raise RuntimeError("The client phase requires Windows Python")
    prepare_paths(args)
    destination = urlsplit(url)
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] += "," + destination.hostname
    report = {"passed": False, "python": sys.executable, "url": url, "phase": "imports"}

    def deadline():
        report.update(error="Windows fixture exceeded its 20-second limit", phase="deadline")
        save(output, report)
        os._exit(124)

    timer = threading.Timer(20, deadline)
    timer.daemon = True
    timer.start()
    client = None
    try:
        import numpy as np
        from client_server.ws.model_client import WsModelClient
        from XPolicyLab.utils.process_data import encode_image_bit
        from protocol_check import observation

        report["phase"] = "connect"
        client = WsModelClient(url=url, evaluation_id="windows-to-wsl-fixture", trial_id="transport-only",
            action_case_id="transport-only", connect_timeout_s=3, handshake_timeout_s=3,
            request_timeout_s=3, max_connect_attempts=1, max_connect_seconds=4, close_timeout_s=1)
        report["phase"] = "episodes"
        episodes = []
        for encoded in (False, True):
            client.call(func_name="reset")
            obs = observation()
            actions = []
            for _ in range(14):
                wire = deepcopy(obs)
                if encoded:
                    for camera in wire["vision"].values():
                        camera["color"] = encode_image_bit(camera["color"])
                client.call(func_name="update_obs", obs=wire)
                chunk = client.call(func_name="get_action")
                assert len(chunk) == 1, "Expected one action per observation"
                action = chunk[0]
                assert set(action) == {f"{side}_{name}" for side in ("left", "right")
                                       for name in ("ee_pose", "ee_joint_state")}
                for side in ("left", "right"):
                    assert np.asarray(action[f"{side}_ee_pose"]).shape == (7,)
                    assert np.asarray(action[f"{side}_ee_joint_state"]).shape == (1,)
                actions.append(action)
                obs["state"].update(deepcopy(action))
            assert np.isclose(actions[-1]["left_ee_joint_state"][0], .25)
            assert actions[-1]["right_ee_pose"][0] > .3
            started = time.monotonic()
            client.call(func_name="reset")
            episodes.append({"encoded_rgb": encoded, "actions": len(actions),
                             "right_final_x": float(actions[-1]["right_ee_pose"][0]),
                             "left_gripper_target": float(actions[-1]["left_ee_joint_state"][0]),
                             "reset_seconds": time.monotonic() - started})
        report.update(passed=True, phase="complete", episodes=episodes)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        (output.parent / "client-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        if report["phase"] == "connect":
            destination = urlsplit(url)
            try:
                with socket.create_connection((destination.hostname, destination.port), timeout=2):
                    report["raw_tcp_connect_ok"] = True
            except OSError as tcp_exc:
                report["raw_tcp_error"] = f"{type(tcp_exc).__name__}: {tcp_exc}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                report.update(passed=False, cleanup_error=f"{type(exc).__name__}: {exc}")
        save(output, report)
        timer.cancel()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


async def attempt(host, output, args):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from manipisa_robodojo.model import Policy

    output.mkdir()
    captures = []

    def fixture(episode):
        episode.begin()
        captures.append(episode.adapter.observation["vision"]["cam_head"]["color"][0, 0].tolist())
        episode.tool("execute_python", {"code":
            "p = state()['state']['frames']['tool_a']['position']\n"
            "r = runtime.submit(Instruction(Opcode.MOVE, ('arm_a',), "
            "PoseGoal('tool_a', (p[0]+.012, p[1], p[2]), (1.,0.,0.,0.)), dwell_s=.08))\n"
            "set_gripper('left', .25)\nstep(12)\nprint(runtime.query(r.call_id).status)"})

    model = Policy({"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee",
                    "eval_batch": False, "artifact_dir": str(output / "episodes"),
                    "agent_wall_limit_s": 15, "action_timeout_s": 5}, runner=fixture)
    server = PolicyServer(model, PolicyServerConfig(host=host, port=0))
    process = None
    result = {"host": host, "passed": False, "phase": "server_start"}
    try:
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        url = f"ws://{'[' + host + ']' if ':' in host else host}:{port}"
        result.update(url=url, phase="windows_client")
        command = [str(args.windows_python), "-B",
                   windows_path(__file__), "--windows-client", url,
                   "--output", windows_path(output / "client-report.json"),
                   "--robodojo", windows_path(args.robodojo)]
        for path in args.windows_deps:
            command += ["--python-deps", windows_path(path)]
        result["windows_command"] = command
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=23)
        (output / "client-stdout.log").write_bytes(stdout)
        (output / "client-stderr.log").write_bytes(stderr)
        result["exit_code"] = process.returncode
        report_file = output / "client-report.json"
        if not report_file.exists():
            raise RuntimeError("Windows client did not write its report; inspect client-stderr.log")
        client_report = json.loads(report_file.read_text(encoding="utf-8"))
        result["client"] = client_report
        if client_report.get("passed"):
            assert len(captures) == 2, f"Expected two policy episodes, got {len(captures)}"
            assert captures[0] == [210, 50, 10]
            assert all(abs(value - expected) < 5 for value, expected in zip(captures[1], [210, 50, 10]))
            assert model._thread is None, "Final reset did not clear the policy worker"
            result["passed"] = True
        result["phase"] = "complete"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        model._closed.set()
        if model._episode is not None:
            model._episode.stop("cross_platform_check_cleanup")
        try:
            await asyncio.wait_for(server.stop(), timeout=4)
        except Exception as exc:
            result.update(passed=False, cleanup_error=f"{type(exc).__name__}: {exc}")
        await asyncio.to_thread(model.reset)
        result["captured_rgb"] = captures
        result["worker_cleared"] = model._thread is None
        save(output / "attempt-report.json", result)
    return result


async def server_check(output, args):
    report = {"passed": False, "server_python": sys.executable, "attempts": [],
              "scope": "Windows official WsModelClient to WSL official PolicyServer, deterministic ManipISA MOVE and raw gripper target; pose echo only, no physics or Codex",
              "official_score": None}
    started = time.monotonic()
    try:
        targets = ((args.host, "selected-host"),)
        for host, label in targets:
            result = await attempt(host, output / label, args)
            report["attempts"].append(result)
            save(output / "report.json", report)
            if result["passed"]:
                report.update(passed=True, working_host=host, working_url=result["url"])
                break
            if result.get("client", {}).get("phase") not in ("connect", None):
                break
    finally:
        report["elapsed_s"] = time.monotonic() - started
        save(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-client", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--windows-python", type=Path,
                        help="Windows python.exe path as seen from WSL")
    parser.add_argument("--windows-deps", type=Path, action="append", default=[],
                        help="Windows dependency overlay, as seen from WSL; repeatable")
    parser.add_argument("--python-deps", type=Path, action="append", default=[],
                        help="Dependency overlay for the current interpreter; repeatable")
    parser.add_argument("--host", choices=("::1", "127.0.0.1", "127.0.0.2"),
                        default="127.0.0.2", help="Loopback address shared by Windows and WSL")
    args = parser.parse_args()
    if args.windows_client:
        if args.output is None:
            parser.error("Client phase requires --output")
        return client_check(args.windows_client, args.output, args)
    if sys.platform != "linux":
        parser.error("Launch the check in WSL using Linux Python 3.11")
    if args.windows_python is None or not args.windows_python.is_file():
        parser.error("--windows-python must point to an existing Windows python.exe")
    prepare_paths(args)
    output = (args.output or WORK / "results" /
              ("cross-platform-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def deadline():
        save(output / "deadline.json", {"passed": False, "error": "Cross-platform check exceeded 59 seconds",
                                       "official_score": None})
        os._exit(124)

    timer = threading.Timer(59, deadline)
    timer.daemon = True
    timer.start()
    try:
        report = asyncio.run(server_check(output, args))
    finally:
        timer.cancel()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
