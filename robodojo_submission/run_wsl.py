#!/usr/bin/env python3
"""Run the ManipISA policy and Codex in WSL; RoboDojo physics can stay on Windows."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

# This launcher only reads the existing ManipISA tree, including Python imports.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ADDON = Path(__file__).resolve().parent
PROJECT = ADDON.parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "protocol", "agent", "server"))
    parser.add_argument("--project", type=Path, default=PROJECT,
                        help="This submission's core checkout (other checkouts are rejected)")
    parser.add_argument("--robodojo", type=Path, default=PROJECT.parent / "RoboDojo")
    parser.add_argument("--output", type=Path,
                        help="New directory within robodojo_submission/results")
    parser.add_argument("--host", choices=("::1", "127.0.0.1", "127.0.0.2"), default="127.0.0.2",
                        help="Policy server loopback address; default: 127.0.0.2")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--model", choices=("gpt-6-astra",), default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", choices=("high",), default="high")
    parser.add_argument("--codex", help="Explicit native Linux Codex executable")
    parser.add_argument("--python-deps", type=Path, action="append", default=[],
                        help="Additional isolated Linux Python dependency directory; repeatable")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("Use a Linux Python interpreter inside WSL, not python.exe.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.project.resolve() != PROJECT:
        parser.error("Use this independent checkout's core; --project cannot select the original project")
    for name, directory, required in (
        ("--project", args.project, "manipisa"),
        ("--robodojo", args.robodojo, "XPolicyLab/client_server/ws/model_server.py"),
    ):
        if not (directory / required).exists():
            parser.error(f"{name} does not contain {required}: {directory}")
    return args


def prepare_paths(args):
    paths = [ADDON, args.project.resolve(), args.robodojo.resolve(),
             args.robodojo.resolve() / "XPolicyLab"]
    paths.extend(path.resolve() for path in args.python_deps)
    for path in reversed(paths):
        while str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    import manipisa
    if Path(manipisa.__file__).resolve().parent != PROJECT / "manipisa":
        raise RuntimeError("Another ManipISA checkout is already loaded; start a fresh process")
    inherited = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    bypass = [part.strip() for part in inherited.split(",") if part.strip()]
    for host in ("localhost", "127.0.0.1", "::1", "[::1]", args.host):
        if host not in bypass:
            bypass.append(host)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(bypass)


def configuration(args, output):
    from manipisa_robodojo.runtime import resolve_codex

    config = {
        "bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee",
        "eval_batch": False, "reasoning_effort": args.reasoning_effort,
        "artifact_dir": str(output / "episodes"), "agent_wall_limit_s": 600,
        "agent_startup_timeout_s": 180, "action_timeout_s": 900,
        "request_timeout_s": 960,
    }
    if args.mode == "agent":
        config.update(agent_wall_limit_s=180, action_timeout_s=370)
    if args.model:
        config["model"] = args.model
    if args.codex:
        config["codex_path"] = args.codex
    config.update(resolve_codex(config))
    return config


def policy_url(host, port):
    return f"ws://{'[' + host + ']' if ':' in host else host}:{port}"


def inspect_runtime(args, config):
    packages = {}
    for package in ("numpy", "scipy", "Pillow", "websockets", "PyYAML", "msgpack", "msgpack-numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.executable,
        "platform": sys.platform,
        "project": str(args.project.resolve()),
        "robodojo": str(args.robodojo.resolve()),
        "codex": {key: str(config[key]) for key in
                  ("codex_path", "model", "reasoning_effort", "codex_home", "config_path")
                  if config.get(key) is not None},
        "packages": packages,
        "msgpack_numpy_importable": importlib.util.find_spec("msgpack_numpy") is not None,
        "policy_url": policy_url(args.host, args.port),
        "provenance": args.provenance,
        "scope": "Configuration inspection only; no model call, physics, or official score",
    }


async def serve(config, port, host="127.0.0.2"):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from manipisa_robodojo.model import Policy

    model = Policy(config)
    server = PolicyServer(model, PolicyServerConfig(host=host, port=port))
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        loop.add_signal_handler(sig, stopped.set)
    try:
        await server.start()
        print(json.dumps({"listening": policy_url(host, port), "model": config["model"],
                          "reasoning_effort": config["reasoning_effort"],
                          "artifact_dir": config["artifact_dir"]}, indent=2), flush=True)
        await stopped.wait()
    finally:
        try:
            # Leave the old queues/events alive until in-flight calls drain.
            # reset() replaces them and would race with a waiting get_action().
            model._closed.set()
            if model._episode is not None:
                model._episode.stop("policy_server_shutdown")
            await server.stop()
        finally:
            try:
                await asyncio.to_thread(model.reset)
            finally:
                for sig in signals:
                    loop.remove_signal_handler(sig)


def main(argv=None):
    args = arguments(argv)
    prepare_paths(args)
    from manipisa_robodojo.provenance import verify_core_tree, verify_official_checkouts
    args.provenance = {"core": verify_core_tree(),
                       "official": verify_official_checkouts(args.robodojo)}
    from manipisa_robodojo.runtime import result_path
    output = result_path(args.output or ADDON / "results" /
                         f"wsl-{args.mode}-{datetime.now():%Y%m%d-%H%M%S-%f}")
    if args.mode == "protocol":
        # The transport fixture does not require a Codex executable or login.
        from protocol_check import check
        print(json.dumps(check(output), indent=2), flush=True)
        return 0

    config = configuration(args, output)
    if args.mode == "inspect":
        print(json.dumps(inspect_runtime(args, config), indent=2), flush=True)
    elif args.mode == "agent":
        from agent_check import check
        print(json.dumps(check(output, config=config), indent=2), flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "launch.json").write_text(
            json.dumps(inspect_runtime(args, config), indent=2), encoding="utf-8")
        asyncio.run(serve(config, args.port, args.host))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
