"""Launch the unmodified RoboDojo evaluator on Windows against a WSL policy server.

Default mode is one diagnostic episode through RoboDojo's own EVAL_NUM option.
--full retains the task's native repeat count; it does not run every task.
The only injected behavior is the WebSocket client's transport timeout.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.util
import importlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys


WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
POLICY = "ManipISA_RoboDojo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT,
                        help="This isolated submission checkout; no original project is imported")
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--host", default="127.0.0.2", help="WSL policy server address")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--seed", type=int, default=0, help="Official evaluator policy seed")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=960.0)
    parser.add_argument("--overlay", type=Path, action="append", default=[],
                        help="Additional isolated dependency directory; repeatable")
    parser.add_argument("--protocol-deps", type=Path,
                        help="Optional Windows WebSocket dependency overlay (higher import priority)")
    parser.add_argument("--accept-eula", action="store_true",
                        help="Confirm acceptance of the NVIDIA Omniverse EULA for this run")
    parser.add_argument("--isaaclab-source", type=Path,
                        help="Optional IsaacLab checkout instead of the installed package")
    parser.add_argument("--curobo-source", type=Path,
                        help="Optional cuRobo checkout; defaults to RoboDojo/third_party/curobo")
    parser.add_argument("--full", action="store_true",
                        help="Use native repeat count for this task (default: one diagnostic episode)")
    parser.add_argument("--check-only", action="store_true", help="Preflight only; do not launch Isaac Sim")
    parser.add_argument("--check-imports", action="store_true", help="Import non-Isaac simulation dependencies during preflight")
    parser.add_argument("--output", type=Path, help="New directory for the launcher manifest")
    return parser.parse_args()


def _inside(path, parent):
    return path == parent or parent in path.parents


def _add_no_proxy(host):
    for key in ("NO_PROXY", "no_proxy"):
        values = [v for v in os.environ.get(key, "").split(",") if v]
        for value in ("127.0.0.1", "localhost", "::1", "[::1]", host):
            if value not in values:
                values.append(value)
        os.environ[key] = ",".join(values)


def _memory_status():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load_percent", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in ("physical_total", "physical_available", "commit_limit",
                "commit_available", "virtual_total", "virtual_available", "extended_available")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {name: getattr(status, name) for name, _ in status._fields_ if name != "length"}


def main():
    args = parse_args()
    if os.name != "nt":
        raise SystemExit("Run this launcher with Windows python.exe; the policy server and Codex run in WSL.")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Isaac Sim 5.1 requires the prepared Windows Python 3.11 environment.")
    if not 1 <= args.port <= 65535 or not math.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0:
        raise SystemExit("Port must be 1..65535 and request timeout must be positive.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.task):
        raise SystemExit("Task must be an official task module name.")
    root, project = args.robodojo_root.resolve(), args.project.resolve()
    if project != PROJECT:
        raise SystemExit("--project must be this isolated submission checkout.")
    if _inside(root, project) or _inside(project, root):
        raise SystemExit("Use a separate RoboDojo checkout, outside this submission repository.")
    entrypoint = root / "src/eval_client/main.py"
    task_config = root / "task/RoboDojo/config" / (args.task + ".yml")
    task_module = root / "task/RoboDojo/tasks" / (args.task + ".py")
    policy_deploy = root / "XPolicyLab/policy" / POLICY / "deploy.yml"
    for path in (entrypoint, task_config, task_module, policy_deploy, project / "manipisa/__init__.py"):
        if not path.is_file():
            raise SystemExit(f"Required file missing: {path}")

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    if args.accept_eula:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if not args.check_only and os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() != "YES":
        raise SystemExit("Accept the NVIDIA Omniverse EULA before launching (--accept-eula).")
    _add_no_proxy(args.host)
    # Never pip-install into the existing Windows environment.
    overlays = list(args.overlay)
    default_overlay = WORK / "sim-deps-windows"
    if default_overlay.is_dir():
        overlays.append(default_overlay)
    protocol_overlay = (args.protocol_deps or WORK / "protocol-deps").resolve()
    if args.protocol_deps is not None and not protocol_overlay.is_dir():
        raise SystemExit(f"Protocol dependency directory missing: {protocol_overlay}")
    if protocol_overlay.is_dir():
        overlays.insert(0, protocol_overlay)
    paths = [Path(__file__).resolve().parent, project, root, root / "XPolicyLab"]
    # Only the official wire protocol needs priority over Isaac Sim's older
    # websockets. Missing simulation packages fall back to the isolated overlay,
    # preserving the existing simulator's Pillow/numpy/other pinned versions.
    if protocol_overlay.is_dir():
        paths.append(protocol_overlay)
    if args.isaaclab_source is not None:
        source = args.isaaclab_source.resolve() / "source"
        if not (source / "isaaclab/isaaclab/__init__.py").is_file():
            raise SystemExit(f"Invalid IsaacLab source checkout: {source.parent}")
        paths.extend(p for p in sorted(source.iterdir()) if (p / p.name / "__init__.py").is_file())
    curobo = (args.curobo_source or root / "third_party/curobo").resolve()
    if not (curobo / "curobo/__init__.py").is_file():
        raise SystemExit(f"Invalid cuRobo source checkout: {curobo}")
    paths.append(curobo)
    for path in paths:
        if not path.is_dir():
            raise SystemExit(f"Dependency directory missing: {path}")
    # Keep benchmark utils ahead of XPolicyLab's namespaced utility package.
    ordered_paths = list(dict.fromkeys(str(path) for path in paths))
    sys.path[:0] = ordered_paths
    from manipisa_robodojo.provenance import verify_core_tree, verify_official_checkouts
    provenance = {"core": verify_core_tree(), "official": verify_official_checkouts(root)}
    package_source = WORK / "policy" / POLICY
    installed_policy = policy_deploy.parent
    for source in package_source.iterdir():
        if source.is_file() and source.suffix in (".py", ".sh", ".yml"):
            installed = installed_policy / source.name
            if not installed.is_file() or installed.read_bytes() != source.read_bytes():
                raise SystemExit(f"Installed submission policy differs: {installed}. Use a fresh registration.")
    fallback_paths = [str(path.resolve()) for path in overlays if path != protocol_overlay]
    sys.path.extend(path for path in fallback_paths if path not in sys.path)
    os.environ["PYTHONPATH"] = os.pathsep.join(ordered_paths + [os.environ.get("PYTHONPATH", "")])
    dll_handles = []
    for overlay in overlays:
        for relative in ("nvidia/cuda_runtime/bin", "nvidia/cuda_runtime/lib",
                         "nvidia/cuda_nvrtc/bin", "nvidia/cuda_nvrtc/lib"):
            directory = overlay / relative
            if directory.is_dir():
                dll_handles.append(os.add_dll_directory(str(directory.resolve())))

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + f"_{os.getpid()}"
    os.environ["ROBODOJO_RUN_ID"] = run_id
    os.environ["EVAL_NUM"] = "native" if args.full else "1"
    label = "native-repeats-wsl-codex" if args.full else "diagnostic-one-episode-wsl-codex"
    url_host = f"[{args.host}]" if ":" in args.host else args.host
    command_args = [str(entrypoint), "--task_name", args.task, "--env_cfg_type", "arx_x5",
                    "--num_envs", "1", "--enable_cameras", "--headless",
                    "--kit_args", " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
                    "--device_id", str(args.device_id), "--policy_name", POLICY,
                    "--port", str(args.port), "--host", args.host, "--protocol", "ws",
                    "--policy_server_url", f"ws://{url_host}:{args.port}",
                    "--additional_info", label, "--seed", str(args.seed)]
    output = (args.output or WORK / "results" / ("windows-sim-" + run_id)).resolve()
    if not _inside(output, WORK / "results"):
        raise SystemExit("Keep launcher output in this submission's results/ directory.")
    output.mkdir(parents=True, exist_ok=False)
    command_args[command_args.index("--kit_args") + 1] += (
        f" --portable-root {(WORK / 'kit-runtime').as_posix()}"
        f" --/app/settings/persistent=false --/log/file={(output / 'isaacsim.log').as_posix()}")
    os.environ["NUMBA_CACHE_DIR"] = str(output / "numba-cache")
    os.environ["CUDA_CACHE_PATH"] = str(output / "cuda-cache")
    os.environ["WARP_CACHE_PATH"] = str(output / "warp-cache")

    modules = ("numpy", "scipy", "PIL", "yaml", "cv2", "msgpack", "msgpack_numpy", "websockets",
               "transforms3d", "open3d", "yourdfpy", "quaternion", "cuda", "warp",
               "curobo", "isaaclab", "isaacsim", "omegaconf")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    assets = root / "Assets"
    asset_status = {relative: (assets / relative).exists() for relative in (
        "Robots/x5", "Room/Simple_Room_nolight", "Material", "Object/RoboDojo/Geometry/camera_stand",
        "Object/RoboDojo/Rigid/bowl", f"Eval_Layout/RoboDojo/arx_x5/{args.seed}")}
    blockers = ["Missing Python modules: " + ", ".join(missing)] if missing else []
    if not assets.is_dir():
        blockers.append("Official Assets directory is missing; download official scene assets first.")
    elif args.task == "stack_bowls":
        absent = [path for path, present in asset_status.items() if not present]
        if absent:
            blockers.append("Missing stack_bowls asset directories: " + ", ".join(absent))
    if "yaml" not in missing:
        import yaml
        with task_config.open(encoding="utf-8") as stream:
            task_cfg = yaml.safe_load(stream) or {}
        if task_cfg.get("Articulation"):
            blockers.append("This task uses the official Linux-only fcntl PhysX monitor; Windows support is not implemented.")
        with policy_deploy.open(encoding="utf-8") as stream:
            deploy_cfg = yaml.safe_load(stream) or {}
        if deploy_cfg.get("eval_batch", False):
            blockers.append("This adapter requires eval_batch=false in its added deploy.yml.")
    imported = {}
    if args.check_imports and not missing:
        from manipisa_robodojo.source_version import load_curobo
        try:
            imported["curobo_source"] = load_curobo(curobo)
        except Exception as exc:
            blockers.append(f"cuRobo source preparation failed: {type(exc).__name__}: {exc}")
        for module_name in ("omegaconf", "transforms3d", "open3d", "yourdfpy", "quaternion",
                            "cuda.core", "warp", "curobo", "curobo.inverse_kinematics"):
            try:
                module = importlib.import_module(module_name)
                imported[module_name] = {"file": getattr(module, "__file__", None),
                                         "version": getattr(module, "__version__", None)}
            except Exception as exc:
                imported[module_name] = {"error": f"{type(exc).__name__}: {exc}"}
                blockers.append(f"Import failed: {module_name}: {type(exc).__name__}: {exc}")
    manifest = {"python": sys.executable, "platform": sys.platform, "official_entrypoint": str(entrypoint),
                "argv": command_args, "working_directory": str(root), "sys_path_additions": ordered_paths,
                "fallback_dependency_paths": fallback_paths,
                "provenance": provenance,
                "mode": "native_repeats_for_one_task" if args.full else "diagnostic_one_episode",
                "EVAL_NUM": os.environ["EVAL_NUM"], "official_score": None,
                "request_timeout_s": args.request_timeout_s,
                "transport_override": "WsModelClient request_timeout_s default only; scoring and physics unchanged",
                "asset_path_presence_only": asset_status, "missing_modules": missing,
                "dependency_imports": imported,
                "memory_at_preflight_bytes": _memory_status(),
                "kit_portable_root": str(WORK / "kit-runtime"),
                "blockers": blockers, "check_only": args.check_only}
    (output / "launcher-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if blockers:
        raise SystemExit("Preflight blocked; see launcher-manifest.json. No simulator was launched.")
    if args.check_only:
        print("Preflight completed. Asset presence and import lookup do not validate CUDA or physics.", flush=True)
        return

    from client_server.ws import model_client
    original_client = model_client.WsModelClient

    class TransportTimeoutClient(original_client):
        def __init__(self, *client_args, **client_kwargs):
            if client_kwargs.get("request_timeout_s") is None:
                client_kwargs["request_timeout_s"] = args.request_timeout_s
            super().__init__(*client_args, **client_kwargs)

    model_client.WsModelClient = TransportTimeoutClient
    from manipisa_robodojo.source_version import load_curobo
    source_version = load_curobo(curobo)
    (output / "curobo-source.json").write_text(json.dumps(source_version, indent=2), encoding="utf-8")
    os.chdir(root)
    sys.argv = command_args
    print(f"Launching official RoboDojo evaluator ({label}); scores will be under {root / 'eval_result'}", flush=True)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
