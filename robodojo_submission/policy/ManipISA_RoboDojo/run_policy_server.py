"""Standard XPolicyLab entrypoint with bounded SIGINT/SIGTERM policy cleanup."""
import asyncio
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def main():
    policy = Path(__file__).resolve().parent
    root = policy.parents[2]
    pointer = policy / "submission-root.json"
    submission = Path(json.loads(pointer.read_text(encoding="utf-8"))["submission_root"]).resolve()
    sys.path[:0] = [str(submission), str(submission.parent), str(root), str(root / "XPolicyLab")]
    from manipisa_robodojo.provenance import verify_core_tree, verify_official_checkouts
    from manipisa_robodojo.runtime import resolve_codex, result_path
    from setup_policy_server import parse_args_and_config
    from run_wsl import serve
    verify_core_tree()
    verify_official_checkouts(root)
    config = parse_args_and_config()
    if config.get("protocol") != "ws":
        raise ValueError("This submission uses the official WebSocket protocol")
    port = int(config["port"])
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    config["artifact_dir"] = str(result_path(config.get("artifact_dir", "results/evaluation")))
    config.update(resolve_codex(config))
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = os.environ.get(key, "") + ",localhost,127.0.0.1,127.0.0.2,::1,[::1]"
    asyncio.run(serve(config, port, config["host"]))


if __name__ == "__main__":
    main()
