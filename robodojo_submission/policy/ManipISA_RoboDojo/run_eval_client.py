"""Execute the original evaluator with the policy's WebSocket request timeout.

The command line is exactly the official main.py command line. Keeping this
launcher in argv[0] also preserves the timeout through official os.execv retries.
"""
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def install_transport_timeout(module, timeout):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_s must be finite and positive")
    original = module.WsModelClient
    class PolicyTimeoutClient(original):
        def __init__(self, *args, **kwargs):
            if kwargs.get("request_timeout_s") is None:
                kwargs["request_timeout_s"] = timeout
            super().__init__(*args, **kwargs)
    module.WsModelClient = PolicyTimeoutClient


def main():
    policy = Path(__file__).resolve().parent
    root = policy.parents[2]
    pointer = policy / "submission-root.json"
    submission = Path(json.loads(pointer.read_text(encoding="utf-8"))["submission_root"]).resolve()
    sys.path[:0] = [str(root), str(root / "XPolicyLab"), str(submission), str(submission.parent)]
    from manipisa_robodojo.provenance import verify_core_tree, verify_official_checkouts
    verify_core_tree()
    verify_official_checkouts(root)
    import yaml
    from client_server.ws import model_client
    config = yaml.safe_load((policy / "deploy.yml").read_text(encoding="utf-8"))
    install_transport_timeout(model_client, config["request_timeout_s"])
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = os.environ.get(key, "") + ",localhost,127.0.0.1,127.0.0.2,::1,[::1]"
    os.chdir(root)
    entrypoint = root / "src/eval_client/main.py"
    # Unlike runpy.run_path, exec leaves argv[0] pointing at this launcher.
    namespace = {"__name__": "__main__", "__file__": str(entrypoint),
                 "__package__": None, "__spec__": None, "__builtins__": __builtins__}
    exec(compile(entrypoint.read_bytes(), str(entrypoint), "exec"), namespace)


if __name__ == "__main__":
    main()
