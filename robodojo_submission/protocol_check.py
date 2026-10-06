"""Exercise official XPolicyLab WebSocket transport without a simulator or model call."""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from manipisa_robodojo.model import Policy
from manipisa.adapters.robodojo import CAMERAS


def observation():
    state = {}
    for side, x in (("left", -.3), ("right", .3)):
        state[f"{side}_arm_joint_state"] = np.zeros(6)
        state[f"{side}_ee_pose"] = np.array([x, 0., 1., 1., 0., 0., 0.])
        state[f"{side}_ee_joint_state"] = np.array([1.])
    return {"data_format_version": "v1.0", "env_idx": 0,
            "additional_info": {"frequency": 25}, "instruction": "Protocol test; not a benchmark task.",
            "state": state, "vision": {name: {"color": np.full((32, 32, 3), [210, 50, 10], np.uint8)}
                                         for name in CAMERAS}}


def check(output, *, runner=None):
    # A workstation's external proxy must not intercept loopback test traffic.
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1,[::1]"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.utils.process_data import encode_image_bit

    host = "127.0.0.1" if os.name == "nt" else "::1"
    url_host = f"[{host}]" if ":" in host else host
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    captures = []

    def fixture_agent(episode):
        episode.begin()
        captures.append(episode.adapter.observation["vision"]["cam_head"]["color"][0, 0].tolist())
        episode.tool("execute_python", {"code":
            "p = state()['state']['frames']['tool_a']['position']\n"
            "r = runtime.submit(Instruction(Opcode.MOVE, ('arm_a',), "
            "PoseGoal('tool_a', (p[0]+.012, p[1], p[2]), (1.,0.,0.,0.)), dwell_s=.08))\n"
            "set_gripper('left', .25)\nstep(12)\nprint(runtime.query(r.call_id).status)"})

    model = Policy({"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "ee",
                    "eval_batch": False, "artifact_dir": str(output / "episodes"),
                    "agent_wall_limit_s": 180, "action_timeout_s": 240}, runner=runner or fixture_agent)
    ready, finish = queue.Queue(), threading.Event()
    failures = []

    def server_main():
        async def serve():
            server = PolicyServer(model, PolicyServerConfig(host=host, port=0))
            try:
                await server.start()
                ready.put(server._server.sockets[0].getsockname()[1])
                while not finish.is_set():
                    await asyncio.sleep(.02)
            finally:
                await server.stop()
        try:
            asyncio.run(serve())
        except BaseException as exc:
            failures.append(repr(exc))
            ready.put(exc)

    server_thread = threading.Thread(target=server_main, daemon=True)
    server_thread.start()
    client = None
    resets = []
    try:
        port = ready.get(timeout=10)
        if isinstance(port, BaseException):
            raise port
        client = WsModelClient(url=f"ws://{url_host}:{port}", evaluation_id="manipisa-protocol",
            trial_id="transport-only", action_case_id="transport-only", request_timeout_s=300,
            connect_timeout_s=5, max_connect_attempts=1, handshake_timeout_s=5)
        for encoded in (False, True):
            client.call(func_name="reset")
            obs = observation()
            actions = []
            for _ in range(14):
                wire = deepcopy(obs)
                if encoded:
                    for cam in wire["vision"].values():
                        cam["color"] = encode_image_bit(cam["color"])
                client.call(func_name="update_obs", obs=wire)
                chunk = client.call(func_name="get_action")
                assert len(chunk) == 1
                action = chunk[0]
                assert set(action) == {f"{side}_{key}" for side in ("left", "right")
                                       for key in ("ee_pose", "ee_joint_state")}
                for side in ("left", "right"):
                    assert np.asarray(action[f"{side}_ee_pose"]).shape == (7,)
                    assert np.asarray(action[f"{side}_ee_joint_state"]).shape == (1,)
                actions.append(action)
                obs["state"].update(deepcopy(action))
            if runner is None:
                assert np.isclose(actions[-1]["left_ee_joint_state"][0], .25)
                assert actions[-1]["right_ee_pose"][0] > .3
                assert abs(captures[-1][0] - 210) < 5 and abs(captures[-1][2] - 10) < 5
            start = time.monotonic()
            client.call(func_name="reset")
            resets.append(time.monotonic() - start)
            assert model._thread is None
        result = {"passed": True, "plain_rgb": True, "encoded_rgb": True,
                  "host": host,
                  "reset_seconds": resets, "captured_rgb": captures,
                  "scope": "Official XPolicyLab transport with a pose-echo fixture; no physics or score",
                  "official_score": None}
        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if client is not None:
            client.close()
        model.reset()
        finish.set()
        server_thread.join(timeout=5)
        if server_thread.is_alive() or failures:
            raise RuntimeError(f"Protocol server cleanup failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(check(parser.parse_args().output), indent=2))
