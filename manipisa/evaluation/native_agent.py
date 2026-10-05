"""Run the installed Codex CLI against local simulation-only MCP tools."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time

from .programs import EpisodeStopped
from .budgets import tool_timeout_seconds
from .processes import stop_process_tree
from .telemetry import otel_records


def run_native_agent(episode):
    out = episode.out
    project_root = Path(__file__).resolve().parents[2]
    agent_config = json.loads((project_root / "configs/ur5_wuji_codex.json").read_text(encoding="utf-8"))["agent"]
    model = agent_config["model"]
    effort = agent_config["reasoning_effort"]
    tool_timeout = tool_timeout_seconds(episode.wall_limit)
    physics_budget_text = f"{episode.budget} physics steps and " if hasattr(episode, "budget") else ""
    use_wsl = agent_config.get("execution_platform") == "wsl"
    wsl = agent_config.get("wsl", {})
    def linux_path(path):
        return wsl["project_root"].rstrip("/") + "/" + Path(path).resolve().relative_to(project_root).as_posix()
    pending, events = queue.Queue(), queue.Queue()
    telemetry, packet_hashes = [], set()
    token = secrets.token_hex(24)
    ending = threading.Event()
    relay_ending = threading.Event()
    relay_thread = None
    transport_errors = []

    def ingest_telemetry(body):
        import hashlib
        digest = hashlib.sha256(body).hexdigest()
        if digest not in packet_hashes:
            packet_hashes.add(digest)
            records = otel_records(json.loads(body))
            telemetry.extend(records)
            with (out / "otel.jsonl").open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/otel/" + token:
                ingest_telemetry(body)
                reply = {}
            elif self.path == "/tool" and self.headers.get("Authorization") == "Bearer " + token:
                response = queue.Queue(maxsize=1)
                pending.put((json.loads(body), response))
                while not ending.is_set():
                    try:
                        reply = response.get(timeout=.2)
                        break
                    except queue.Empty:
                        continue
                else:
                    # Do not return terminal evaluator labels to the policy or
                    # allow a fresh inference after the terminal boundary.
                    return
            else:
                self.send_error(403)
                return
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    work = out / "agent_workspace"
    work.mkdir()
    mcp_script = Path(__file__).with_name("agent_mcp.py")
    executable = wsl["codex_executable"] if use_wsl else shutil.which("codex") or "codex"
    mcp_python = wsl["python_executable"] if use_wsl else sys.executable
    mcp_path = linux_path(mcp_script) if use_wsl else str(mcp_script)
    command = [executable, "exec", "--ignore-user-config", "--ephemeral",
               "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--cd", linux_path(work) if use_wsl else str(work),
               "--model", model, "-c", f'model_reasoning_effort={json.dumps(effort)}',
               "-c", 'approval_policy="on-request"', "-c", 'approvals_reviewer="user"',
               "-c", 'web_search="disabled"', "--disable", "shell_tool", "--disable", "multi_agent",
               "--disable", "plugins", "--disable", "apps", "--disable", "skill_search",
               "-c", 'project_doc_max_bytes=0', "-c", 'otel.log_user_prompt=false',
               "-c", f'otel.exporter={{otlp-http={{endpoint="{endpoint}/otel/{token}",protocol="json"}}}}',
               "-c", f'mcp_servers.experiment.command={json.dumps(mcp_python)}',
               "-c", f'mcp_servers.experiment.args={json.dumps([mcp_path, "--endpoint", endpoint, "--token", token, "--timeout", str(tool_timeout)])}',
               "-c", 'mcp_servers.experiment.required=true',
               "-c", f'mcp_servers.experiment.tool_timeout_sec={tool_timeout}',
               "-c", 'mcp_servers.experiment.enabled_tools=["start_task","observe","read_api","execute_python"]',
               "-c", 'mcp_servers.experiment.default_tools_approval_mode="prompt"',
               "-c", 'mcp_servers.experiment.tools.start_task.approval_mode="approve"',
               "-c", 'mcp_servers.experiment.tools.observe.approval_mode="approve"',
               "-c", 'mcp_servers.experiment.tools.read_api.approval_mode="approve"',
               "-c", 'mcp_servers.experiment.tools.execute_python.approval_mode="approve"',
               f"The human has authorized this local simulation benchmark: Bench2Dex, Direct versus ManipISA, dual UR5+Wuji, {model} / {effort}, privileged state plus RGB, one episode per interface, {physics_budget_text}{episode.wall_limit:g} wall seconds per episode. You are the evaluated robot agent. First call experiment.start_task exactly once. Then use only the experiment tools to complete the received task. Work autonomously within its budget; do not ask the human questions. execute_python controls only this local simulation through the described public APIs; no real robot, external network, or user file edits are part of the task. This is a simulated robot task, not a software development task."]
    # The only token is an ephemeral loopback capability, not account auth.
    (out / "agent-command.json").write_text(json.dumps([c.replace(token, "<local-capability>") for c in command], indent=2), encoding="utf-8")
    import os
    env = dict(os.environ)
    env["OTEL_BLRP_SCHEDULE_DELAY"] = "500"
    env["PYTHONUTF8"] = "1"
    stderr = (out / "agent-stderr.log").open("w", encoding="utf-8")
    heartbeat_ending = threading.Event()
    heartbeat_thread = None
    stop_path = out / "agent-stop"
    if use_wsl:
        heartbeat = out / "host-heartbeat"
        heartbeat.touch()
        spool = out / "wsl_bridge"
        requests_dir, responses_dir = spool / "requests", spool / "responses"
        requests_dir.mkdir(parents=True)
        responses_dir.mkdir()
        def send_response(path, response):
            while not ending.is_set():
                try:
                    reply = response.get(timeout=.2)
                except queue.Empty:
                    continue
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(reply), encoding="utf-8")
                temporary.replace(path)
                return
        def relay_loop():
            try:
                while True:
                    for path in sorted(requests_dir.glob("*.json")):
                        message = json.loads(path.read_text(encoding="utf-8"))
                        if message["kind"] == "otel":
                            ingest_telemetry(json.dumps(message["payload"]).encode())
                        elif message["kind"] == "tool":
                            response = queue.Queue(maxsize=1)
                            pending.put((message["payload"], response))
                            threading.Thread(target=send_response, args=(responses_dir / path.name, response), daemon=True).start()
                        else:
                            raise ValueError("Unknown WSL relay message kind")
                        path.unlink()
                    if relay_ending.is_set():
                        break
                    relay_ending.wait(.02)
            except Exception as exc:
                transport_errors.append(f"{type(exc).__name__}: {exc}")
        relay_thread = threading.Thread(target=relay_loop, daemon=True)
        relay_thread.start()
        launch = ["wsl.exe", "--distribution", wsl["distribution"], "--user", wsl["user"],
                  "--cd", linux_path(work), "--exec", wsl["python_executable"], "-B",
                  linux_path(Path(__file__).with_name("wsl_agent.py")),
                  "--heartbeat", linux_path(heartbeat), "--stop", linux_path(stop_path),
                  "--metadata", linux_path(out / "wsl-agent.json")]
        (out / "agent-launch.json").write_text(json.dumps({"execution_platform": "wsl", "command": launch,
            "model": model, "reasoning_effort": effort}, indent=2), encoding="utf-8")
        proc = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                cwd=work, env=env, text=True, encoding="utf-8", errors="replace")
        def heartbeat_loop():
            while not heartbeat_ending.wait(1):
                heartbeat.touch()
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        try:
            proc.stdin.write(json.dumps({"command": command, "cwd": linux_path(work),
                "relay": {"spool": linux_path(spool), "token": token, "endpoint": endpoint},
                "env": {"OTEL_BLRP_SCHEDULE_DELAY": "500", "PYTHONUTF8": "1"}}) + "\n")
            proc.stdin.close()
        except BaseException:
            stop_path.touch()
            heartbeat_ending.set()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop_process_tree(proc)
            relay_ending.set()
            ending.set()
            relay_thread.join(timeout=3)
            server.shutdown()
            server.server_close()
            stderr.close()
            raise
    else:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                                cwd=work, env=env, text=True, encoding="utf-8", errors="replace")
    def read_events():
        with (out / "agent-events.jsonl").open("w", encoding="utf-8") as stream:
            for line in proc.stdout:
                stream.write(line)
                stream.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.put((event, time.perf_counter()))
    reader = threading.Thread(target=read_events, daemon=True)
    reader.start()
    startup = time.perf_counter()
    start_utc, end_utc = None, None
    completed = False
    try:
        while proc.poll() is None and episode.reason is None:
            if transport_errors:
                raise RuntimeError("WSL relay failed: " + transport_errors[0])
            if episode.started is not None:
                episode.check_time()
            elif time.perf_counter() - startup > 120:
                raise RuntimeError("Agent did not request its task within 120 seconds")
            while not events.empty():
                event, timestamp = events.get_nowait()
                if event.get("type") in ("turn.completed", "turn.failed"):
                    completed = event.get("type") == "turn.completed"
                    if episode.started is not None:
                        episode.stop("agent_stopped" if completed else "agent_error")
                        episode.terminal_time = timestamp
                    else:
                        raise RuntimeError("Agent ended before receiving the task")
            try:
                request, response = pending.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                name, arguments = request["name"], request.get("arguments", {})
                if name == "start_task":
                    if episode.started is not None:
                        raise ValueError("Task already delivered")
                    episode.begin()
                    start_utc = datetime.now(timezone.utc).isoformat()
                    description = task_instructions(episode)
                    reply = {"content": [{"type": "text", "text": description}, *episode.first_observation]}
                else:
                    reply = episode.tool(name, arguments)
            except EpisodeStopped:
                raise
            except Exception as exc:
                reply = {"isError": True, "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}]}
            response.put(reply)
        if episode.started is not None and episode.reason is None:
            episode.stop("agent_stopped" if proc.returncode == 0 else "agent_error")
    except EpisodeStopped as exc:
        if episode.reason is None:
            episode.stop(str(exc) or "wall_timeout")
    finally:
        end_utc = datetime.now(timezone.utc).isoformat()
        # Hold the current tool response until termination. This gives the
        # telemetry exporter a flush interval without a new policy inference.
        if proc.poll() is None:
            time.sleep(1.1)
            # If the wall deadline arrived during inference, allow its usage
            # to settle while holding all tool replies and the physics clock.
            settle_deadline = time.perf_counter() + 25
            while time.perf_counter() < settle_deadline:
                sends = [r for r in telemetry if r.get("event.name") == "codex.websocket_request"
                         and r.get("model") == model]
                finishes = [r for r in telemetry if r.get("event.name") == "codex.sse_event"
                            and r.get("event.kind") == "response.completed" and r.get("model") == model]
                if sends and len(finishes) >= len(sends):
                    break
                time.sleep(.25)
            if use_wsl:
                heartbeat_ending.set()
                stop_path.touch()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_process_tree(proc)
            else:
                stop_process_tree(proc)
        heartbeat_ending.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        relay_ending.set()
        if relay_thread is not None:
            relay_thread.join(timeout=3)
        ending.set()
        reader.join(timeout=3)
        server.shutdown()
        server.server_close()
        stderr.close()
    effective_configs = [{k: r.get(k) for k in ("model", "reasoning_effort", "sandbox_policy", "approval_policy", "app.version", "auth_mode")}
                         for r in telemetry if r.get("event.name") == "codex.conversation_starts"]
    wsl_metadata = None
    if use_wsl:
        cleanup_deadline = time.perf_counter() + 25
        while time.perf_counter() < cleanup_deadline:
            if (out / "wsl-agent.json").exists():
                wsl_metadata = json.loads((out / "wsl-agent.json").read_text(encoding="utf-8"))
                if wsl_metadata.get("phase") == "finished":
                    break
            time.sleep(.2)
        wsl_metadata = wsl_metadata or {}
        if wsl_metadata.get("cleanup", {}).get("complete") is not True:
            raise RuntimeError("WSL agent process cleanup was not verified; see wsl-agent.json")
    return {"telemetry": telemetry, "task_start_utc": start_utc, "task_end_utc": end_utc,
            "effective_configs": effective_configs,
            "execution_platform": "wsl" if use_wsl else sys.platform,
            "wsl_process": wsl_metadata,
            "process_completed": completed, "exit_code": proc.returncode}


def task_instructions(episode):
    if hasattr(episode, "agent_instructions"):
        return episode.agent_instructions()
    common = f"""Task: {episode.task['description']}
Robot: dual UR5 + dual Wuji, fixed base. Interface: {episode.method}.
Observation: identical privileged robot/object state plus RGB cameras in this order: {", ".join(episode.rig.camera_ids)}.
Budget: {episode.budget} physics steps at {1 / episode.dt:g} Hz; {episode.wall_limit} wall seconds.
Physics pauses during reasoning. Tools, debugging and retries consume wall time.
Use execute_python with persistent variables; np, torch, math, math_utils, state(), step(n) are preloaded.
Print focused diagnostics: stdout above 4000 characters is shown as a marked head/tail preview; full output is logged.
Ordinary Python errors may occur after physical actions; check the returned physics_step before retrying.
Do not import modules. state() reads current public state without stepping. step(n) is the only physics clock.
Do not access private attributes, hidden evaluator data, demonstration actions, files, network, or reset/teleport state.
Use read_api('index') for available reference documents. Use observe for an explicit full-state/RGB refresh.
After an execute_python call advances physics, its reply automatically includes current RGB images and
visual_feedback with the matching physics_step, camera_ids, compact joint/object state and camera calibration.
A query-only call returns visual_feedback.status=unchanged without duplicate images. A Python error after
partial motion still returns fresh RGB; a visual_feedback error means no valid new images are available.
Images arrive as image content blocks, ordered by camera_ids. When calling these tools via functions.exec,
forward each text block with text(c.text) and each image block with image(c). Do not print/stringify the
whole tool result: base64 text is not visual input. Inspect each newly returned image before continuing.
Body states use position xyz/quaternion wxyz/linear velocity/angular velocity; object pose_world uses quaternion xyzw.
Success is determined independently by the environment; no success or stage labels are exposed.
Solve the task using physical actions and feedback. Use bounded action segments and return between
approach, hand closure, test lift and transport so you can inspect the new images and adjust.
Images are produced when the Python call returns, not between step() calls inside one long script.
Persistent controllers and closed-loop Python control are allowed; step(n) keeps its full execution semantics.
"""
    if episode.method == "direct":
        common += """Control using native SDK: robot.set_joint_position_target(target, joint_ids=None),
robot.set_joint_velocity_target, robot.set_joint_effort_target, robot.find_joints, robot.find_bodies.
robot.data contains COPIES of joint_pos, joint_vel, joint_pos_limits, soft_joint_pos_limits,
default_joint_pos, body_state_w, body_link_state_w, root_state_w, root_pos_w, root_quat_w.
robot.joint_names/body_names/device/is_fixed_base/num_joints are available.
robot.root_physx_view.get_jacobians() returns a copy. DifferentialIKController and DifferentialIKControllerCfg are preloaded.
Native control targets have a leading environment dimension of 1. Use the SDK reference for IK.
You may create persistent controllers, trajectories, and feedback programs yourself.
"""
    else:
        common += """Control using runtime.submit/query/cancel/update/feedback and Instruction, Opcode, Mode,
PoseGoal, ShapeGoal, ContactGoal, GraspGoal, WrenchGoal, ContactRequirement, ContactPoint, Status.
Read 'instructions' for exact dataclass constructors and 'contracts' for execution semantics.
Read 'quickstart' for tested MOVE, SHAPE_HAND, query and explicit handoff examples before issuing commands.
submit returns an ExecutionReport: query/cancel/update take report.call_id, whereas replace_handle takes
the latest terminal report.control_handle. Inspect status/reason/evidence before retrying a rejection.
arm_a/hand_a are right; arm_b/hand_b are left. tool_a/tool_b are wrist frames; palm_a/palm_b are observed hand palms.
step(n) advances the runtime and evaluator together. Methods return actual contract feedback, not task scores.
Each Python reply includes compact runtime feedback: current state, all running/active instructions,
new or changed terminal instructions, and the complete current control_handles mapping.
Valid zero-contact records are grouped in state.empty_contact_groups; expand each record for its ids.
state.contacts contains the remaining full records; absence from that mapping alone does not mean no contact.
Use runtime.query(call_id) for a full instruction report, and runtime.feedback() for full state/history.
Print only needed fields or slices of those results; avoid printing all of runtime.feedback().
Read 'feedback' for the compact response format. Explicit queries do not consume automatic feedback updates.
Retained hold handles occupy resources; use submit(..., replace_handle=...) for same-resource takeover.
Rigid task objects have contacts named side_region_objectid; regions palm/thumb_tip/index_tip/middle_tip/ring_tip/little_tip.
Palm contact uses the arm controller and finger contact uses the hand controller.
No calibrated separation or friction is supplied: BREAK_CONTACT/CONTROL_GRASP may reject unsupported evidence.
Articulated object link contact bindings and object-level MOVE are currently unsupported.
"""
    return common
