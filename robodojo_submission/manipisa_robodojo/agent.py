"""RoboDojo-specific Codex transport, without patching the Bench2Dex runner."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time

from manipisa.evaluation import agent_mcp
from manipisa.evaluation.programs import EpisodeStopped
from manipisa.evaluation.budgets import tool_timeout_seconds
from manipisa.evaluation.telemetry import otel_records
from .runtime import resolve_codex, result_path, SUBMISSION_ROOT


BOOTSTRAP = (
    "Run the user's authorized local RoboDojo SIMULATION evaluation using ManipISA. "
    "The original environment controls dual ARX X5 arms and grippers. "
    "First call experiment.start_task exactly once to receive the task, budget and observations. "
    "Then complete the task autonomously using only experiment tools. "
    "The tools exchange public robot observations and three RGB cameras with the local simulator. "
    "execute_python controls this simulation through the supplied API only. "
    "There is no real robot or external actuation. Do not use files, shell, network or hidden "
    "evaluator information. Do not claim task success from command submission. "
    "If a tool approval is denied, report the denial and stop; do not bypass it."
)


def _source_hashes():
    roots = {"manipisa": Path(agent_mcp.__file__).resolve().parents[1],
             "addon": Path(__file__).resolve().parent}
    return {f"{name}/{path.relative_to(root).as_posix()}": hashlib.sha256(path.read_bytes()).hexdigest()
            for name, root in roots.items() for path in sorted(root.rglob("*.py"))}


def _stop(process):
    """Stop only the process group/tree created by this runner, within reset's budget."""
    if process is None:
        return
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=5, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def codex_command(settings, work, endpoint, capability, wall_limit):
    """Build the fixed, simulation-only child command without starting a process."""
    model = settings["model"]
    tool_timeout = tool_timeout_seconds(wall_limit)
    command = [settings["codex_path"],
               "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
               "--sandbox", "read-only", "--json", "--cd", str(work), "--model", model]
    overrides = {
        "model_reasoning_effort": settings["reasoning_effort"],
        "approval_policy": "on-request", "approvals_reviewer": "auto_review",
        "web_search": "disabled", "project_doc_max_bytes": 0,
        "forced_login_method": "chatgpt",
        "mcp_servers.experiment.command": sys.executable,
        "mcp_servers.experiment.args": [str(Path(agent_mcp.__file__).resolve()),
            "--endpoint", endpoint, "--token", capability, "--timeout", str(tool_timeout)],
        "mcp_servers.experiment.required": True,
        "mcp_servers.experiment.tool_timeout_sec": tool_timeout,
        "mcp_servers.experiment.enabled_tools": ["start_task", "observe", "read_api", "execute_python"],
        "otel.log_user_prompt": False,
    }
    for key, value in overrides.items():
        command += ["-c", key + "=" + json.dumps(value)]
    command += ["-c", f'otel.exporter={{otlp-http={{endpoint="{endpoint}/otel/{capability}",protocol="json"}}}}']
    for feature in ("shell_tool", "multi_agent", "multi_agent_v2", "plugins", "apps", "skill_search",
                    "view_image", "browser_use", "browser_use_external", "computer_use",
                    "image_generation", "hooks", "memories", "workspace_dependencies"):
        command += ["--disable", feature]
    command.append(BOOTSTRAP)
    return command


class CodexRunner:
    def __init__(self, config):
        if Path(agent_mcp.__file__).resolve().parents[2] != SUBMISSION_ROOT.parent:
            raise RuntimeError("The runner loaded a core outside this submission checkout")
        self.config = dict(config)
        self.settings = resolve_codex(self.config)

    def __call__(self, episode):
        pending, events = queue.Queue(), queue.Queue()
        closing = threading.Event()
        capability = secrets.token_hex(24)
        telemetry = []
        out = result_path(episode.out)
        sources_before = _source_hashes()
        (out / "source-sha256.json").write_text(json.dumps(sources_before, indent=2), encoding="utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/otel/" + capability:
                    records = otel_records(json.loads(body))
                    telemetry.extend(records)
                    with (out / "otel.jsonl").open("a", encoding="utf-8") as stream:
                        for record in records:
                            stream.write(json.dumps(record) + "\n")
                    reply = {}
                elif self.path == "/tool" and self.headers.get("Authorization") == "Bearer " + capability:
                    response = queue.Queue(maxsize=1)
                    pending.put((json.loads(body), response))
                    while not closing.is_set():
                        try:
                            reply = response.get(timeout=.05)
                            break
                        except queue.Empty:
                            pass
                    else:
                        self.send_error(410, "Episode ended")
                        return
                else:
                    self.send_error(403)
                    return
                data = json.dumps(reply).encode()
                if self.path == "/tool":
                    try:
                        episode.check_time()
                    except EpisodeStopped:
                        self.close_connection = True
                        return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        # A local IPv6 endpoint also works on hosts whose WSL IPv4 loopback is
        # intercepted. It never binds a public interface.
        class LocalHTTPServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        server = LocalHTTPServer(("::1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        endpoint = f"http://[::1]:{server.server_port}"
        work = out / "agent_workspace"
        work.mkdir()
        model = self.settings["model"]
        command = codex_command(self.settings, work, endpoint, capability, episode.wall_limit)
        (out / "agent-command.json").write_text(json.dumps(
            [part.replace(capability, "<local-capability>") for part in command], indent=2), encoding="utf-8")
        (out / "codex-runtime.json").write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        process = reader = None
        started_utc = ended_utc = None
        completed = False
        error = None
        stderr = (out / "agent-stderr.log").open("w", encoding="utf-8")
        try:
            env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OTEL_BLRP_SCHEDULE_DELAY="500")
            env["NO_PROXY"] = env.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1,[::1]"
            env["no_proxy"] = env["NO_PROXY"]
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=stderr, cwd=work, env=env, text=True, encoding="utf-8", errors="replace",
                start_new_session=os.name != "nt")

            def read_events():
                with (out / "agent-events.jsonl").open("w", encoding="utf-8") as stream:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        try:
                            events.put(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            reader = threading.Thread(target=read_events, daemon=True)
            reader.start()
            startup = time.monotonic()
            while episode.reason is None:
                if episode.started is not None:
                    episode.check_time()
                elif time.monotonic() - startup > self.config.get("agent_startup_timeout_s", 180):
                    raise TimeoutError("Codex did not request the RoboDojo task before startup timeout")
                while not events.empty():
                    event = events.get_nowait()
                    item = event.get("item", {})
                    if item.get("type") == "mcp_tool_call" and item.get("status") == "failed":
                        message = str(item.get("error") or item.get("result") or "No failure details")
                        if "approval" in message.lower() or item.get("tool") == "start_task":
                            raise RuntimeError("Codex MCP call failed: " + message)
                    if event.get("type") in ("turn.completed", "turn.failed", "error"):
                        if episode.started is None:
                            raise RuntimeError("Codex ended before receiving the RoboDojo task: " + str(event))
                        completed = event.get("type") == "turn.completed"
                        episode.stop("agent_stopped" if completed else "agent_error")
                if episode.reason is not None:
                    break
                if process.poll() is not None and not reader.is_alive() and events.empty():
                    raise RuntimeError(f"Codex exited unexpectedly ({process.returncode}); see agent-stderr.log")
                try:
                    request, response = pending.get(timeout=.05)
                except queue.Empty:
                    continue
                name = request["name"]
                try:
                    if name == "start_task":
                        episode.begin()
                        started_utc = datetime.now(timezone.utc).isoformat()
                        reply = {"content": [{"type": "text", "text": episode.agent_instructions()},
                                              *episode.first_observation]}
                        episode.check_time()
                    else:
                        reply = episode.tool(name, request.get("arguments", {}))
                except EpisodeStopped:
                    raise
                except Exception as exc:
                    reply = {"isError": True, "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}]}
                episode.check_time()
                response.put(reply)
        except EpisodeStopped:
            pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended_utc = datetime.now(timezone.utc).isoformat()
            # Never wait for pending inference to settle on an environment reset.
            # Reset must finish inside RoboDojoModel's 15-second join deadline.
            try:
                _stop(process)
            finally:
                closing.set()
                if reader is not None:
                    reader.join(timeout=1)
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=1)
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                stderr.close()
                result = {"model": model, "reasoning_effort": self.settings["reasoning_effort"],
                          "codex_path": self.settings["codex_path"],
                          "task_start_utc": started_utc, "task_end_utc": ended_utc,
                          "completed": completed, "error": error,
                          "exit_code": None if process is None else process.returncode,
                          "official_score": None,
                          "source_files_changed_during_episode": sources_before != _source_hashes(),
                          "cost_scope": "Raw telemetry; forced termination may leave usage incomplete"}
                (out / "agent-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
