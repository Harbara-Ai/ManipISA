"""Loopback HTTP to shared-file transport for one owned WSL agent."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid


class FileRelay:
    def __init__(self, spool, token, heartbeat, stop, *, heartbeat_max_age_s=20.0):
        self.spool = Path(spool)
        self.requests = self.spool / "requests"
        self.responses = self.spool / "responses"
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        self.heartbeat = Path(heartbeat)
        self.stop = Path(stop)
        self.max_age = heartbeat_max_age_s
        self.closing = threading.Event()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/otel/" + token:
                    kind = "otel"
                elif self.path == "/tool" and self.headers.get("Authorization") == "Bearer " + token:
                    kind = "tool"
                else:
                    self.send_error(403)
                    return
                if relay.should_close():
                    self.close_connection = True
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 64 * 1024 * 1024:
                        raise ValueError("Invalid body length")
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError):
                    self.send_error(400)
                    return
                request_id = uuid.uuid4().hex
                destination = relay.requests / (request_id + ".json")
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps({"kind": kind, "payload": payload}), encoding="utf-8")
                temporary.replace(destination)
                if kind == "otel":
                    reply = {}
                else:
                    response_path = relay.responses / (request_id + ".json")
                    while not relay.should_close():
                        if response_path.is_file():
                            reply = json.loads(response_path.read_text(encoding="utf-8"))
                            response_path.unlink()
                            break
                        relay.closing.wait(0.05)
                    else:
                        # Close the socket without a terminal evaluator reply.
                        self.close_connection = True
                        return
                    if relay.should_close():
                        self.close_connection = True
                        return
                data = json.dumps(reply).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        # Mirrored networking on this host intercepts 127.0.0.1; 127.0.0.2 is
        # still private Linux loopback and has been verified without firewall changes.
        self.server = ThreadingHTTPServer(("127.0.0.2", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = "http://127.0.0.2:" + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def should_close(self):
        if self.closing.is_set() or self.stop.exists():
            return True
        try:
            return time.time() - self.heartbeat.stat().st_mtime > self.max_age
        except FileNotFoundError:
            return True

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.closing.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
