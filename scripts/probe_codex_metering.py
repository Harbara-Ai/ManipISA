"""One non-robot model request to verify native Codex telemetry.

Stores OTel bodies only on loopback in a fresh local directory. Never reads
authentication files or redirects the model provider. No robot task is sent.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    out = ROOT / "artifacts" / ("codex-metering-probe-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    work = out / "workspace"
    work.mkdir()
    packets = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            index = len(packets)
            (out / f"otel-{index:04d}.bin").write_bytes(data)
            packets.append({"path": self.path, "content_type": self.headers.get("Content-Type"), "bytes": len(data)})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/logs"
    command = [shutil.which("codex") or "codex", "exec", "--ignore-user-config", "--ephemeral",
               "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--cd", str(work),
               "--model", "gpt-6.1-sol", "-c", 'model_reasoning_effort="high"',
               "-c", 'web_search="disabled"', "--disable", "shell_tool", "--disable", "multi_agent",
               "--disable", "plugins", "-c", 'otel.log_user_prompt=false',
               "-c", f'otel.exporter={{otlp-http={{endpoint="{endpoint}",protocol="json"}}}}',
               "Reply with exactly METERING_OK. Do not call any tools. This is a model connectivity and native usage probe."]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    try:
        with (out / "events.jsonl").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.timeout, cwd=work)
                result = {"exit_code": proc.returncode, "timeout": False}
            except subprocess.TimeoutExpired:
                result = {"exit_code": None, "timeout": True}
    finally:
        server.shutdown()
        server.server_close()
    result["otel_packets"] = packets
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
