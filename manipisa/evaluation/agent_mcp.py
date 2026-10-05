"""Minimal stdio MCP client bridge to a single local simulation process."""
from __future__ import annotations
import argparse
import json
import math
import sys
import urllib.request

TOOLS = [
    {"name": "start_task", "description": "Receive the robot task and initial state/RGB. Call exactly once before other tools.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "observe", "description": "Read current robot/object state and RGB images. Physics remains paused.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "execute_python", "description": "Execute Python in the persistent simulation control namespace described in the task. Use step(n) to advance physics. Calls that advance physics return current RGB image blocks and compact state after execution, including partial motion before ordinary errors. Return between action segments to inspect images. Print focused diagnostics.",
     "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}},
    {"name": "read_api", "description": "Read allowed SDK or instruction API reference text. Use name='index' to list documents.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}},
]

for tool in TOOLS:
    tool["annotations"] = {"readOnlyHint": tool["name"] != "execute_python", "destructiveHint": False,
                           "openWorldHint": False, "idempotentHint": tool["name"] in ("observe", "read_api")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--timeout", type=float, default=660, help="HTTP timeout supplied by the episode host")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        try:
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                          "serverInfo": {"name": "manipisa-benchmark", "version": "0.2"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "ping":
                result = {}
            elif method == "tools/call":
                body = json.dumps(request["params"]).encode()
                http = urllib.request.Request(args.endpoint + "/tool", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.token})
                with urllib.request.urlopen(http, timeout=args.timeout) as response:
                    result = json.load(response)
            else:
                result = {}
            answer = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            answer = {"jsonrpc": "2.0", "id": request["id"], "result":
                      {"isError": True, "content": [{"type": "text", "text": str(exc)}]}}
        print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    main()
