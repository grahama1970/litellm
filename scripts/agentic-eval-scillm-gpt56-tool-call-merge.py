#!/usr/bin/env python3
"""Retained eval for the gpt-5.6 split-index tool-call delta fix (2026-09-12).

Modes:
  replay — deterministic: the captured live SSE sequence must assemble into
           exactly ONE well-formed call through both fixed paths
           (collect_response and renumber_tool_call_deltas).
  live   — real-world: drive a streaming gpt-5.6 tool-call request through the
           RUNNING scillm proxy and assemble deltas merge-by-index (the exact
           consumer contract Pi's tau-scillm bridge uses); require one
           well-formed call (non-empty id, name, and arguments).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCILLM_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCILLM_SRC))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-captured", "object": "chat.completion.chunk", "created": 1789232922,
        "model": "gpt-5.6-sol",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


CAPTURED: list[dict[str, Any]] = [
    _chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_captured_1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]}),
    _chunk({"tool_calls": [{"index": 1, "id": "", "type": "function", "function": {"name": "", "arguments": "{\""}}]}),
    *[_chunk({"tool_calls": [{"index": 1, "function": {"arguments": p}}]}) for p in ("path", "\":", " \"/", "tmp", "/x", ".md", "\"}")],
    _chunk({}, finish="tool_calls"),
]


def _wellformed(calls: list[dict[str, Any]]) -> bool:
    return (
        len(calls) == 1
        and bool(calls[0].get("id"))
        and bool(calls[0]["function"]["name"])
        and bool(calls[0]["function"]["arguments"])
    )


def mode_replay() -> int:
    from scillm.proxy.streaming import collect_response, renumber_tool_call_deltas

    async def gen() -> Any:
        for c in CAPTURED:
            yield c

    assembled = asyncio.run(collect_response(gen()))
    calls = assembled["choices"][0]["message"]["tool_calls"]

    async def raw_gen() -> Any:
        for c in CAPTURED:
            yield (f"data: {json.dumps(c)}\n\n").encode()

    async def consume() -> list[dict[str, Any]]:
        by: dict[int, dict[str, str]] = {}
        async for block in renumber_tool_call_deltas(raw_gen()):
            for line in block.splitlines():
                if not line.startswith("data: {"):
                    continue
                d = json.loads(line[6:])
                for ch in d.get("choices") or []:
                    for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                        i = tc.get("index", 0)
                        fn = tc.get("function") or {}
                        e = by.setdefault(i, {"id": "", "name": "", "args": ""})
                        e["id"] += tc.get("id") or ""
                        e["name"] += fn.get("name") or ""
                        e["args"] += fn.get("arguments") or ""
        return [{"id": e["id"], "function": {"name": e["name"], "arguments": e["args"]}} for _, e in sorted(by.items())]

    forwarded = asyncio.run(consume())
    if _wellformed(calls) and _wellformed(forwarded):
        print("REPLAY_PASS: captured gpt-5.6 sequence assembles one well-formed call on both fixed paths")
        return 0
    print(f"REPLAY_FAIL: collect={calls} forwarded={forwarded}")
    return 1


def _proxy_key() -> str:
    out = subprocess.run(
        ["docker", "inspect", "docker-scillm-proxy-1", "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    for line in out.splitlines():
        for name in ("SCILLM_MASTER_KEY", "LITELLM_MASTER_KEY", "SCILLM_PROXY_KEY"):
            if line.startswith(f"{name}=") and line[len(name) + 1:]:
                return line[len(name) + 1:].strip()
    raise SystemExit("no proxy key found in docker-scillm-proxy-1 env")


_OUT_PATH = ["gpt56-live-readback.json"]


def mode_live() -> int:
    import httpx

    body = {
        "model": "gpt-5.6", "stream": True,
        "messages": [{"role": "user", "content": "Read the file /tmp/eval-probe.md using the read_file tool. Call the tool now."}],
        "tools": TOOLS,
    }
    with httpx.Client(timeout=90.0) as client:
        with client.stream(
            "POST", "http://127.0.0.1:4001/v1/chat/completions",
            headers={"Authorization": f"Bearer {_proxy_key()}", "X-Caller-Skill": "scillm-gpt56-merge-eval", "Content-Type": "application/json"},
            json=body,
        ) as resp:
            resp.raise_for_status()
            by: dict[int, dict[str, str]] = {}
            finish = None
            for line in resp.iter_lines():
                if not line.startswith("data: {"):
                    continue
                d = json.loads(line[6:])
                for ch in d.get("choices") or []:
                    for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                        i = tc.get("index", 0)
                        fn = tc.get("function") or {}
                        e = by.setdefault(i, {"id": "", "name": "", "args": ""})
                        e["id"] += tc.get("id") or ""
                        e["name"] += fn.get("name") or ""
                        e["args"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    calls = [{"id": e["id"], "function": {"name": e["name"], "arguments": e["args"]}} for _, e in sorted(by.items())]
    readback = {"schema": "scillm.gpt56_live_readback.v1", "finish": finish,
                "call_count": len(calls), "calls": calls}
    Path(_OUT_PATH[0]).write_text(json.dumps(readback, indent=1))
    if finish == "tool_calls" and _wellformed(calls):
        print(f"LIVE_PASS: running proxy assembled one well-formed call (id={calls[0]['id'][:18]}...)")
        return 0
    print(f"LIVE_FAIL: finish={finish} calls={calls}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["replay", "live"])
    parser.add_argument("--out", default="gpt56-live-readback.json")
    args = parser.parse_args()
    _OUT_PATH[0] = args.out
    return {"replay": mode_replay, "live": mode_live}[args.mode]()


if __name__ == "__main__":
    raise SystemExit(main())
