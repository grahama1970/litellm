"""gpt-5.6 streamed tool-call index quirk: arguments must merge into the open call.

Captured 2026-09-12 from a live gpt-5.6 stream through scillm: the id+name
delta arrives with tool_calls[].index=0 while every arguments fragment arrives
with index=1, id="", name="". Merge-by-index produced two broken entries
(id+name with empty args; nameless with the real args) — the exact split that
broke the tau-scillm reviewer revive. The fix merges identity-less argument
deltas into the most recent open call.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from scillm.proxy.streaming import collect_response


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1789232922,
        "model": "gpt-5.6-sol",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


CAPTURED_GPT56_SEQUENCE: list[dict[str, Any]] = [
    _chunk({
        "role": "assistant",
        "tool_calls": [{
            "index": 0,
            "id": "call_jrNtEoRoYH5TpweneecdAVzc",
            "type": "function",
            "function": {"name": "read_file", "arguments": ""},
        }],
    }),
    _chunk({
        "tool_calls": [{
            "index": 1,
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": "{\""},
        }],
    }),
    *[
        _chunk({"tool_calls": [{"index": 1, "function": {"arguments": part}}]})
        for part in ("path", "\":", " \"/", "tmp", "/x", ".md", "\"}")
    ],
    _chunk({}, finish="tool_calls"),
]


def _run(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    async def gen() -> Any:
        for c in chunks:
            yield c

    return asyncio.run(collect_response(gen()))


def test_gpt56_renumbered_argument_fragments_merge_into_open_call() -> None:
    result = _run(CAPTURED_GPT56_SEQUENCE)
    calls = result["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1, calls
    assert calls[0]["id"] == "call_jrNtEoRoYH5TpweneecdAVzc"
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path": "/tmp/x.md"}'


def test_genuine_parallel_calls_stay_separate() -> None:
    result = _run([
        _chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                 "function": {"name": "first", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                 "function": {"name": "second", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ])
    calls = result["choices"][0]["message"]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert calls[1]["function"]["arguments"] == "{}"


def test_fail_before_fix_shape_is_two_split_entries_under_pure_index_merge() -> None:
    # Non-regression documentation: the captured sequence with merge-by-index
    # alone yields 2 entries (the bug). Guards against reintroducing index-only
    # keying: the merged result above must stay 1 entry.
    result = _run(CAPTURED_GPT56_SEQUENCE)
    assert not any(
        (not c["function"]["name"]) or (not c["function"]["arguments"])
        for c in result["choices"][0]["message"]["tool_calls"]
    )
