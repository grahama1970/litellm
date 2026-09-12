"""SSE streaming response handler for OpenAI-compatible async streams.

Converts openai SDK async streams into FastAPI/Starlette StreamingResponse
objects with proper SSE formatting and graceful disconnect handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, AsyncIterator, Optional

from loguru import logger
from starlette.responses import StreamingResponse

DEFAULT_STREAM_HEARTBEAT_S = 15.0

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Build a named SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _is_done_chunk(chunk: str | bytes) -> bool:
    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="ignore")
    else:
        text = chunk
    return "data: [DONE]" in text


async def renumber_tool_call_deltas(raw: Any) -> Any:
    """Rewrite gpt-5.6 split-index tool_call deltas in a raw OAuth SSE stream.

    The upstream emits the id+name delta at one tool_calls index and the
    argument fragments at another; merge-by-index clients then assemble two
    broken calls. This pass renumbers identity-less argument fragments onto
    the most recent identity-bearing call, leaving every other byte intact.
    Mirrors the fix in ``_sse_generator`` for the raw-byte OAuth path.
    """
    import json as _json

    last_identity_index: int | None = None
    carry = ""

    async for chunk in raw:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else chunk
        carry += text
        parts = carry.split("\n\n")
        carry = parts.pop()
        for event in parts:
            if not event.strip():
                continue
            lines = event.splitlines()
            out_lines = []
            for line in lines:
                if line.startswith("data: ") and line[6:].strip().startswith("{"):
                    try:
                        payload = _json.loads(line[6:])
                    except Exception:
                        out_lines.append(line)
                        continue
                    if isinstance(payload, dict):
                        for choice in payload.get("choices") or []:
                            delta = choice.get("delta") if isinstance(choice, dict) else None
                            if not isinstance(delta, dict):
                                continue
                            for tc in delta.get("tool_calls") or []:
                                if not isinstance(tc, dict):
                                    continue
                                fn = tc.get("function") or {}
                                if tc.get("id") or fn.get("name"):
                                    last_identity_index = tc.get("index", 0)
                                elif (
                                    last_identity_index is not None
                                    and fn.get("arguments")
                                    and tc.get("index") != last_identity_index
                                ):
                                    tc["index"] = last_identity_index
                    out_lines.append("data: " + _json.dumps(payload))
                else:
                    out_lines.append(line)
            yield "\n".join(out_lines) + "\n\n"
    if carry.strip():
        yield carry


async def sse_liveness_wrapper(
    stream: AsyncIterator[str] | AsyncIterator[bytes],
    *,
    model: str = "",
    started_at: float | None = None,
    overall_timeout_s: float | None = None,
    heartbeat_interval_s: float = DEFAULT_STREAM_HEARTBEAT_S,
    progress_events: bool = False,
) -> AsyncIterator[str | bytes]:
    """Wrap an SSE stream with heartbeat and overall-budget enforcement.

    Provider stream reads can legitimately be silent for long reasoning spans.
    This wrapper keeps the client connection live with heartbeat comments while
    a pending provider read is in progress, and enforces a separate overall
    request budget so streams cannot hang forever.
    """
    started = started_at or time.monotonic()
    deadline = started + overall_timeout_s if overall_timeout_s and overall_timeout_s > 0 else None
    heartbeat_interval = max(0.001, float(heartbeat_interval_s or DEFAULT_STREAM_HEARTBEAT_S))
    iterator = stream.__aiter__()
    pending: asyncio.Task[Any] | None = None
    done_sent = False

    if progress_events:
        yield _sse_event(
            "started",
            {"model": model, "elapsed_ms": int((time.monotonic() - started) * 1000)},
        )

    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(iterator.__anext__())

            wait_timeout = heartbeat_interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pending
                    error_payload = {
                        "error": {
                            "message": f"stream_timeout: overall budget exceeded after {overall_timeout_s}s",
                            "type": "timeout_error",
                        }
                    }
                    yield _sse_event("error", error_payload) if progress_events else f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    done_sent = True
                    return
                wait_timeout = min(wait_timeout, remaining)

            done, _ = await asyncio.wait({pending}, timeout=wait_timeout)
            if not done:
                heartbeat = {
                    "model": model,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                }
                if progress_events:
                    yield _sse_event("heartbeat", heartbeat)
                else:
                    yield f": heartbeat {json.dumps(heartbeat)}\n\n"
                continue

            task = pending
            pending = None
            try:
                chunk = task.result()
            except StopAsyncIteration:
                if progress_events:
                    yield _sse_event(
                        "done",
                        {"model": model, "elapsed_ms": int((time.monotonic() - started) * 1000)},
                    )
                if not done_sent:
                    yield "data: [DONE]\n\n"
                return

            if _is_done_chunk(chunk):
                if progress_events:
                    yield _sse_event(
                        "done",
                        {"model": model, "elapsed_ms": int((time.monotonic() - started) * 1000)},
                    )
                yield chunk
                done_sent = True
                return
            yield chunk
    except Exception as exc:
        logger.error("SSE liveness wrapper interrupted: {}", exc)
        error_payload = {"error": {"message": f"stream_interrupted: {exc}", "type": "stream_error"}}
        yield _sse_event("error", error_payload) if progress_events else f"data: {json.dumps(error_payload)}\n\n"
        if not done_sent:
            yield "data: [DONE]\n\n"
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


def _compute_cost_event(usage: dict[str, Any], model: str) -> Optional[str]:
    """Build an SSE cost event from accumulated usage, or None if unavailable."""
    try:
        from chutes.middleware.pricing import estimate_cost_usd
    except ImportError:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

    cost_data = {
        "x-cost-usd": f"{cost:.6f}" if cost is not None else "unknown",
        "x-cost-input-tokens": str(int(prompt_tokens or 0)),
        "x-cost-output-tokens": str(int(completion_tokens or 0)),
    }
    return f"data: {json.dumps({'cost': cost_data})}\n\n"


async def _sse_generator(stream: Any, model: str = "") -> AsyncIterator[str]:
    """Iterate over an OpenAI async stream, yielding SSE-formatted lines.

    On provider disconnect or any mid-stream exception the error is logged
    and the stream is closed cleanly with a ``[DONE]`` sentinel.

    After the stream completes, if any chunk contained a ``usage`` field,
    a ``cost`` SSE event is emitted with ``x-cost-usd``,
    ``x-cost-input-tokens``, and ``x-cost-output-tokens``.
    """
    last_usage: dict[str, Any] = {}
    stream_model: str = model
    # gpt-5.6 streaming quirk (2026-09-12): id+name delta and argument
    # fragments can carry different tool_call indices. Renumber identity-less
    # argument fragments onto the most recent identity-bearing call so any
    # merge-by-index client (the OpenAI contract, e.g. Pi's tau-scillm
    # bridge) assembles one well-formed call instead of two broken ones.
    last_identity_index: int | None = None
    try:
        async for chunk in stream:
            payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            # Track usage from the final chunk (OpenAI sends it when
            # stream_options.include_usage is true).
            if isinstance(payload, dict):
                if payload.get("usage"):
                    last_usage = payload["usage"]
                if payload.get("model"):
                    stream_model = payload["model"]
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if tc.get("id") or fn.get("name"):
                            last_identity_index = tc.get("index", 0)
                        elif (
                            last_identity_index is not None
                            and fn.get("arguments")
                            and tc.get("index") != last_identity_index
                        ):
                            tc["index"] = last_identity_index
            yield f"data: {json.dumps(payload)}\n\n"
    except Exception as exc:
        logger.error("Stream interrupted: {}", exc)
        # Signal the error to clients instead of silently closing
        error_payload = {
            "error": {
                "message": f"stream_interrupted: {exc}",
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        # Emit cost event if we accumulated usage data
        if last_usage:
            cost_event = _compute_cost_event(last_usage, stream_model)
            if cost_event:
                yield cost_event
        yield "data: [DONE]\n\n"


async def stream_response(
    stream: Any,
    model: str = "",
    *,
    started_at: float | None = None,
    overall_timeout_s: float | None = None,
    heartbeat_interval_s: float = DEFAULT_STREAM_HEARTBEAT_S,
    progress_events: bool = False,
) -> StreamingResponse:
    """Convert an OpenAI ``AsyncStream`` into a Starlette ``StreamingResponse``.

    Parameters
    ----------
    stream:
        An async iterable of chat completion chunks (e.g. from
        ``openai.AsyncOpenAI.chat.completions.create(stream=True)``).
    model:
        The model name used for this request (for cost estimation).

    Returns
    -------
    StreamingResponse
        A streaming HTTP response with ``text/event-stream`` content type
        and appropriate SSE headers.
    """
    return StreamingResponse(
        sse_liveness_wrapper(
            _sse_generator(stream, model=model),
            model=model,
            started_at=started_at,
            overall_timeout_s=overall_timeout_s,
            heartbeat_interval_s=heartbeat_interval_s,
            progress_events=progress_events,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def collect_response(stream: Any) -> dict[str, Any]:
    """Collect all chunks from an OpenAI async stream into a single response dict.

    Assembles ``delta`` content fields across chunks and returns the final
    aggregated response suitable for non-streaming callers.

    Parameters
    ----------
    stream:
        An async iterable of chat completion chunks.

    Returns
    -------
    dict
        The assembled response with concatenated content and final metadata.
    """
    result: dict[str, Any] = {}
    collected_content: list[str] = []
    collected_reasoning_content: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    try:
        async for chunk in stream:
            data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk

            # Preserve top-level metadata from every chunk (last wins).
            for key in ("id", "object", "created", "model", "system_fingerprint"):
                if key in data and data[key] is not None:
                    result[key] = data[key]

            if "usage" in data and data["usage"] is not None:
                result["usage"] = data["usage"]

            choices = data.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            # Accumulate text content.
            if delta.get("content"):
                collected_content.append(delta["content"])
            if delta.get("reasoning_content"):
                collected_reasoning_content.append(delta["reasoning_content"])

            # Accumulate tool calls.
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                # gpt-5.6 streaming quirk (scillm 2026-09-12): the id+name
                # delta and the argument fragments can carry DIFFERENT
                # tool_call indices. A delta with no id and no name but with
                # arguments belongs to the most recent open call, not to a
                # new call keyed by its renumbered index. Identity-bearing
                # deltas (id or name) still open new entries, so genuinely
                # parallel calls are unaffected.
                has_identity = bool(tc.get("id") or fn.get("name"))
                if (
                    not has_identity
                    and fn.get("arguments")
                    and tool_calls_by_index
                ):
                    tool_calls_by_index[max(tool_calls_by_index)]["function"][
                        "arguments"
                    ] += fn["arguments"]
                    continue
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_by_index[idx]
                if fn.get("name"):
                    entry["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]

            # Capture finish_reason from the final chunk.
            if choice.get("finish_reason"):
                result["_finish_reason"] = choice["finish_reason"]

    except Exception as exc:
        logger.error("Error collecting stream: {}", exc)

    # Build the assembled response.
    message: dict[str, Any] = {"role": "assistant"}
    if collected_content:
        message["content"] = "".join(collected_content)
    elif collected_reasoning_content:
        # Some OpenCode/Kimi routes stream useful visible output under
        # reasoning_content. Preserve it as fallback content so helper callers
        # do not receive an empty response.
        message["content"] = "".join(collected_reasoning_content)
    if collected_reasoning_content:
        message["reasoning_content"] = "".join(collected_reasoning_content)
    if tool_calls_by_index:
        message["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]

    result["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": result.pop("_finish_reason", "stop"),
        }
    ]

    return result
