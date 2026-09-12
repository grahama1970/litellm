"""FastAPI application for the scillm proxy.

Thin OpenAI-compatible proxy that routes through middleware chain → router → streaming.
FastAPI application for the scillm proxy (~350 lines).
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import yaml
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger

from scillm.proxy.config import ProxyConfig, load_config
from scillm.proxy.errors import ProxyError, proxy_error_handler
from scillm.proxy.middleware import BaseMiddleware, MiddlewareChain, MiddlewareReject
from scillm.proxy import openai_images
from scillm.proxy.openai_images import OPENAI_IMAGE_MODEL_CONFIGS
from scillm.proxy.providers.opencode_go import (
    OPENCODE_GO_PROVIDER,
    describe_opencode_go_model,
    is_opencode_go_model,
    list_opencode_go_models_from_cli,
    list_opencode_go_models_from_cli_sync,
    list_opencode_go_models_from_server,
    list_opencode_go_models_from_server_sync,
    list_opencode_go_models_from_zen_sync,
    opencode_go_input_capabilities,
    static_opencode_go_models,
)
from scillm.proxy.providers.claude_models import claude_catalog_payload, static_claude_model_ids
from scillm.proxy.providers.codex_models import codex_catalog_payload, discover_codex_models, resolve_codex_model
from scillm.proxy.router import Router
from scillm.proxy.router import ProxyError as RouterProxyError
from scillm.proxy.streaming import DEFAULT_STREAM_HEARTBEAT_S, SSE_HEADERS, _sse_generator, renumber_tool_call_deltas, sse_liveness_wrapper
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Globals (populated during lifespan)
# ---------------------------------------------------------------------------
_config: ProxyConfig | None = None
_router: Router | None = None
_middleware_chain: MiddlewareChain | None = None
_start_time: float = 0.0
_embedding_client: httpx.AsyncClient | None = None

EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8602")
MODELS_DEV_API_URL = os.environ.get("MODELS_DEV_API_URL", "https://models.dev/api.json")
MODELS_DEV_CACHE_TTL_S = float(os.environ.get("MODELS_DEV_CACHE_TTL_S", "900"))
_models_dev_cache: dict[str, Any] | None = None
_models_dev_cache_ts: float = 0.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(request: Request) -> str | None:
    """Validate Bearer token. Returns None if OK, error message if not."""
    if not _config or not _config.general.master_key:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return "Missing Bearer token"
    token = auth[7:].strip()
    if token != _config.general.master_key:
        return "Invalid API key"
    return None


def _require_caller_skill(request: Request) -> str:
    """Return the required X-Caller-Skill value or fail before provider work."""
    caller_skill = request.headers.get("x-caller-skill", "").strip()
    if not caller_skill:
        raise ProxyError(400, "X-Caller-Skill header is required", "caller_skill_required")
    return caller_skill


def _is_empty_length_response(response_dict: dict[str, Any]) -> bool:
    """Return true when a provider spent its budget without visible output."""
    choices = response_dict.get("choices", [])
    if not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") != "length":
        return False

    message = choice.get("message") or {}
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return False
    if content is None:
        return True
    if isinstance(content, str) and not content.strip():
        return True
    return False


def _is_empty_zero_usage_response(response_dict: dict[str, Any]) -> bool:
    """Return true for empty responses that claim no prompt reached the model."""
    usage = response_dict.get("usage") or {}
    if any((usage.get(key) or 0) != 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        return False

    choices = response_dict.get("choices", [])
    if not choices:
        return True
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        return False

    content = message.get("content")
    if content is None:
        return True
    if isinstance(content, str) and not content.strip():
        return True
    return False


# ---------------------------------------------------------------------------
# Request Validation (fail loudly on common mistakes)
# ---------------------------------------------------------------------------

# Deprecated Chutes models. Do not remap these to aliases; Chutes calls must use
# exact live model IDs selected from current inventory.
_DEPRECATED_MODELS: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "Model removed from Chutes. Use ops-chutes to choose an exact live model ID.",
    "deepseek-ai/DeepSeek-V3.1-TEE": "Model deprecated. Use ops-chutes to choose an exact live model ID.",
    "deepseek-ai/DeepSeek-V3-0324": "Model deprecated. Use ops-chutes to choose an exact live model ID.",
    "deepseek-ai/DeepSeek-V3-0324-TEE": "Model deprecated. Use ops-chutes to choose an exact live model ID.",
}

# Env vars that agents might set with model names — validate at startup
_MODEL_ENV_VARS = [
    "CHUTES_TEXT_MODEL",
    "CHUTES_VLM_MODEL",
    "CHUTES_MODEL_ID",
    "CHUTES_MODEL",
]


def _check_env_for_deprecated_models() -> None:
    """Warn at startup if env vars reference deprecated models."""
    for var in _MODEL_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val in _DEPRECATED_MODELS:
            logger.error(
                "ENV VAR DEPRECATED: {}='{}' — {}. "
                "Update .env to an exact live Chutes model ID from ops-chutes.",
                var, val, _DEPRECATED_MODELS[val],
            )

# Direct provider model names that should use aliases instead
_DIRECT_MODEL_PATTERNS = {
    "claude-": "Use an explicit Claude profile such as 'claude-sonnet'; OAuth profiles do not cross-fallback.",
    "gpt-4": "Use an explicit Codex/OpenAI profile such as 'gpt-5.5' or 'codex-vision'.",
    "gemini-": "Use 'gemini-flash' or 'gemini-flash-high' for Gemini API-key routes.",
}

_CHUTES_PROVIDER_PREFIXES = (
    "deepseek-ai/",
    "qwen/",
    "moonshotai/",
    "tngtech/",
    "zai-org/",
)

_CODEX_OAUTH_PREFIXES = ("codex", "gpt", "o1", "o3", "o4")
_UNSUPPORTED_CHATGPT_CODEX_OAUTH_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.2-codex",
}
# Known good model aliases (checked at startup from config)
_VALID_MODEL_ALIASES: set[str] = set()

_RETIRED_MODEL_PREFIXES = ("text-",)

_REVIEW_FANOUT_MODEL_PREFERENCE = (
    "gpt-5.5",
    "oc-kimi",
    "oc-glm",
    "oc-deepseek",
    "oc-qwen",
)

_BATCH_FORWARD_KEYS = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "response_format",
    "tools",
    "tool_choice",
    "seed",
    "logprobs",
    "top_logprobs",
)

_IMAGE_GENERATION_MODELS: dict[str, dict[str, Any]] = {
    "z-image-turbo": {
        "provider": "chutes",
        "slug": "chutes-z-image-turbo",
        "endpoint_env": "CHUTES_Z_IMAGE_TURBO_URL",
        "default_endpoint": "https://chutes-z-image-turbo.chutes.ai/generate",
        "output_content_type": "image/png",
        "supports_batch": True,
        "limits": {
            "prompt_min_length": 3,
            "prompt_max_length": 1200,
            "width_min": 576,
            "width_max": 2048,
            "height_min": 576,
            "height_max": 2048,
            "num_inference_steps_min": 1,
            "num_inference_steps_max": 100,
            "guidance_scale_min": 0.0,
            "guidance_scale_max": 5.0,
            "shift_min": 1.0,
            "shift_max": 10.0,
            "max_sequence_length_min": 256,
            "max_sequence_length_max": 2048,
            "seed_min": 0,
            "seed_max": 2**32 - 1,
        },
    },
    "chutes-z-image-turbo": {
        "alias_for": "z-image-turbo",
    },
    "chutes/z-image-turbo": {
        "alias_for": "z-image-turbo",
    },
    **OPENAI_IMAGE_MODEL_CONFIGS,
}

_DEFAULT_MODEL_POOLS: dict[str, dict[str, Any]] = {
    "qra-chutes-only": {
        "description": "QRA extraction through Chutes only; use when fallback lanes fail schema or return empty content.",
        "strategy": "single_provider",
        "lanes": [
            {
                "name": "qra-chutes-deepseek",
                "provider": "chutes",
                "model": os.environ.get("SCILLM_QRA_CHUTES_MODEL", ""),
                "weight": 1,
                "max_concurrency": 5,
                "timeout": 420.0,
                "require_hot_chutes": True,
            },
        ],
    },
    "qra-deepseek-pool": {
        "description": "QRA extraction pool using independent Chutes and OpenCode Go lanes.",
        "strategy": "weighted_round_robin",
        "lanes": [
            {
                "name": "qra-chutes-deepseek",
                "provider": "chutes",
                "model": os.environ.get("SCILLM_QRA_CHUTES_MODEL", ""),
                "weight": int(os.environ.get("SCILLM_QRA_CHUTES_WEIGHT", "3")),
                "max_concurrency": 5,
                "timeout": 420.0,
                "require_hot_chutes": True,
            },
            {
                "name": "qra-opencode-go-deepseek-v4-flash",
                "provider": "opencode-go",
                "model": os.environ.get("SCILLM_QRA_OPENCODE_GO_MODEL", "opencode-go/deepseek-v4-flash"),
                "weight": int(os.environ.get("SCILLM_QRA_OPENCODE_GO_WEIGHT", "2")),
                "max_concurrency": int(os.environ.get("SCILLM_QRA_OPENCODE_GO_MAX_CONCURRENCY", "4")),
                "timeout": float(os.environ.get("SCILLM_QRA_OPENCODE_GO_TIMEOUT_S", "620")),
                "require_hot_chutes": False,
            },
        ],
    }
}


def _is_direct_chutes_model(model: str) -> bool:
    """Return true for direct Chutes provider/model ids."""
    return model.lower().startswith(_CHUTES_PROVIDER_PREFIXES)


def _is_chutes_deployment(group_name: str, model: str, api_base: str | None = None) -> bool:
    """Return true when a configured deployment is intended for Chutes."""
    lower_group = group_name.lower()
    lower_model = str(model or "").lower()
    lower_base = str(api_base or "").lower()
    return (
        "chutes" in lower_base
        or lower_group.startswith("chutes-")
        or lower_model.startswith(_CHUTES_PROVIDER_PREFIXES)
    )


def _chutes_config_inventory(available_models: list[str]) -> dict[str, Any]:
    """Compare configured Chutes routing state with live provider inventory."""
    available = set(available_models)
    configured_groups: dict[str, list[str]] = {}
    configured_models: set[str] = set()
    fallback_references: dict[str, list[str]] = {}
    alias_resolutions: dict[str, str] = {}

    if _config is None:
        return {
            "status": "proxy_not_ready",
            "configured_groups": {},
            "configured_models": [],
            "configured_available_models": [],
            "configured_unavailable_models": [],
            "fallback_references": {},
            "unavailable_fallback_targets": [],
            "alias_resolutions": {},
        }

    for group_name, group in _config.model_groups.items():
        group_models = [
            dep.model
            for dep in group.deployments
            if _is_chutes_deployment(group_name, dep.model, dep.api_base)
        ]
        if group_models:
            configured_groups[group_name] = group_models
            configured_models.update(group_models)

    for source, targets in _config.fallbacks.items():
        if source in configured_groups or source in configured_models or _is_direct_chutes_model(source):
            fallback_references[source] = list(targets)
            continue
        for target in targets:
            if target in configured_groups or target in configured_models or _is_direct_chutes_model(target):
                fallback_references[source] = list(targets)
                break

    def fallback_target_available(target: str) -> bool:
        if target in configured_groups:
            return any(model in available for model in configured_groups[target]) if available else True
        if target in configured_models:
            return target in available if available else True
        return target in available if available else True

    unavailable_fallback_targets = sorted(
        {
            target
            for targets in fallback_references.values()
            for target in targets
            if not fallback_target_available(target)
        }
    )

    for alias, target in _config.aliases.items():
        if (
            alias in configured_groups
            or target in configured_groups
            or target in configured_models
            or _is_direct_chutes_model(alias)
            or _is_direct_chutes_model(target)
        ):
            resolved = target
            if target in configured_groups and configured_groups[target]:
                resolved = configured_groups[target][0]
            alias_resolutions[alias] = resolved

    configured_model_list = sorted(configured_models)
    unavailable_models = sorted(model for model in configured_model_list if available and model not in available)
    status = "ok" if not unavailable_models and not unavailable_fallback_targets else "config_drift"
    return {
        "status": status,
        "configured_groups": configured_groups,
        "configured_models": configured_model_list,
        "configured_available_models": sorted(model for model in configured_model_list if not available or model in available),
        "configured_unavailable_models": unavailable_models,
        "fallback_references": fallback_references,
        "unavailable_fallback_targets": unavailable_fallback_targets,
        "alias_resolutions": alias_resolutions,
    }


def _is_codex_oauth_model(model: str) -> bool:
    """Return true for model ids handled by the Codex OAuth router."""
    normalized = model.lower()
    return (
        normalized.startswith(_CODEX_OAUTH_PREFIXES)
        and "/" not in model
        and normalized not in _UNSUPPORTED_CHATGPT_CODEX_OAUTH_MODELS
    )


def _codex_oauth_available() -> bool:
    """Return true when Codex OAuth credentials are available."""
    try:
        from scillm.proxy.providers.auth import is_codex_available
        return is_codex_available()
    except Exception:
        return False


def _claude_oauth_available() -> bool:
    """Return true when Claude OAuth credentials are available."""
    try:
        from scillm.proxy.providers.auth import is_anthropic_available
        return is_anthropic_available()
    except Exception:
        return False


def _provider_for_model_selector(model: str) -> str | None:
    lower = model.lower()
    if lower.startswith("claude") and "/" not in model:
        return "claude"
    if _is_codex_oauth_model(model):
        return "codex"
    if is_opencode_go_model(model):
        return OPENCODE_GO_PROVIDER
    return None


def _opencode_go_catalog_payload_for_validation(*, refresh: bool = False) -> dict[str, Any]:
    source = "static_registry"
    errors: list[str] = []
    models: list[str] = []
    if refresh:
        try:
            models = list_opencode_go_models_from_cli_sync(refresh=refresh)
            if models:
                source = "opencode_cli"
        except Exception as exc:
            errors.append(f"cli: {str(exc)[:500]}")
            models = []
        if not models:
            try:
                models = list_opencode_go_models_from_server_sync()
                if models:
                    source = "opencode_server"
            except Exception as exc:
                errors.append(f"server: {str(exc)[:500]}")
                models = []
        if not models:
            try:
                models = list_opencode_go_models_from_zen_sync()
                if models:
                    source = "zen_gateway"
            except Exception as exc:
                errors.append(f"zen: {str(exc)[:500]}")
                models = []
    if not models:
        models = static_opencode_go_models()
    return {
        "provider": OPENCODE_GO_PROVIDER,
        "pattern": "opencode-go/*",
        "source": source,
        "available": bool(_config and _config.opencode_go_api_key),
        "live": source in {"opencode_cli", "opencode_server", "zen_gateway"},
        "error": "; ".join(errors) if errors else None,
        "errors": errors,
        "models": [
            {
                **describe_opencode_go_model(model, key_configured=bool(_config and _config.opencode_go_api_key)),
                "reasoning_efforts": [],
                "reasoning_source": "not_advertised",
            }
            for model in models
        ],
    }


def _provider_model_catalog(provider: str, *, refresh: bool = False) -> dict[str, Any]:
    if provider == "claude":
        return {
            "provider": "claude",
            "pattern": "claude-*",
            **claude_catalog_payload(refresh=refresh),
        }
    if provider == "codex":
        catalog = codex_catalog_payload()
        return {
            "provider": "codex",
            "pattern": "gpt-* | codex-* | o1* | o3* | o4*",
            "source": catalog["source"],
            "available": _codex_oauth_available(),
            "live": bool(catalog.get("available")),
            "error": None if catalog.get("available") else "codex_models_cache_unavailable",
            "models": catalog["models"],
        }
    if provider == OPENCODE_GO_PROVIDER:
        return {
            "provider": OPENCODE_GO_PROVIDER,
            "pattern": "opencode-go/*",
            **_opencode_go_catalog_payload_for_validation(refresh=refresh),
        }
    return {
        "provider": provider,
        "pattern": None,
        "source": "unknown",
        "available": False,
        "live": False,
        "error": "unknown_provider",
        "models": [],
    }


def _model_ids_from_catalog(catalog: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in catalog.get("models", []):
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            ids.append(row["id"])
    return ids


def _model_not_available_error(model: str, provider: str, catalog: dict[str, Any]) -> ProxyError:
    ids = _model_ids_from_catalog(catalog)
    suggestions = difflib.get_close_matches(model.lower(), [item.lower() for item in ids], n=5, cutoff=0.35)
    lower_to_original = {item.lower(): item for item in ids}
    resolved_suggestions = [lower_to_original[item] for item in suggestions]
    details = {
        "provider": provider,
        "requested_model": model,
        "pattern": catalog.get("pattern"),
        "catalog_source": catalog.get("source"),
        "catalog_live": catalog.get("live"),
        "auth_status": "configured" if catalog.get("available") else "unavailable",
        "available_models": catalog.get("models", []),
        "suggestions": resolved_suggestions,
        "refresh_hint": (
            "/v1/scillm/models?refresh_provider_models=true"
            if provider in {"claude", OPENCODE_GO_PROVIDER}
            else "/v1/scillm/models"
        ),
    }
    msg = f"Model '{model}' is not available for provider '{provider}'."
    if resolved_suggestions:
        msg += f" Did you mean: {', '.join(resolved_suggestions)}?"
    raise ProxyError(400, msg, "model_not_available", details=details)


def _validate_dynamic_provider_model(model: str) -> bool:
    """Validate dynamically-routed provider models.

    Returns true when the model belongs to a dynamic provider family and has
    been accepted or rejected. False means the caller should continue with the
    legacy configured-alias validation path.
    """
    provider = _provider_for_model_selector(model)
    if provider is None:
        return False

    if provider == "codex":
        catalog = _provider_model_catalog(provider)
        discovered = discover_codex_models()
        if discovered and resolve_codex_model(model, discovered) is None:
            _model_not_available_error(model, provider, catalog)
        return bool(_codex_oauth_available())

    if provider == "claude" and model in static_claude_model_ids():
        return True
    if provider == OPENCODE_GO_PROVIDER and model in static_opencode_go_models():
        return True

    catalog = _provider_model_catalog(provider, refresh=True)
    ids = set(_model_ids_from_catalog(catalog))
    if model not in ids:
        _model_not_available_error(model, provider, catalog)
    return True


def _message_has_multimodal_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and (
                part.get("type") in {"image_url", "image", "document"}
                or "inlineData" in part
            )
            for part in content
        )
    return isinstance(content, dict) and (
        "image_url" in content
        or "inlineData" in content
        or content.get("type") in {"image_url", "image", "document"}
    )


def _count_multimodal_parts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"image_count": 0, "document_count": 0, "inline_data_count": 0}
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [content] if isinstance(content, dict) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"image_url", "image"} or "image_url" in part:
                counts["image_count"] += 1
            if part_type == "document":
                counts["document_count"] += 1
            if "inlineData" in part:
                counts["inline_data_count"] += 1
    return counts


def _normalize_reasoning_effort(body: dict[str, Any]) -> str | None:
    effort = body.get("reasoning_effort")
    if effort is None:
        reasoning = body.get("reasoning")
        if isinstance(reasoning, str):
            effort = reasoning
        elif isinstance(reasoning, dict):
            effort = reasoning.get("effort")
    if effort is None:
        extra_body = body.get("extra_body")
        if isinstance(extra_body, dict):
            extra_reasoning = extra_body.get("reasoning")
            if isinstance(extra_reasoning, str):
                effort = extra_reasoning
            elif isinstance(extra_reasoning, dict):
                effort = extra_reasoning.get("effort")
            effort = effort or extra_body.get("reasoning_effort")
    return str(effort) if effort is not None else None


def _reasoning_efforts_from_catalog_row(row: dict[str, Any]) -> tuple[str, ...]:
    efforts = row.get("reasoning_efforts")
    if not isinstance(efforts, list):
        return ()
    normalized: list[str] = []
    for effort in efforts:
        if isinstance(effort, str) and effort.strip():
            normalized.append(effort.strip().lower())
    return tuple(dict.fromkeys(normalized))


def _reasoning_catalog_for_model(provider: str, model: str) -> tuple[bool, bool, tuple[str, ...], dict[str, Any]]:
    if provider == "codex":
        discovered = discover_codex_models()
        catalog = _provider_model_catalog(provider)
        if not discovered:
            return False, False, (), catalog
        model_info = resolve_codex_model(model, discovered)
        if model_info is None:
            return True, False, (), catalog
        return True, True, model_info.reasoning_efforts, catalog

    catalog = _provider_model_catalog(provider)
    rows = catalog.get("models", [])
    normalized_model = model.strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id.strip().lower() == normalized_model:
            return True, True, _reasoning_efforts_from_catalog_row(row), catalog
    return bool(rows), False, (), catalog


def _raise_unsupported_reasoning_effort(
    *,
    provider: str,
    model: str,
    effort: str,
    efforts: tuple[str, ...],
    catalog: dict[str, Any],
) -> None:
    available = list(efforts)
    accepted = ["none", *available]
    if available:
        guidance = f"Use one of: {', '.join(accepted)}; or omit reasoning/reasoning_effort."
    else:
        guidance = (
            f"Provider '{provider}' does not advertise reasoning controls for model '{model}'. "
            "Omit reasoning/reasoning_effort."
        )
    details = {
        "provider": provider,
        "model": model,
        "requested_reasoning_effort": effort,
        "available_reasoning_efforts": available,
        "accepted_reasoning_values": accepted,
        "reasoning_source": catalog.get("source"),
        "reasoning_catalog_live": catalog.get("live"),
        "reasoning_catalog_available": catalog.get("available"),
        "project_agent_message": guidance,
        "refresh_hint": (
            "/v1/scillm/models?refresh_provider_models=true or /v1/scillm/providers"
        ),
    }
    raise ProxyError(
        400,
        (
            f"Unsupported reasoning effort '{effort}' for provider '{provider}' "
            f"model '{model}'. {guidance}"
        ),
        "unsupported_reasoning_effort",
        advice=guidance,
        details=details,
    )


def _validate_provider_reasoning_effort(model: str, body: dict[str, Any]) -> None:
    effort = _normalize_reasoning_effort(body)
    if effort is None:
        return
    provider = _provider_for_model_selector(model)
    if provider not in {"codex", "claude", OPENCODE_GO_PROVIDER}:
        return
    normalized = effort.strip().lower()
    catalog_available, model_found, efforts, catalog = _reasoning_catalog_for_model(provider, model)
    if catalog_available and model_found and normalized != "none" and normalized not in efforts:
        _raise_unsupported_reasoning_effort(
            provider=provider,
            model=model,
            effort=effort,
            efforts=efforts,
            catalog=catalog,
        )
    if normalized != effort:
        body["reasoning_effort"] = normalized


def _oauth_provider_for_model(model: str) -> str | None:
    lower = model.lower()
    if lower.startswith("claude"):
        return "anthropic-oauth"
    if lower.startswith(("gpt", "codex", "o1", "o3", "o4")):
        return "codex-oauth"
    return None


def _provider_field_for_reasoning(provider: str | None) -> str | None:
    if provider == "codex-oauth":
        return "reasoning.effort"
    if provider == "anthropic-oauth":
        return "thinking.budget_tokens"
    if provider == "gemini":
        return "generationConfig.thinkingConfig.thinkingLevel"
    return None


def _attach_proof_fields(body: dict[str, Any], response_dict: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Attach scillm proof fields promised by the public skill contract."""
    requested_model = str(body.get("model") or "")
    served_model = str(response_dict.get("model") or requested_model)
    provider = _oauth_provider_for_model(served_model) or _oauth_provider_for_model(requested_model)
    reasoning_effort = _normalize_reasoning_effort(body)

    if reasoning_effort is not None:
        provider_field = _provider_field_for_reasoning(provider)
        response_dict["scillm_reasoning"] = {
            "requested_effort": reasoning_effort,
            "applied_effort": reasoning_effort if provider_field and reasoning_effort != "none" else None,
            "forwarded": bool(provider_field and reasoning_effort != "none"),
            "provider_field": provider_field,
            "ignored_reason": None if provider_field and reasoning_effort != "none" else "none_requested" if reasoning_effort == "none" else "unsupported_effort_for_provider",
        }

    if any(isinstance(msg, dict) and _message_has_multimodal_content(msg) for msg in messages):
        counts = _count_multimodal_parts(messages)
        image_seen_by = provider if provider in {"codex-oauth", "anthropic-oauth"} else None
        response_dict["scillm_multimodal"] = {
            "input_multimodal": True,
            **counts,
            "image_seen_by": image_seen_by,
            "routed_to_provider": provider,
            "model_requested": requested_model,
            "model_served": served_model,
        }


def _model_pool(pool_name: str) -> dict[str, Any] | None:
    pool = _DEFAULT_MODEL_POOLS.get(pool_name)
    if pool is None:
        return None
    return {
        **pool,
        "lanes": [dict(lane) for lane in pool["lanes"]],
    }


def _model_pool_status(
    pool_name: str,
    pool: dict[str, Any],
    concurrency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build pool-aware lane status from provider concurrency state."""
    concurrency = concurrency or {}
    lane_statuses = []
    seen_providers: set[str] = set()
    aggregate_in_flight = 0
    aggregate_limit = 0
    aggregate_queued = 0

    for lane in pool.get("lanes", []):
        provider = str(lane.get("provider") or "")
        provider_state = concurrency.get(provider, {}) if isinstance(concurrency, dict) else {}
        effective_limit = int(provider_state.get("effective_limit") or lane.get("max_concurrency") or 1)
        configured_limit = int(provider_state.get("configured_limit") or lane.get("max_concurrency") or effective_limit)
        semaphore_in_flight = int(
            provider_state.get("semaphore_in_flight")
            or provider_state.get("actual_in_flight")
            or provider_state.get("in_flight")
            or 0
        )
        live_in_flight = int(provider_state.get("live_in_flight") or semaphore_in_flight)
        queued = int(provider_state.get("queued") or 0)
        registry_in_flight = int(provider_state.get("registry_in_flight") or 0)
        stale_active_calls = int(provider_state.get("stale_active_calls") or 0)
        registry_drift = int(
            provider_state.get("registry_drift")
            if provider_state.get("registry_drift") is not None
            else registry_in_flight - semaphore_in_flight
        )
        status = {
            "name": lane.get("name"),
            "provider": provider,
            "model": lane.get("model"),
            "weight": lane.get("weight", 1),
            "require_hot_chutes": bool(lane.get("require_hot_chutes")),
            "lane_limit": int(lane.get("max_concurrency") or effective_limit),
            "configured_limit": configured_limit,
            "effective_limit": effective_limit,
            "in_flight": live_in_flight,
            "actual_in_flight": semaphore_in_flight,
            "live_in_flight": live_in_flight,
            "queued": queued,
            "available": int(provider_state.get("available") or max(0, effective_limit - semaphore_in_flight)),
            "paused": bool(provider_state.get("paused", False)),
            "backoff_active": bool(provider_state.get("backoff_active", False)),
            "pause_remaining_s": provider_state.get("pause_remaining_s", 0),
            "registry_in_flight": registry_in_flight,
            "stale_active_calls": stale_active_calls,
            "semaphore_in_flight": semaphore_in_flight,
            "registry_drift": registry_drift,
            "drift": registry_drift,
        }
        lane_statuses.append(status)
        if provider and provider not in seen_providers:
            seen_providers.add(provider)
            aggregate_in_flight += live_in_flight
            aggregate_limit += effective_limit
            aggregate_queued += queued

    return {
        "name": pool_name,
        "strategy": pool.get("strategy"),
        "in_flight": aggregate_in_flight,
        "actual_in_flight": aggregate_in_flight,
        "live_in_flight": aggregate_in_flight,
        "limit": aggregate_limit,
        "queued": aggregate_queued,
        "available": max(0, aggregate_limit - aggregate_in_flight),
        "lanes": lane_statuses,
    }


def _weighted_lane_sequence(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand lane weights into a deterministic weighted round-robin sequence."""
    sequence: list[dict[str, Any]] = []
    for lane in lanes:
        weight = int(lane.get("weight") or 1)
        sequence.extend([lane] * max(1, weight))
    if not sequence:
        raise ProxyError(400, "model pool must include at least one lane", "invalid_request_error")
    return sequence


def _lane_for_index(lanes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    sequence = _weighted_lane_sequence(lanes)
    return sequence[index % len(sequence)]


def _lane_provider_available(lane: dict[str, Any], concurrency: dict[str, Any]) -> int:
    provider = str(lane.get("provider") or "")
    provider_state = concurrency.get(provider, {}) if isinstance(concurrency, dict) else {}
    if bool(provider_state.get("paused", False)):
        return 0
    try:
        return max(0, int(provider_state.get("available")))
    except (TypeError, ValueError):
        pass
    try:
        effective_limit = int(provider_state.get("effective_limit") or lane.get("max_concurrency") or 1)
        in_flight = int(
            provider_state.get("semaphore_in_flight")
            or provider_state.get("actual_in_flight")
            or provider_state.get("in_flight")
            or 0
        )
        return max(0, effective_limit - in_flight)
    except (TypeError, ValueError):
        return int(lane.get("max_concurrency") or 1)


def _lane_for_index_with_capacity(
    lanes: list[dict[str, Any]],
    index: int,
    concurrency: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pick the weighted lane, but spill to an available provider if it is full.

    QRA callers often submit one-item batch requests. Without a global capacity
    check those requests can all select the weighted Chutes lane, then queue
    behind Chutes while another provider lane has slots free.
    """
    preferred = _lane_for_index(lanes, index)
    if not concurrency:
        return preferred
    if _lane_provider_available(preferred, concurrency) > 0:
        return preferred
    available_lanes = [lane for lane in lanes if _lane_provider_available(lane, concurrency) > 0]
    if not available_lanes:
        return preferred
    return max(
        available_lanes,
        key=lambda lane: (
            _lane_provider_available(lane, concurrency),
            int(lane.get("weight") or 1),
            -lanes.index(lane),
        ),
    )


def _messages_for_batch_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    for key in ("prompt", "input", "content"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return [{"role": "user", "content": value}]
    raise ProxyError(400, "each batch item needs messages, prompt, input, or content", "invalid_request_error")


def _item_id(item: dict[str, Any], index: int) -> str:
    return str(item.get("item_id") or item.get("id") or f"item-{index + 1}")


def _batch_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _message_text_content(message: dict[str, Any]) -> str:
    """Return assistant text, falling back to reasoning_content when content is empty."""
    content = message.get("content") or ""
    if not content and isinstance(message.get("reasoning_content"), str):
        content = message["reasoning_content"]
    return str(content or "")


def _chat_completion_text_content(data: dict[str, Any]) -> str:
    message = (data.get("choices") or [{}])[0].get("message") or {}
    if not isinstance(message, dict):
        return ""
    return _message_text_content(message)


async def _collect_chat_sse_lines(lines: AsyncIterator[str], *, requested_model: str) -> dict[str, Any]:
    """Collect an OpenAI-compatible chat SSE stream into a response dict."""
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    served_model = requested_model
    usage: dict[str, Any] | None = None
    done_seen = False
    last_event_name = "message"

    async for line in lines:
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            last_event_name = line[len("event:"):].strip() or "message"
            continue
        if not line.startswith("data:"):
            continue

        data_text = line[len("data:"):].strip()
        if data_text == "[DONE]":
            done_seen = True
            break
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            continue

        if "error" in event:
            error_payload = event["error"] if isinstance(event["error"], dict) else {"message": str(event["error"])}
            raise ProxyError(
                502,
                error_payload.get("message") or json.dumps(error_payload, sort_keys=True),
                error_payload.get("type") or last_event_name or "stream_error",
                details=error_payload,
            )

        if event.get("model"):
            served_model = str(event["model"])
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]

        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        content = delta.get("content")
        if content is None:
            content = message.get("content")
        if content:
            chunks.append(str(content))
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning_content")
        if reasoning:
            reasoning_chunks.append(str(reasoning))

    if not done_seen:
        raise ProxyError(502, "stream ended without [DONE]", "stream_error")

    final_content = "".join(chunks) or "".join(reasoning_chunks)
    response: dict[str, Any] = {
        "model": served_model,
        "choices": [{"message": {"role": "assistant", "content": final_content}}],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _image_model_config(model: str) -> dict[str, Any]:
    """Return image-generation model config, resolving public aliases."""
    requested = str(model or "z-image-turbo").strip() or "z-image-turbo"
    name = requested
    config = _IMAGE_GENERATION_MODELS.get(name)
    if isinstance(config, dict) and config.get("alias_for"):
        name = str(config["alias_for"])
        config = _IMAGE_GENERATION_MODELS.get(name)
    if not isinstance(config, dict) or config.get("alias_for"):
        available = ", ".join(sorted(k for k, v in _IMAGE_GENERATION_MODELS.items() if not v.get("alias_for")))
        raise ProxyError(400, f"Unknown image generation model '{requested}'. Available: {available}.", "invalid_request_error")
    alias_note = None
    if requested != name:
        alias_cfg = _IMAGE_GENERATION_MODELS.get(requested) or {}
        alias_note = alias_cfg.get("alias_note")
    result = {"requested_model": requested, "name": name, **config}
    if alias_note:
        result["alias_note"] = alias_note
    return result


def _chutes_image_api_key() -> str:
    if _config is not None and _config.chutes_api_key:
        return _config.chutes_api_key
    key = os.environ.get("CHUTES_API_KEY") or os.environ.get("CHUTES_API_TOKEN")
    if key:
        return key
    raise ProxyError(503, "CHUTES_API_KEY is required for image generation", "service_unavailable")


def _chutes_image_endpoint(model_config: dict[str, Any]) -> str:
    env_name = str(model_config.get("endpoint_env") or "")
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    value = os.environ.get("CHUTES_IMAGE_API_BASE", "").strip()
    if value:
        return value.rstrip("/") + "/generate" if not value.rstrip("/").endswith("/generate") else value.rstrip("/")
    return str(model_config["default_endpoint"])


def _parse_image_size(body: dict[str, Any]) -> tuple[int, int]:
    width = _int_or_none(body.get("width"))
    height = _int_or_none(body.get("height"))
    size = body.get("size")
    if (width is None or height is None) and isinstance(size, str) and "x" in size.lower():
        left, right = size.lower().split("x", 1)
        width = width if width is not None else _int_or_none(left.strip())
        height = height if height is not None else _int_or_none(right.strip())
    return width or 1024, height or 1024


def _clamp_image_value(
    value: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    field: str,
) -> int | float:
    if value < minimum or value > maximum:
        raise ProxyError(400, f"{field} must be between {minimum} and {maximum}", "invalid_request_error")
    return value


def _image_generation_args(body: dict[str, Any], *, item_index: int = 0) -> dict[str, Any]:
    """Normalize OpenAI-style image request fields to the z-image-turbo schema."""
    model_config = _image_model_config(str(body.get("model") or "z-image-turbo"))
    limits = model_config["limits"]
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProxyError(400, "prompt is required for image generation", "invalid_request_error")
    prompt = prompt.strip()
    prompt_len = len(prompt)
    if prompt_len < limits["prompt_min_length"] or prompt_len > limits["prompt_max_length"]:
        raise ProxyError(
            400,
            f"prompt length must be between {limits['prompt_min_length']} and {limits['prompt_max_length']} characters",
            "invalid_request_error",
        )

    width, height = _parse_image_size(body)
    width = int(_clamp_image_value(width, minimum=limits["width_min"], maximum=limits["width_max"], field="width"))
    height = int(_clamp_image_value(height, minimum=limits["height_min"], maximum=limits["height_max"], field="height"))

    steps = _int_or_none(body.get("num_inference_steps") or body.get("steps")) or 9
    steps = int(_clamp_image_value(
        steps,
        minimum=limits["num_inference_steps_min"],
        maximum=limits["num_inference_steps_max"],
        field="num_inference_steps",
    ))
    guidance_scale = _float_or_none(body.get("guidance_scale"))
    if guidance_scale is None:
        guidance_scale = 0.0
    guidance_scale = float(_clamp_image_value(
        guidance_scale,
        minimum=limits["guidance_scale_min"],
        maximum=limits["guidance_scale_max"],
        field="guidance_scale",
    ))
    shift = _float_or_none(body.get("shift"))
    if shift is None:
        shift = 3.0
    shift = float(_clamp_image_value(shift, minimum=limits["shift_min"], maximum=limits["shift_max"], field="shift"))
    max_sequence_length = _int_or_none(body.get("max_sequence_length")) or 512
    max_sequence_length = int(_clamp_image_value(
        max_sequence_length,
        minimum=limits["max_sequence_length_min"],
        maximum=limits["max_sequence_length_max"],
        field="max_sequence_length",
    ))
    seed = _int_or_none(body.get("seed"))
    if seed is not None:
        seed = int(_clamp_image_value(seed + item_index, minimum=limits["seed_min"], maximum=limits["seed_max"], field="seed"))

    args: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "max_sequence_length": max_sequence_length,
    }
    if seed is not None:
        args["seed"] = seed
    return args


def _normalize_image_response_payload(
    *,
    raw_content: bytes,
    content_type: str,
    response_format: str,
) -> dict[str, Any]:
    """Normalize Chutes PNG bytes or JSON wrappers into one OpenAI-style data item."""
    if content_type.startswith("image/"):
        b64 = base64.b64encode(raw_content).decode("ascii")
        if response_format == "url":
            return {"url": f"data:{content_type};base64,{b64}"}
        return {"b64_json": b64}

    try:
        payload = json.loads(raw_content.decode("utf-8"))
    except Exception as exc:
        raise ProxyError(502, f"image provider returned non-image, non-JSON response: {content_type}", "provider_error") from exc
    if not isinstance(payload, dict):
        raise ProxyError(502, "image provider returned malformed JSON response", "provider_error")

    for key in ("b64_json", "image_base64", "base64", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return {"b64_json": value.split(",", 1)[-1] if value.startswith("data:") else value}
    for key in ("url", "image_url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            if response_format == "b64_json" and value.startswith("data:") and "," in value:
                return {"b64_json": value.split(",", 1)[1]}
            return {"url": value}
    raise ProxyError(502, "image provider response did not include image bytes, b64_json, or url", "provider_error")


async def _call_chutes_image_generation(
    *,
    client: httpx.AsyncClient,
    model_config: dict[str, Any],
    args: dict[str, Any],
    response_format: str,
    caller_skill: str,
    timeout: float,
) -> dict[str, Any]:
    """Call one Chutes image-generation chute and return one normalized data item."""
    endpoint = _chutes_image_endpoint(model_config)
    api_key = _chutes_image_api_key()
    try:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Caller-Skill": caller_skill,
            },
            json={"input_args": args},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise ProxyError(502, f"image provider request failed: {exc}", "provider_error") from exc
    if response.status_code >= 400:
        message = response.text[:500] or f"HTTP {response.status_code}"
        raise ProxyError(502, f"image provider returned {response.status_code}: {message}", "provider_error")
    item = _normalize_image_response_payload(
        raw_content=response.content,
        content_type=(response.headers.get("content-type") or "application/octet-stream").split(";", 1)[0].strip().lower(),
        response_format=response_format,
    )
    item["revised_prompt"] = args["prompt"]
    return item


async def _generate_images_for_body(body: dict[str, Any], *, caller_skill: str) -> dict[str, Any]:
    """Generate one OpenAI-style image response; supports n via bounded asyncio."""
    started = time.monotonic()
    model_name = str(body.get("model") or "z-image-turbo")
    model_config = _image_model_config(model_name)
    provider = str(model_config.get("provider") or "chutes")
    response_format = str(body.get("response_format") or "b64_json")
    if response_format not in {"b64_json", "url"}:
        raise ProxyError(400, "response_format must be b64_json or url", "invalid_request_error")
    limits = model_config.get("limits") or {}
    n_max = int(limits.get("n_max") or 4)
    n = _int_or_none(body.get("n")) or 1
    if n < int(limits.get("n_min") or 1) or n > n_max:
        raise ProxyError(400, f"n must be between {limits.get('n_min', 1)} and {n_max} for image generation", "invalid_request_error")
    timeout = float(body.get("timeout") or 180.0)
    semaphore = asyncio.Semaphore(min(n, int(body.get("max_concurrency") or 4)))
    created = int(time.time())

    async def run_one(client: httpx.AsyncClient, index: int) -> dict[str, Any]:
        async with semaphore:
            if provider == "openai":
                payload = openai_images.openai_image_generation_payload(body, model_config=model_config)
                return await openai_images.call_openai_image_generation(
                    client=client,
                    payload=payload,
                    response_format=response_format,
                    timeout=timeout,
                )
            args = _image_generation_args(body, item_index=index)
            return await _call_chutes_image_generation(
                client=client,
                model_config=model_config,
                args=args,
                response_format=response_format,
                caller_skill=caller_skill,
                timeout=timeout,
            )

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(run_one(client, index)) for index in range(n)]
        data = [await task for task in asyncio.as_completed(tasks)]

    response: dict[str, Any] = {
        "created": created,
        "object": "list",
        "model": model_config.get("name", model_name),
        "provider": provider,
        "data": data,
        "ordered": False,
        "completion_order": "as_completed",
    }
    if model_config.get("requested_model") and model_config.get("requested_model") != model_config.get("name"):
        response["model_requested"] = model_config["requested_model"]
        if model_config.get("alias_note"):
            response["alias_note"] = model_config["alias_note"]
    elapsed_ms = int((time.monotonic() - started) * 1000)
    first = data[0] if data else {}
    response["scillm"] = {
        "status": "completed",
        "terminal": True,
        "surface": "/v1/images/generations",
        "caller_skill": caller_skill,
        "elapsed_ms": elapsed_ms,
        "image_count": len(data),
        "has_b64": bool(isinstance(first, dict) and first.get("b64_json")),
    }
    return response


def _request_deadline_timeout_s(body: dict[str, Any]) -> float | None:
    """Return caller-facing end-to-end timeout, not provider estimate."""
    explicit_timeout = _float_or_none(body.get("timeout"))
    if explicit_timeout is not None:
        return explicit_timeout
    return _float_or_none(body.get("_policy_max_timeout_s"))


def _stream_heartbeat_interval_s(body: dict[str, Any]) -> float:
    """Return heartbeat cadence for streaming responses."""
    for key in ("stream_heartbeat_s", "heartbeat_interval_s", "idle_timeout", "read_timeout"):
        value = _float_or_none(body.get(key))
        if value is not None and value > 0:
            return value
    return DEFAULT_STREAM_HEARTBEAT_S


def _timeout_error_details(
    body: dict[str, Any],
    *,
    model: str,
    started_at: float,
    timeout_s: float,
    final_provider_error: Exception | str | None = None,
) -> dict[str, Any]:
    metadata = body.get("_scillm_metadata")
    if not isinstance(metadata, dict):
        metadata = body.get("scillm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "caller": body.get("_caller_skill") or body.get("_headers", {}).get("x-caller-skill") or "unknown",
        "item_id": metadata.get("item_id"),
        "batch_id": metadata.get("batch_id"),
        "provider": body.get("_concurrency_provider"),
        "model": model,
        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        "timeout_s": timeout_s,
        "cascade_attempts": body.get("_cascade_attempts", 0),
        "final_provider_error": str(final_provider_error) if final_provider_error else None,
    }


async def _await_with_phase_progress(
    task_coro: Any,
    *,
    phase: str,
    model: str | None,
    heartbeat_s: float | None,
    wall_time_s: float,
) -> AsyncIterator[dict[str, Any]]:
    """Yield heartbeat events while awaiting a long preflight/probe coroutine."""
    task = asyncio.create_task(task_coro)
    started = time.monotonic()
    heartbeat = max(0.01, float(heartbeat_s or 15.0))
    deadline = started + max(1.0, float(wall_time_s or heartbeat))
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                yield {
                    "type": "prompt_phase_result",
                    "ok": False,
                    "phase": phase,
                    "model": model,
                    "error": "wall_time_exceeded",
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                return
            done, _ = await asyncio.wait({task}, timeout=min(heartbeat, remaining), return_when=asyncio.FIRST_COMPLETED)
            if done:
                yield {
                    "type": "prompt_phase_result",
                    "ok": True,
                    "phase": phase,
                    "model": model,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "result": task.result(),
                }
                return
            yield {
                "type": "prompt_phase_progress",
                "ok": True,
                "phase": phase,
                "model": model,
                "elapsed_s": round(time.monotonic() - started, 3),
                "deadline_s": round(max(0.0, deadline - time.monotonic()), 3),
            }
    except Exception as exc:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        yield {
            "type": "prompt_phase_result",
            "ok": False,
            "phase": phase,
            "model": model,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def _stream_error_from_chunk(chunk: str | bytes) -> RuntimeError | None:
    """Return a terminal stream error represented inside an SSE chunk."""
    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="replace")
    else:
        text = chunk
    for event in text.split("\n\n"):
        data_lines = [line[6:] for line in event.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            payload = yaml.safe_load(data_text)
        except Exception:
            continue
        if not isinstance(payload, dict) or "error" not in payload:
            continue
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "stream_error")
            error_type = str(error.get("type") or "stream_error")
            return RuntimeError(f"{error_type}: {message}")
        return RuntimeError(str(error))
    return None


def _stream_chunk_has_visible_output(chunk: str | bytes) -> bool:
    """Return True when an SSE chunk carries visible content or tool output."""
    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="replace")
    else:
        text = chunk
    for event in text.split("\n\n"):
        data_lines = [line[6:] for line in event.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            payload = yaml.safe_load(data_text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str) and delta["content"]:
                    return True
                if isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
                    return True
                if delta.get("tool_calls"):
                    return True
            message = choice.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("content"), str) and message["content"]:
                    return True
                if isinstance(message.get("reasoning_content"), str) and message["reasoning_content"]:
                    return True
                if message.get("tool_calls"):
                    return True
    return False


def _stream_chunk_reasoning_chars(chunk: str | bytes) -> int:
    """Count reasoning_content characters in an OpenAI-compatible SSE chunk."""
    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="replace")
    else:
        text = chunk
    total = 0
    for event in text.split("\n\n"):
        data_lines = [line[6:] for line in event.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            payload = yaml.safe_load(data_text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for source_key in ("delta", "message"):
                source = choice.get(source_key)
                if isinstance(source, dict) and isinstance(source.get("reasoning_content"), str):
                    total += len(source["reasoning_content"])
    return total


def _stream_terminal_diagnostics_from_chunk(chunk: str | bytes) -> dict[str, Any]:
    """Extract terminal OpenAI-compatible SSE diagnostics from a chunk."""
    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="replace")
    else:
        text = chunk
    diagnostics: dict[str, Any] = {}
    for event in text.split("\n\n"):
        data_lines = [line[6:] for line in event.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if data_text == "[DONE]":
            diagnostics["saw_done"] = True
            continue
        if not data_text:
            continue
        try:
            payload = yaml.safe_load(data_text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("usage"), dict):
            diagnostics["usage"] = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                diagnostics["finish_reason"] = finish_reason
    return diagnostics


async def _stream_with_middleware_lifecycle(
    stream: AsyncIterator[str | bytes],
    body: dict[str, Any],
    middleware_chain: MiddlewareChain,
) -> AsyncIterator[str | bytes]:
    """Finalize middleware only after the client-visible stream terminates."""
    terminal_error: Exception | None = None
    saw_visible_output = False
    reasoning_chars = 0
    try:
        async for chunk in stream:
            terminal_error = terminal_error or _stream_error_from_chunk(chunk)
            saw_visible_output = saw_visible_output or _stream_chunk_has_visible_output(chunk)
            reasoning_chars += _stream_chunk_reasoning_chars(chunk)
            diagnostics = _stream_terminal_diagnostics_from_chunk(chunk)
            if "finish_reason" in diagnostics:
                body["_stream_terminal_finish_reason"] = diagnostics["finish_reason"]
            if "usage" in diagnostics:
                body["_stream_terminal_usage"] = diagnostics["usage"]
            if diagnostics.get("saw_done"):
                body["_stream_saw_done"] = True
            yield chunk
    except asyncio.CancelledError as exc:
        await middleware_chain.run_on_error(body, exc)
        raise
    except Exception as exc:
        await middleware_chain.run_on_error(body, exc)
        raise
    else:
        body["_stream_visible_output"] = saw_visible_output
        body["_stream_reasoning_output"] = reasoning_chars > 0
        body["_stream_reasoning_chars"] = reasoning_chars
        if terminal_error is not None:
            await middleware_chain.run_on_error(body, terminal_error)
        elif not saw_visible_output:
            await middleware_chain.run_on_error(
                body,
                RuntimeError("stream_empty_output: no visible content or tool calls before [DONE]"),
            )
        else:
            await middleware_chain.run_post_call(body, {"stream": True, "stream_completed": True})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_adapter_manifests() -> dict[str, dict[str, Any]]:
    """Load local read-only adapter capability manifests."""
    manifests: dict[str, dict[str, Any]] = {}
    manifest_dir = _repo_root() / "registry" / "adapters"
    if not manifest_dir.exists():
        return manifests
    for path in sorted(manifest_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            logger.warning("failed to read adapter manifest {}: {}", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        adapter_id = str(data.get("id") or path.stem)
        manifests[adapter_id] = data
    return manifests


def _adapter_id_for_deployment(dep: Any, group_name: str = "") -> str:
    provider = (dep.custom_llm_provider or "").lower()
    model = (dep.model or "").lower()
    api_base = (dep.api_base or "").lower()
    group = group_name.lower()

    if provider == "anthropic-oauth" or model.startswith("claude") or "anthropic" in api_base:
        return "claude_oauth"
    if provider == "codex-oauth" or model.startswith(("gpt", "codex")):
        return "codex_oauth"
    if provider in {"gemini-oauth", "gemini"} or model.startswith("gemini") or "generativelanguage.googleapis.com" in api_base:
        return "gemini"
    if provider.startswith("opencode-go") or group.startswith("opencode-go/"):
        return "opencode_go"
    if "ollama" in api_base or ":" in model:
        return "ollama"
    if "chutes" in api_base or "/" in dep.model:
        return "chutes"
    return "openai_compatible"


def _deployment_supports(dep: Any, group_name: str, adapters: dict[str, dict[str, Any]]) -> dict[str, bool]:
    adapter_id = _adapter_id_for_deployment(dep, group_name)
    adapter_supports = adapters.get(adapter_id, {}).get("supports", {})
    supports = {
        "text": True,
        "image": bool(adapter_supports.get("image", False)),
        "pdf": bool(adapter_supports.get("pdf", False)),
        "zip": bool(adapter_supports.get("zip", False)),
        "streaming": bool(adapter_supports.get("streaming", True)),
        "tools": bool(adapter_supports.get("tools", True)),
    }
    model = (dep.model or "").lower()
    group = group_name.lower()
    if group.startswith("vlm") or "glm-4.6v" in model or "vl" in model:
        supports["image"] = True
    if adapter_id == "opencode_go":
        opencode_name = group_name if group_name.startswith("opencode-go/") else f"opencode-go/{dep.model}"
        opencode_caps = opencode_go_input_capabilities(opencode_name)
        supports["image"] = opencode_caps["image"]
        supports["pdf"] = opencode_caps["pdf"]
        supports["zip"] = False
    return supports


def _resolved_model_for_validation(model: str) -> str:
    """Resolve public aliases before capability validation."""
    if _config is None:
        return model
    return str(_config.aliases.get(model, model))


def _merge_supports(items: list[dict[str, bool]]) -> dict[str, bool]:
    keys = {"text", "image", "pdf", "zip", "streaming", "tools"}
    return {key: any(item.get(key, False) for item in items) for key in sorted(keys)}


def _suggest_model(unknown: str, candidates: set[str], n: int = 3) -> list[str]:
    """Suggest closest matching model names using fuzzy matching.

    Returns up to n suggestions, sorted by similarity (best first).
    """
    # Filter out internal model names (deepseek-ai/, Qwen/, etc.)
    user_facing = [m for m in candidates if "/" not in m]
    matches = difflib.get_close_matches(unknown.lower(), [m.lower() for m in user_facing], n=n, cutoff=0.4)
    # Map back to original case
    lower_to_orig = {m.lower(): m for m in user_facing}
    return [lower_to_orig[m] for m in matches]


def _is_public_selectable_model(name: str) -> bool:
    """Return false for compatibility-only or retired model names."""
    value = (name or "").strip()
    if not value:
        return False
    if value.startswith(_RETIRED_MODEL_PREFIXES):
        return False
    if _is_direct_chutes_model(value):
        return False
    if value in {"local-text", "moonshot-text"}:
        return False
    return True


def _model_catalog_entry(
    *,
    name: str,
    kind: str,
    target: str | None = None,
    deployments: int | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized model registry row for UI selection."""
    selectable = _is_public_selectable_model(name)
    if target is not None and not _is_public_selectable_model(target):
        selectable = False
    entry: dict[str, Any] = {
        "kind": kind,
        "selectable": selectable,
        "review_fanout_default": name in _REVIEW_FANOUT_MODEL_PREFERENCE and selectable,
    }
    if target is not None:
        entry["target"] = target
    if deployments is not None:
        entry["deployments"] = deployments
    if capabilities is not None:
        entry["capabilities"] = capabilities
    return entry


def _public_model_capabilities(model: str) -> dict[str, bool]:
    resolved = _resolved_model_for_validation(model)
    if resolved in _IMAGE_GENERATION_MODELS:
        return {
            "text_input": True,
            "image_input": False,
            "pdf_input": False,
            "image_output": True,
            "image_generation": True,
            "streaming": False,
            "tools": False,
        }
    if is_opencode_go_model(resolved):
        caps = opencode_go_input_capabilities(resolved)
        return {
            "text_input": caps["text"],
            "image_input": caps["image"],
            "pdf_input": caps["pdf"],
            "image_output": False,
            "image_generation": False,
            "streaming": True,
            "tools": True,
        }
    lower = resolved.lower()
    image_input = lower.startswith(("vlm", "gpt", "codex", "claude")) or lower in {
        "gemini-flash",
        "gemini-flash-high",
        "moonshot-text",
    }
    pdf_input = lower.startswith(("vlm", "claude")) or lower in {"gemini-flash", "gemini-flash-high"}
    return {
        "text_input": True,
        "image_input": image_input,
        "pdf_input": pdf_input,
        "image_output": False,
        "image_generation": False,
        "streaming": True,
        "tools": True,
    }


def _build_model_catalog() -> dict[str, dict[str, Any]]:
    """Return model aliases and groups with a fail-closed selection contract."""
    if _config is None:
        return {}

    models: dict[str, dict[str, Any]] = {}
    for name, target in _config.aliases.items():
        models[name] = _model_catalog_entry(
            name=name,
            kind="alias",
            target=str(target),
            capabilities=_public_model_capabilities(name),
        )
    for name, group in _config.model_groups.items():
        models[name] = _model_catalog_entry(
            name=name,
            kind="group",
            deployments=len(group.deployments),
            capabilities=_public_model_capabilities(name),
        )
    for image_name, image_config in _IMAGE_GENERATION_MODELS.items():
        if image_config.get("alias_for"):
            continue
        models[image_name] = _model_catalog_entry(
            name=image_name,
            kind="image_generation",
            target=str(image_config.get("slug") or image_name),
            deployments=1,
            capabilities=_public_model_capabilities(image_name),
        )
    return dict(sorted(models.items()))


def _models_dev_provider_key(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.strip().lower()
    if normalized in {"opencode-go", "opencode_go", "opencode"}:
        return "opencode"
    return normalized


async def _fetch_models_dev_catalog(*, refresh: bool = False) -> dict[str, Any]:
    """Fetch models.dev catalog as advisory metadata.

    This catalog is intentionally not the routing source of truth. scillm's
    local model catalog and provider adapters decide what is callable.
    """
    global _models_dev_cache, _models_dev_cache_ts

    now = time.monotonic()
    if (
        not refresh
        and _models_dev_cache is not None
        and now - _models_dev_cache_ts < MODELS_DEV_CACHE_TTL_S
    ):
        return _models_dev_cache

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(MODELS_DEV_API_URL)
        response.raise_for_status()
        catalog = response.json()
        if not isinstance(catalog, dict):
            raise ValueError("models.dev catalog root is not an object")
        _models_dev_cache = catalog
        _models_dev_cache_ts = now
        return catalog


def _models_dev_extract(
    catalog: dict[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    provider_key = _models_dev_provider_key(provider)
    provider_record = catalog.get(provider_key) if provider_key else None
    if provider_key and not isinstance(provider_record, dict):
        return {
            "provider": provider,
            "provider_key": provider_key,
            "found": False,
            "model": model,
            "advisory_only": True,
        }

    if not provider_key:
        return {
            "providers": sorted(catalog.keys()),
            "provider_count": len(catalog),
            "advisory_only": True,
        }

    models = provider_record.get("models", {}) if isinstance(provider_record, dict) else {}
    if not isinstance(models, dict):
        models = {}

    if model:
        model_key = model.removeprefix("opencode-go/") if provider_key == "opencode" else model
        model_record = models.get(model_key)
        return {
            "provider": provider,
            "provider_key": provider_key,
            "model": model,
            "model_key": model_key,
            "found": isinstance(model_record, dict),
            "advisory_only": True,
            "record": model_record if isinstance(model_record, dict) else None,
        }

    return {
        "provider": provider,
        "provider_key": provider_key,
        "found": True,
        "advisory_only": True,
        "model_count": len(models),
        "models": models,
    }


def _models_dev_advisory_for_name(catalog: dict[str, Any], name: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    target = str(entry.get("target") or name)
    if target.startswith("opencode-go/"):
        result = _models_dev_extract(catalog, provider="opencode-go", model=target)
        record = result.get("record")
        if isinstance(record, dict):
            return {
                "provider": "opencode-go",
                "model": target,
                "modalities": record.get("modalities"),
                "attachment": record.get("attachment"),
                "reasoning": record.get("reasoning"),
                "tool_call": record.get("tool_call"),
                "status": record.get("status"),
                "advisory_only": True,
            }
    return None


def _validate_model_request(model: str, body: dict, request: Request) -> None:
    """Validate model request and auto-fix common issues.

    Guardrails (in order):
    1. Reject deprecated models with clear error
    2. Reject unknown models with helpful suggestions
    3. Reject empty messages
    4. Warn on direct model names (should use aliases)
    5. Auto-enable x-expect-json when response_format: json_object is used
    6. Strip max_tokens (causes empty output on reasoning models)
    7. Clamp temperature to valid range
    8. Warn on very long prompts
    """
    messages = body.get("messages", [])

    # 0a. Handle string messages — common agent mistake
    # Agents often pass messages="prompt" instead of messages=[{"role": "user", "content": "prompt"}]
    if isinstance(messages, str):
        logger.warning(
            "Auto-wrapping string messages as list — caller passed string instead of array. "
            "Correct format: messages=[{\"role\": \"user\", \"content\": \"...\"}]"
        )
        body["messages"] = [{"role": "user", "content": messages}]
        messages = body["messages"]

    # 0b. Reject empty messages — crashes some providers
    if not messages:
        raise ProxyError(
            400,
            "messages is required and cannot be empty. "
            "Provide at least one message: [{\"role\": \"user\", \"content\": \"...\"}]. "
            "scillm is an HTTP API — call via httpx, not import.",
            "invalid_request_error",
        )

    # 1. Log deprecated models but let alias map handle the remap
    if model in _DEPRECATED_MODELS:
        caller = request.headers.get("x-caller-skill", "unknown")
        logger.info(
            "Deprecated model '{}' requested by '{}' — auto-remapping via alias. {}",
            model, caller, _DEPRECATED_MODELS[model],
        )

    if model.lower() in _UNSUPPORTED_CHATGPT_CODEX_OAUTH_MODELS:
        raise ProxyError(
            400,
            (
                f"Model '{model}' is not supported for one-shot Codex OAuth "
                "with a ChatGPT account. Use 'gpt-5.5' for "
                "POST /v1/chat/completions, or use a codex_exec profile for "
                "bounded CLI worker calls."
            ),
            "invalid_request_error",
        )

    _validate_provider_reasoning_effort(model, body)
    dynamic_provider_checked = _validate_dynamic_provider_model(model)

    # 2. Reject unknown models with helpful error + suggestions
    # Only check if we have loaded valid aliases (after startup)
    if (
        _VALID_MODEL_ALIASES
        and not dynamic_provider_checked
        and model not in _VALID_MODEL_ALIASES
        and not is_opencode_go_model(model)
        and not _is_direct_chutes_model(model)
        and not (_is_codex_oauth_model(model) and _codex_oauth_available())
    ):
        suggestions = _suggest_model(model, _VALID_MODEL_ALIASES)
        available = ", ".join(sorted(m for m in _VALID_MODEL_ALIASES if "/" not in m))
        msg = f"Unknown model '{model}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        msg += f" Available: {available}."
        raise ProxyError(400, msg, "invalid_request_error")

    # 3. Warn on direct model names (log only, don't block)
    model_lower = model.lower()
    for pattern, hint in _DIRECT_MODEL_PATTERNS.items():
        if model_lower.startswith(pattern) and "/" not in model:
            logger.warning(
                f"Direct model name '{model}' used without fallbacks. {hint}"
            )
            break

    # 4. Auto-enable JSON repair when response_format: json_object is used
    # This is the #1 cause of failures — Claude ignores response_format and
    # returns markdown-wrapped JSON. Auto-fix makes it "just work".
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        # Always enable JSON repair when JSON is requested
        body["_expect_json"] = True
        logger.debug("Auto-enabled JSON repair for response_format: json_object")

    # 5. Strip output token caps — causes 90% empty responses on reasoning models
    # Reasoning models (DeepSeek-R1, o1, etc) spend tokens on internal reasoning.
    # If a completion token cap is set too low, all tokens go to reasoning and
    # output is empty.
    # Better to let the model decide than risk empty responses.
    for token_cap_field in ("max_tokens", "max_completion_tokens"):
        if body.get(token_cap_field) is None:
            body.pop(token_cap_field, None)
        elif token_cap_field in body:
            logger.warning(
                f"Stripping {token_cap_field}={body[token_cap_field]} — causes empty output on reasoning models. "
                f"See MEMORY.md: 'Never use max_tokens'."
            )
            del body[token_cap_field]

    # Also honor explicit x-expect-json header
    if request.headers.get("x-expect-json", "").lower() in ("true", "1", "yes"):
        body["_expect_json"] = True

    # 6. Clamp temperature to valid range (0-2)
    # Values outside this range break most providers
    temp = body.get("temperature")
    if temp is not None:
        if not isinstance(temp, (int, float)):
            logger.warning(f"Invalid temperature type {type(temp).__name__}, removing")
            del body["temperature"]
        elif temp < 0 or temp > 2:
            clamped = max(0, min(2, temp))
            logger.warning(f"Clamping temperature {temp} to valid range: {clamped}")
            body["temperature"] = clamped

    # 7. Warn on very long prompts (>100k chars ~ 25k tokens)
    # These can timeout or OOM on some providers
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if total_chars > 100000:
        logger.warning(
            f"Very long prompt ({total_chars:,} chars, ~{total_chars//4:,} tokens). "
            f"May timeout on some providers."
        )

    # 8. Detect empty content in messages — crashes some providers
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            logger.warning(
                f"Message {i} has empty content — may crash some providers. "
                f"Ensure all messages have non-empty content."
            )

    # 9. Detect inlineData with non-Gemini model — silent failure
    # inlineData only works with Gemini's native API
    has_inline_data = any(
        isinstance(msg.get("content"), list) and
        any(isinstance(p, dict) and "inlineData" in p for p in msg.get("content", []))
        for msg in messages
    )
    has_multimodal_content = any(
        isinstance(msg, dict) and _message_has_multimodal_content(msg)
        for msg in messages
    )
    resolved_model = _resolved_model_for_validation(model)
    if is_opencode_go_model(resolved_model) and has_multimodal_content:
        counts = _count_multimodal_parts(messages)
        opencode_caps = opencode_go_input_capabilities(resolved_model)
        image_only = (
            counts["image_count"] > 0
            and counts["document_count"] == 0
            and counts["inline_data_count"] == 0
        )
        if image_only and opencode_caps["image"]:
            return
        unsupported_parts: list[str] = []
        if counts["image_count"] and not opencode_caps["image"]:
            unsupported_parts.append("image")
        if counts["document_count"] or counts["inline_data_count"]:
            unsupported_parts.append("pdf/document/inlineData")
        raise ProxyError(
            400,
            f"OpenCode Go model '{model}' resolves to '{resolved_model}' and does not support "
            f"{', '.join(unsupported_parts) or 'this multimodal payload'} through /scillm; "
            "it is text-only for this payload. "
            "Use model='oc-kimi' or 'opencode-go/kimi-k2.6' for OpenAI-style image_url PNG/JPEG review, "
            "or use Gemini/Claude for PDFs and inlineData.",
            "invalid_request_error",
        )
    if has_inline_data and not model_lower.startswith("gemini"):
        raise ProxyError(
            400,
            f"inlineData parts only work with Gemini models (gemini-flash, gemini-flash-high, or direct gemini-* IDs). "
            f"You used model='{model}'. For images with other models, use image_url format. "
            f"For PDFs with Claude, use document blocks. See SKILL.md 'Sending Multiple Files'.",
            "invalid_request_error",
        )

    # 10. Detect tool_choice: "required" with Codex — Codex rejects it
    tool_choice = body.get("tool_choice")
    if tool_choice == "required" and model_lower.startswith(("gpt-", "codex")):
        logger.warning(
            "tool_choice='required' not supported by Codex — forcing to 'auto'. "
            "Codex always uses auto tool selection."
        )
        body["tool_choice"] = "auto"


# ---------------------------------------------------------------------------
# Middleware loading
# ---------------------------------------------------------------------------


async def _load_middleware(config: ProxyConfig) -> list[BaseMiddleware]:
    """Instantiate middleware from config.

    Order matters:
      1. AbuseGuard — blocks clients after repeated 4xx errors (pre_call)
      2. CacheMiddleware — returns cached responses, dedupes in-flight (pre_call) — BEFORE concurrency
      3. TimeoutEstimator — queries /latency-stats, sets _dynamic_timeout_ms (pre_call)
      4. CallerPolicy — per-caller blast-radius controls (pre_call)
      5. ChutesRouter — selects least saturated Chutes model variant (pre_call)
      6. VlmRouter — rewrites model before routing (pre_call)
      7. ConcurrencyMiddleware — acquires provider semaphore (pre_call), releases (post_call/on_error)
      8. JsonGuard — validates JSON responses, repairs or raises (post_call)
      9. BudgetMiddleware — tracks spend, exposes via headers (post_call)
      10. CostHeaderMiddleware — injects x-cost-usd headers (post_call)
      11. ArangoLogMiddleware — logs to ArangoDB llm_call_log (post_call, on_error) — LAST

    All persistence uses ArangoDB (no Redis):
      - CacheMiddleware → scillm_response_cache
      - ConcurrencyMiddleware → scillm_concurrency_state
      - ArangoLogMiddleware → llm_call_log
    """
    from chutes.middleware.vlm_router import VlmRouter
    from chutes.middleware.concurrency_guard import ConcurrencyMiddleware
    from chutes.middleware.json_guard import JsonGuard
    from chutes.middleware.abuse_guard import AbuseGuardMiddleware
    from chutes.middleware.chutes_router import ChutesRouter

    middlewares: list[BaseMiddleware] = [AbuseGuardMiddleware()]

    # Cache middleware — returns cached responses, dedupes in-flight requests
    # Must be BEFORE ConcurrencyMiddleware so cache hits don't consume concurrency slots
    try:
        from chutes.middleware.cache_init import CacheMiddleware
        cache_mw = CacheMiddleware()
        await cache_mw.initialize()
        middlewares.append(cache_mw)
        logger.info("CacheMiddleware loaded (backend: {})", cache_mw._backend)
    except (ImportError, Exception) as exc:
        logger.debug("CacheMiddleware not loaded: {}", exc)

    # Timeout estimator — queries /latency-stats, sets _dynamic_timeout_ms
    try:
        from chutes.middleware.timeout_estimator import TimeoutEstimatorMiddleware
        middlewares.append(TimeoutEstimatorMiddleware())
        logger.info("TimeoutEstimatorMiddleware loaded")
    except (ImportError, Exception) as exc:
        logger.debug("TimeoutEstimatorMiddleware not loaded: {}", exc)

    if config.caller_profiles:
        try:
            from chutes.middleware.caller_policy import CallerPolicyMiddleware
            middlewares.append(CallerPolicyMiddleware(config))
            logger.info("CallerPolicyMiddleware loaded ({} profiles)", len(config.caller_profiles))
        except (ImportError, Exception) as exc:
            logger.debug("CallerPolicyMiddleware not loaded: {}", exc)

    # ChutesRouter selects best model variant based on utilization (before VlmRouter)
    middlewares.extend([ChutesRouter(), VlmRouter(), ConcurrencyMiddleware(), JsonGuard()])
    logger.info("ChutesRouter loaded (utilization-aware routing)")

    # Grounding guard — verifies response is grounded in provided source text
    # Supports course correction: retries with helpful error messages
    try:
        from chutes.middleware.grounding_guard import GroundingGuard
        middlewares.append(GroundingGuard())
        logger.info("GroundingGuard loaded (source grounding with course correction)")
    except (ImportError, Exception) as exc:
        logger.debug("GroundingGuard not loaded: {}", exc)

    # Schema guard — validates JSON against schema with course correction
    # Supports course correction: retries with detailed field-level errors
    try:
        from chutes.middleware.schema_guard import SchemaGuard
        middlewares.append(SchemaGuard())
        logger.info("SchemaGuard loaded (JSON schema validation with course correction)")
    except (ImportError, Exception) as exc:
        logger.debug("SchemaGuard not loaded: {}", exc)

    # Budget guard is optional — only loads if chutes env vars are set
    try:
        from chutes.middleware.budget_guard import BudgetMiddleware
        middlewares.append(BudgetMiddleware())
        logger.info("BudgetMiddleware loaded")
    except (ImportError, Exception) as exc:
        logger.debug("BudgetMiddleware not loaded: {}", exc)

    # Cost header middleware — injects x-cost-usd headers
    try:
        from chutes.middleware.pricing import CostHeaderMiddleware
        middlewares.append(CostHeaderMiddleware())
        logger.info("CostHeaderMiddleware loaded")
    except (ImportError, Exception) as exc:
        logger.debug("CostHeaderMiddleware not loaded: {}", exc)

    # ArangoDB logging — writes to llm_call_log for /learn-timeout and /orchestrate
    try:
        from chutes.middleware.arango_log import ArangoLogMiddleware
        middlewares.append(ArangoLogMiddleware())
        logger.info("ArangoLogMiddleware loaded")
    except (ImportError, Exception) as exc:
        logger.debug("ArangoLogMiddleware not loaded: {}", exc)

    # Active calls tracking — exposes in-flight requests for live monitoring.
    # It must register after ConcurrencyMiddleware acquires a provider slot so
    # active-calls and health semaphore state describe the same live lifecycle.
    try:
        from chutes.middleware.active_calls import ActiveCallsMiddleware, start_active_call_janitor
        active_mw = ActiveCallsMiddleware()
        start_active_call_janitor()
        insert_at = next(
            (
                index + 1
                for index, middleware in enumerate(middlewares)
                if type(middleware).__name__ == "ConcurrencyMiddleware"
            ),
            0,
        )
        middlewares.insert(insert_at, active_mw)
        logger.info("ActiveCallsMiddleware loaded")
    except (ImportError, Exception) as exc:
        logger.debug("ActiveCallsMiddleware not loaded: {}", exc)

    # Batch progress tracking — broadcasts to WebSocket subscribers
    try:
        from chutes.middleware.batch_ws import BatchProgressMiddleware
        middlewares.append(BatchProgressMiddleware())
        logger.info("BatchProgressMiddleware loaded (WebSocket batch tracking)")
    except (ImportError, Exception) as exc:
        logger.debug("BatchProgressMiddleware not loaded: {}", exc)

    return middlewares


# ---------------------------------------------------------------------------
# Chutes warm model auto-detection
# ---------------------------------------------------------------------------

CHUTES_API_BASE = os.environ.get("CHUTES_API_BASE", "https://llm.chutes.ai")
CHUTES_MGMT_API = "https://api.chutes.ai"


async def _find_warm_chutes_variant(model: str, client: httpx.AsyncClient, headers: dict) -> str | None:
    """Find a hot variant in the same model family.

    E.g., if model is deepseek-ai/DeepSeek-V3.1-TEE and V3.2-TEE is hot, return V3.2-TEE.
    Returns None if no better variant found or on error.
    """
    import re

    try:
        logger.debug("Checking warm variants for {}", model)
        # Get all available models from inference API
        resp = await client.get(f"{CHUTES_API_BASE}/v1/models", headers=headers)
        if resp.status_code != 200:
            logger.warning("Chutes /v1/models returned {}: {}", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        all_models = data.get("data", [])
        logger.debug("Got {} models from Chutes API", len(all_models))

        # Find the family using the same logic as ops-chutes find_family()
        org = model.split("/")[0]
        # Strip TEE suffix and version qualifiers to get core family
        base = re.sub(r"-TEE$", "", model)
        base_name = base.split("/")[-1]
        # Strip version suffixes: .1, .2, -0528, -Speciale, -Terminus, -FP8, -Instruct-2507
        core = re.sub(r"[-.](\\d+|Speciale|Terminus|FP8|Instruct|Thinking)(-\\d{4})?$", "", base_name)
        core = re.sub(r"\\.\\d+$", "", core)

        family = []
        for m in all_models:
            mid = m.get("id", "")
            if not mid.startswith(org + "/"):
                continue
            m_name = mid.split("/")[-1]
            m_stripped = re.sub(r"-TEE$", "", m_name)
            m_core = re.sub(r"[-.](\\d+|Speciale|Terminus|FP8|Instruct|Thinking)(-\\d{4})?$", "", m_stripped)
            m_core = re.sub(r"\\.\\d+$", "", m_core)
            if m_core == core:
                family.append(m)

        logger.debug("Found {} models in {} family", len(family), core)
        if not family:
            return None

        # Check each variant's hot status via management API
        hot_variants = []
        for m in family:
            mid = m["id"]
            chute_id = m.get("chute_id", "")
            if not chute_id:
                continue
            try:
                detail_resp = await client.get(f"{CHUTES_MGMT_API}/chutes/{chute_id}", headers=headers)
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    is_hot = detail.get("hot")
                    logger.debug("Model {} chute_id={} hot={}", mid, chute_id, is_hot)
                    if is_hot:
                        hot_variants.append(mid)
            except Exception as exc:
                logger.debug("Failed to get chute details for {}: {}", mid, exc)
                continue

        if not hot_variants:
            logger.debug("No hot variants found for family {}", core)
            return None

        # Prefer the originally requested model if it's hot
        if model in hot_variants:
            logger.debug("Configured model {} is already hot", model)
            return None  # Already using the right one

        # Return first hot variant (could add latency ranking later)
        logger.info("Found hot Chutes variant: {} (configured: {})", hot_variants[0], model)
        return hot_variants[0]

    except Exception as exc:
        logger.warning("Warm variant check failed: {}", exc)
        return None


async def _auto_select_warm_models(config: ProxyConfig, router) -> None:
    """Check Chutes models and switch to hot variants if available.

    Modifies router deployments in-place to use warm models.
    """
    logger.info("Checking for warm Chutes model variants...")
    chutes_base = os.environ.get("CHUTES_API_BASE", CHUTES_API_BASE)
    chutes_key = os.environ.get("CHUTES_API_KEY") or os.environ.get("CHUTES_API_TOKEN")
    if not chutes_key:
        logger.debug("No CHUTES_API_KEY — skipping warm model check")
        return

    headers = {"Authorization": f"Bearer {chutes_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for group_name, group in config.model_groups.items():
            for dep in group.deployments:
                if not dep.api_base or chutes_base not in dep.api_base:
                    continue

                warm = await _find_warm_chutes_variant(dep.model, client, headers)
                if warm and warm != dep.model:
                    old_model = dep.model
                    dep.model = warm
                    logger.warning(
                        "AUTO-SWITCH: {} deployment {} → {} (hot)",
                        group_name, old_model, warm,
                    )


async def _warmup_chutes_models(config: ProxyConfig) -> None:
    """Call the Chutes warmup API for each Chutes deployment on startup.

    Runs in the background — doesn't block proxy readiness.
    Uses GET /chutes/warmup/{model_name}?quick=true which returns immediately
    and creates a bounty for miners to spin up instances.
    """
    chutes_base = os.environ.get("CHUTES_API_BASE", CHUTES_API_BASE)
    chutes_key = os.environ.get("CHUTES_API_KEY") or os.environ.get("CHUTES_API_TOKEN")
    if not chutes_key:
        logger.debug("No CHUTES_API_KEY — skipping warmup")
        return

    # Collect unique Chutes model names from config
    models_to_warm: set[str] = set()
    for group in config.model_groups.values():
        for dep in group.deployments:
            if dep.api_base and chutes_base in dep.api_base:
                models_to_warm.add(dep.model)

    if not models_to_warm:
        return

    headers = {"Authorization": f"Bearer {chutes_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in models_to_warm:
            try:
                resp = await client.get(
                    f"{CHUTES_MGMT_API}/chutes/warmup/{model}",
                    params={"quick": "true"},
                    headers=headers,
                )
                if resp.status_code == 200:
                    logger.info("Chutes warmup sent for {} — miners notified", model)
                else:
                    logger.warning("Chutes warmup {} returned {}: {}", model, resp.status_code, resp.text[:200])
            except Exception as exc:
                logger.warning("Chutes warmup {} failed: {}", model, exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config, init router and middleware on startup."""
    global _config, _router, _middleware_chain, _start_time, _embedding_client

    config_path = os.environ.get("CONFIG_FILE_PATH", "local/proxy_server_config.yaml")
    logger.info("Loading config from {}", config_path)
    _config = load_config(config_path)
    _router = Router(_config)
    _middleware_chain = MiddlewareChain(await _load_middleware(_config))
    _embedding_client = httpx.AsyncClient(base_url=EMBEDDING_SERVICE_URL, timeout=30.0)
    _start_time = time.monotonic()

    # Populate valid model aliases for request validation
    _VALID_MODEL_ALIASES.clear()
    _VALID_MODEL_ALIASES.update(_config.model_groups.keys())
    _VALID_MODEL_ALIASES.update(_config.aliases.keys())
    logger.debug("Valid model aliases: {}", sorted(_VALID_MODEL_ALIASES))

    # Validate env vars don't reference deprecated models
    _check_env_for_deprecated_models()

    # Load persisted concurrency backoff state from ArangoDB (survives restarts)
    from chutes.middleware.concurrency_guard import load_state_from_arango
    await load_state_from_arango()

    logger.info(
        "scillm proxy started — {} model groups, {} aliases, {} fallback chains",
        len(_config.model_groups),
        len(_config.aliases),
        len(_config.fallbacks),
    )

    if os.environ.get("SCILLM_CHUTES_STARTUP_AUTO_SELECT", "").lower() in {"1", "true", "yes"}:
        await _auto_select_warm_models(_config, _router)
    else:
        logger.info("Chutes startup auto-select disabled; use ops-chutes for live model selection")

    if os.environ.get("SCILLM_CHUTES_STARTUP_WARMUP", "").lower() in {"1", "true", "yes"}:
        asyncio.create_task(_warmup_chutes_models(_config))
    else:
        logger.info("Chutes startup warmup disabled; use ops-chutes for warmup/health checks")

    yield
    # Graceful shutdown: drain in-flight requests before closing clients
    logger.info("scillm proxy shutting down — draining connections...")
    if _embedding_client:
        await _embedding_client.aclose()
    if _router:
        await _router.close()
    logger.info("scillm proxy shut down cleanly")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="scillm proxy", version="2.0.0", lifespan=lifespan)
app.add_exception_handler(ProxyError, proxy_error_handler)


@app.exception_handler(MiddlewareReject)
async def _reject_handler(request: Request, exc: MiddlewareReject):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "middleware_reject", "code": exc.status_code}},
    )


# ---------------------------------------------------------------------------
# Prometheus metrics (optional)
# ---------------------------------------------------------------------------

_prom_available = False
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    _prom_available = True
except ImportError:
    logger.warning("prometheus_client not installed, /metrics disabled")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Simple health check — no auth required. Returns status and uptime."""
    uptime = time.monotonic() - _start_time if _start_time else 0
    return {
        "status": "ok" if _config is not None else "starting",
        "uptime_seconds": round(uptime, 1),
    }


@app.get("/health/liveliness")
async def health_liveliness():
    """Basic liveness probe for Docker healthcheck."""
    return {"status": "ok"}


@app.get("/health/readiness")
async def health_readiness():
    """Readiness probe — checks config loaded."""
    if _config is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready", "model_groups": len(_config.model_groups)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    # Request tracing
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])

    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    # Check if client is blocked for repeated bad requests (early rejection)
    try:
        from chutes.middleware.abuse_guard import check_client_blocked
        auth = request.headers.get("authorization", "")
        client_ip = request.client.host if request.client else "unknown"
        check_client_blocked(auth, client_ip)
    except ImportError:
        pass  # Abuse guard not loaded

    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not model:
        raise ProxyError(400, "model is required", "invalid_request_error")
    if not messages:
        raise ProxyError(400, "messages is required", "invalid_request_error")

    # -------------------------------------------------------------------------
    # GUARDRAILS: Make common mistakes fail loudly or auto-fix
    # -------------------------------------------------------------------------
    _validate_model_request(model, body, request)
    normalized_reasoning_effort = _normalize_reasoning_effort(body)
    if normalized_reasoning_effort is not None:
        body["reasoning_effort"] = normalized_reasoning_effort

    if _middleware_chain is None or _router is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")

    # ── Caller identification ─────────────────────────────────────────────────
    # X-Caller-Skill is required so calls are attributable and failure reports
    # can be routed to the owning skill/project.
    caller_skill = _require_caller_skill(request)

    # Inject headers for middleware (arango_log.py uses x-caller-skill)
    body["_headers"] = dict(request.headers)
    body["_caller_skill"] = caller_skill
    start = time.monotonic()
    deadline_timeout_s = _request_deadline_timeout_s(body)

    # Pre-call middleware (can modify request or reject). If a later
    # pre_call hook rejects, earlier hooks such as ActiveCallsMiddleware and
    # ConcurrencyMiddleware must still see on_error and clean up.
    try:
        if deadline_timeout_s is not None and deadline_timeout_s > 0:
            remaining = (start + deadline_timeout_s) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired before middleware")
            async with asyncio.timeout(remaining):
                body = await _middleware_chain.run_pre_call(body)
        else:
            body = await _middleware_chain.run_pre_call(body)
    except TimeoutError as exc:
        proxy_exc = ProxyError(
            504,
            f"Request timed out after {deadline_timeout_s}s before provider call completed",
            "timeout_error",
            details=_timeout_error_details(
                body,
                model=model,
                started_at=start,
                timeout_s=deadline_timeout_s or 0,
                final_provider_error=exc,
            ),
        )
        await _middleware_chain.run_on_error(body, proxy_exc)
        raise proxy_exc
    except BaseException as exc:
        await _middleware_chain.run_on_error(body, exc if isinstance(exc, Exception) else Exception(type(exc).__name__))
        raise
    model = body.get("model", model)
    messages = body.get("messages", messages)

    # Extract metadata AFTER pre_call but BEFORE routing (LLM never sees it).
    # Stash under private key so post_call middleware can access it.
    caller_metadata = body.pop("scillm_metadata", None)
    if caller_metadata is not None:
        logger.debug("scillm_metadata received: {}", caller_metadata)
        body["_scillm_metadata"] = caller_metadata  # for arango_log.py

    # Extract kwargs for the openai client
    kwargs: dict[str, Any] = {}
    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "stop", "n", "response_format",
                "tools", "tool_choice", "seed", "logprobs", "top_logprobs",
                "reasoning_effort", "require_exact_model", "allow_model_remap"):
        if key in body:
            kwargs[key] = body[key]
    kwargs["stream"] = stream
    provider_bound_diagnostics: dict[str, Any] = {}
    kwargs["_provider_bound_diagnostics"] = provider_bound_diagnostics
    body["_provider_bound_diagnostics"] = provider_bound_diagnostics

    # Pass dynamic timeout from TimeoutEstimatorMiddleware to router
    if "_dynamic_timeout_ms" in body:
        kwargs["_dynamic_timeout_ms"] = body["_dynamic_timeout_ms"]
    if "_policy_max_timeout_s" in body:
        kwargs["_policy_max_timeout_s"] = body["_policy_max_timeout_s"]

    policy_deadline_timeout_s = _request_deadline_timeout_s(body)
    if deadline_timeout_s is None or (
        policy_deadline_timeout_s is not None
        and policy_deadline_timeout_s < deadline_timeout_s
    ):
        deadline_timeout_s = policy_deadline_timeout_s
    if deadline_timeout_s is not None and deadline_timeout_s > 0:
        kwargs["_deadline_started_at"] = start
        kwargs["_deadline_timeout_s"] = deadline_timeout_s
        kwargs["_deadline_at"] = start + deadline_timeout_s
        kwargs["_caller_skill"] = caller_skill
        kwargs["_scillm_metadata"] = caller_metadata if isinstance(caller_metadata, dict) else {}

    # Pass dynamic fallback chain from ChutesRouter (utilization-aware ordering)
    if "_dynamic_fallback_chain" in body:
        kwargs["_dynamic_fallback_chain"] = body["_dynamic_fallback_chain"]

    # -------------------------------------------------------------------------
    # BATCH RESUME: Check if this work item already completed successfully
    # -------------------------------------------------------------------------
    # Skills pass scillm_metadata: {"batch_id": "X", "item_id": "123"}
    # If found in llm_call_log with status=ok, return cached response
    if caller_metadata and not stream:
        try:
            from chutes.middleware.batch_resume import check_batch_resume
            cached_response = await check_batch_resume(caller_metadata)
            if cached_response:
                elapsed = time.monotonic() - start
                body["_cache_hit"] = True  # Skip arango_log for cache hits
                return JSONResponse(
                    content=cached_response,
                    headers={
                        "x-request-id": request_id,
                        "x-latency-ms": str(int(elapsed * 1000)),
                        "x-batch-resumed": "true",
                    },
                )
        except ImportError:
            pass  # batch_resume module not available

    # -------------------------------------------------------------------------
    # COURSE CORRECTION RETRY LOOP
    # Schema and Grounding guards may request retries with correction prompts.
    # Max 3 correction attempts to prevent infinite loops.
    # -------------------------------------------------------------------------
    MAX_CORRECTION_ATTEMPTS = 3
    correction_attempt = 0
    working_messages = list(messages)  # Copy to allow modification

    while True:
        try:
            if deadline_timeout_s is not None and deadline_timeout_s > 0:
                remaining = (start + deadline_timeout_s) - time.monotonic()
                if remaining <= 0:
                    details = _timeout_error_details(
                        body,
                        model=model,
                        started_at=start,
                        timeout_s=deadline_timeout_s,
                    )
                    raise ProxyError(
                        504,
                        f"Request timed out after {deadline_timeout_s}s before provider call completed",
                        "timeout_error",
                        details=details,
                    )
                async with asyncio.timeout(remaining):
                    result = await _router.complete(model, working_messages, **kwargs)
            else:
                result = await _router.complete(model, working_messages, **kwargs)

            if stream:
                heartbeat_interval_s = _stream_heartbeat_interval_s(body)
                progress_events = bool(body.get("stream_progress_events") or body.get("progress_events"))
                # OAuth providers return AsyncIterator[bytes] (already SSE-formatted).
                # The openai SDK returns its own async stream type.
                if hasattr(result, "__aiter__") and not hasattr(result, "response"):
                    # Raw byte stream from OAuth providers — pipe directly,
                    # through the gpt-5.6 tool_call index renumber pass.
                    stream_iter = sse_liveness_wrapper(
                        renumber_tool_call_deltas(result),
                        model=model,
                        started_at=start,
                        overall_timeout_s=deadline_timeout_s,
                        heartbeat_interval_s=heartbeat_interval_s,
                        progress_events=progress_events,
                    )
                else:
                    stream_iter = sse_liveness_wrapper(
                        _sse_generator(result, model=model),
                        model=model,
                        started_at=start,
                        overall_timeout_s=deadline_timeout_s,
                        heartbeat_interval_s=heartbeat_interval_s,
                        progress_events=progress_events,
                    )
                response = StreamingResponse(
                    _stream_with_middleware_lifecycle(stream_iter, body, _middleware_chain),
                    media_type="text/event-stream",
                    headers=SSE_HEADERS,
                )
                return response
            else:
                # Non-streaming: result is a ChatCompletion object
                response_dict = result.model_dump()

                # Detect thinking-model token exhaustion:
                # finish_reason="length" with no visible content means the
                # provider spent its output budget internally without producing
                # an answer. OpenCode Go DeepSeek V4 can report thousands of
                # output tokens in this state, so do not require usage=0.
                usage = response_dict.get("usage") or {}
                if _is_empty_length_response(response_dict):
                    req_max = body.get("max_tokens", "unset")
                    total = usage.get("total_tokens", "?")
                    completion = usage.get("completion_tokens", "?")
                    raise ProxyError(
                        502,
                        f"Provider exhausted output budget with no visible response "
                        f"(max_tokens={req_max}, completion_tokens={completion}, "
                        f"total_tokens={total}). Use a shorter prompt, a stricter "
                        f"final-only JSON prompt, or a non-thinking model.",
                        "thinking_budget_exhausted",
                    )

                if _is_empty_zero_usage_response(response_dict):
                    raise ProxyError(
                        502,
                        "Provider returned no visible response and zero prompt/completion tokens; "
                        "rejecting empty 200 false green.",
                        "provider_empty_zero_usage_response",
                        details={
                            "model": response_dict.get("model", model),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )

                _attach_proof_fields(body, response_dict, working_messages)
                response_dict = await _middleware_chain.run_post_call(body, response_dict)

                # Echo opaque metadata back — LLM never saw it, can't fabricate it
                if caller_metadata is not None:
                    response_dict["scillm_metadata"] = caller_metadata

                elapsed = time.monotonic() - start

                # Extract cost headers stashed by CostHeaderMiddleware
                cost_headers = response_dict.pop("_cost_headers", {})
                # Extract timeout headers stashed by TimeoutEstimatorMiddleware
                timeout_headers = response_dict.pop("_timeout_headers", {})
                # Extract grounding headers stashed by GroundingGuard
                grounding_headers = response_dict.pop("_grounding_headers", {})

                resp_headers = {
                    "x-request-id": request_id,
                    "x-latency-ms": str(int(elapsed * 1000)),
                }
                resp_headers.update(cost_headers)
                resp_headers.update(timeout_headers)
                resp_headers.update(grounding_headers)

                return JSONResponse(
                    content=response_dict,
                    headers=resp_headers,
                )

        except asyncio.CancelledError as exc:
            await _middleware_chain.run_on_error(body, exc)
            raise
        except (ProxyError, MiddlewareReject) as exc:
            await _middleware_chain.run_on_error(body, exc)
            raise
        except RouterProxyError as exc:
            # Convert router's ProxyError to enriched ProxyError for LLM analysis
            details = getattr(exc, "details", {}) or {}
            if details.get("error_type"):
                error_type = str(details["error_type"])
            elif details.get("provider_error_code") == "PROVIDER_AUTH_FAILED":
                error_type = "provider_auth_error"
            elif details.get("provider_error_code"):
                error_type = "provider_error"
            else:
                error_type = "timeout_error" if exc.status_code == 504 else "router_error"
            proxy_exc = ProxyError(
                exc.status_code,
                exc.message,
                error_type,
                details=details,
            )
            await _middleware_chain.run_on_error(body, proxy_exc)
            raise proxy_exc
        except TimeoutError as exc:
            details = _timeout_error_details(
                body,
                model=model,
                started_at=start,
                timeout_s=deadline_timeout_s or 0,
                final_provider_error=exc,
            )
            proxy_exc = ProxyError(
                504,
                f"Request timed out after {deadline_timeout_s}s",
                "timeout_error",
                details=details,
            )
            await _middleware_chain.run_on_error(body, proxy_exc)
            raise proxy_exc
        except Exception as exc:
            # Import retry exceptions here to avoid circular import at module level
            from chutes.middleware.json_guard import JsonValidationFailed

            # Check for schema/grounding retry requests (course correction)
            try:
                from chutes.middleware.schema_guard import SchemaRetryNeeded, SchemaValidationFailed
                from chutes.middleware.grounding_guard import GroundingRetryNeeded
            except ImportError:
                SchemaRetryNeeded = None
                SchemaValidationFailed = None
                GroundingRetryNeeded = None

            # Handle Schema course correction retry
            if SchemaRetryNeeded and isinstance(exc, SchemaRetryNeeded):
                correction_attempt += 1
                if correction_attempt >= MAX_CORRECTION_ATTEMPTS:
                    logger.error(
                        "[{}] Schema validation max corrections exceeded ({})",
                        request_id, correction_attempt,
                    )
                    proxy_exc = ProxyError(
                        422,
                        f"Schema validation failed after {correction_attempt} correction attempts. "
                        f"Last correction hint: {exc.hint[:300]}",
                        "schema_validation_failed",
                    )
                    await _middleware_chain.run_on_error(body, proxy_exc)
                    raise proxy_exc
                logger.info(
                    "[{}] Schema validation failed, retrying with correction (attempt {})",
                    request_id, correction_attempt,
                )
                # Append correction prompt and retry
                working_messages = list(working_messages) + [
                    {"role": "user", "content": exc.hint}
                ]
                body["messages"] = working_messages
                continue  # Retry the completion

            # Handle Schema validation final failure
            if SchemaValidationFailed and isinstance(exc, SchemaValidationFailed):
                error_details = "; ".join(
                    e.get("message", str(e)) for e in exc.errors[:5]
                )
                proxy_exc = ProxyError(
                    422,
                    f"Schema validation failed: {error_details}",
                    "schema_validation_failed",
                    advice="Fix your prompt to return JSON matching the schema, or remove json_schema from request.",
                )
                await _middleware_chain.run_on_error(body, proxy_exc)
                raise proxy_exc

            # Handle Grounding course correction retry
            if GroundingRetryNeeded and isinstance(exc, GroundingRetryNeeded):
                correction_attempt += 1
                if correction_attempt >= MAX_CORRECTION_ATTEMPTS:
                    logger.error(
                        "[{}] Grounding max corrections exceeded ({})",
                        request_id, correction_attempt,
                    )
                    proxy_exc = ProxyError(
                        422,
                        f"Grounding verification failed after {correction_attempt} correction attempts. "
                        f"Response not sufficiently grounded in source text.",
                        "grounding_failed",
                    )
                    await _middleware_chain.run_on_error(body, proxy_exc)
                    raise proxy_exc
                logger.info(
                    "[{}] Grounding failed, retrying with correction (attempt {})",
                    request_id, correction_attempt,
                )
                # Append correction prompt and retry
                working_messages = list(working_messages) + [
                    {"role": "user", "content": exc.hint}
                ]
                body["messages"] = working_messages
                continue  # Retry the completion

            if isinstance(exc, JsonValidationFailed):
                logger.warning(
                    "[{}] JSON validation failed for model={}, signalling upstream error",
                    request_id,
                    model,
                )
                proxy_exc = ProxyError(
                    502,
                    f"JSON validation failed after repair: {exc.raw_text[:200]}",
                    "json_validation_error",
                )
                await _middleware_chain.run_on_error(body, proxy_exc)
                raise proxy_exc
            await _middleware_chain.run_on_error(body, exc)
            raise


@app.get("/v1/scillm/model-pools")
async def scillm_model_pools(request: Request):
    """List server-side model pools for concurrent batch routing."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    concurrency = {}
    try:
        from chutes.middleware.concurrency_guard import get_concurrency_status
        concurrency = get_concurrency_status()
    except ImportError:
        pass

    return {
        "pools": {
            name: {
                **pool,
                "lanes": [dict(lane) for lane in pool["lanes"]],
                "status": _model_pool_status(name, pool, concurrency),
            }
            for name, pool in _DEFAULT_MODEL_POOLS.items()
        }
    }


@app.get("/v1/scillm/model-pools/{pool_name}/status")
async def scillm_model_pool_status(request: Request, pool_name: str):
    """Return live aggregate and per-lane concurrency for one model pool."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    pool = _model_pool(pool_name)
    if pool is None:
        raise ProxyError(
            404,
            f"Unknown model_pool '{pool_name}'. Available: {', '.join(sorted(_DEFAULT_MODEL_POOLS))}.",
            "not_found",
        )

    concurrency = {}
    try:
        from chutes.middleware.concurrency_guard import get_concurrency_status
        concurrency = get_concurrency_status()
    except ImportError:
        pass

    return _model_pool_status(pool_name, pool, concurrency)


@app.post("/v1/images/generations")
async def image_generations(request: Request):
    """OpenAI-style image generation endpoint (Chutes z-image-turbo or GPT image models)."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")
    caller_skill = _require_caller_skill(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ProxyError(400, "image generation body must be a JSON object", "invalid_request_error")
    return await _generate_images_for_body(body, caller_skill=caller_skill)


@app.post("/v1/scillm/batch/images/generations")
async def scillm_batch_image_generations(request: Request):
    """Run image generations concurrently using asyncio.create_task/as_completed."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")
    caller_skill = _require_caller_skill(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise ProxyError(400, "batch image generation body must be a JSON object", "invalid_request_error")
    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ProxyError(400, "items must be a non-empty list", "invalid_request_error")
    if any(not isinstance(item, dict) for item in items):
        raise ProxyError(400, "each item must be an object", "invalid_request_error")
    max_items = int(body.get("max_items") or 100)
    if len(items) > max_items:
        raise ProxyError(400, f"batch has {len(items)} items; max_items is {max_items}", "invalid_request_error")
    max_concurrency = int(body.get("max_concurrency") or 4)
    if max_concurrency < 1 or max_concurrency > 16:
        raise ProxyError(400, "max_concurrency must be between 1 and 16", "invalid_request_error")

    batch_id = str(body.get("batch_id") or f"image-batch-{uuid.uuid4().hex[:12]}")
    shared = {
        key: body[key]
        for key in (
            "model",
            "size",
            "width",
            "height",
            "response_format",
            "quality",
            "background",
            "output_format",
            "output_compression",
            "moderation",
            "num_inference_steps",
            "steps",
            "guidance_scale",
            "shift",
            "max_sequence_length",
            "timeout",
        )
        if key in body
    }
    semaphore = asyncio.Semaphore(max_concurrency)
    started = time.monotonic()

    async def run_one(item: dict[str, Any], index: int) -> dict[str, Any]:
        item_id = _item_id(item, index)
        item_started = time.monotonic()
        payload = {**shared, **item, "n": item.get("n", 1)}
        payload["scillm_metadata"] = {
            **(item.get("scillm_metadata") if isinstance(item.get("scillm_metadata"), dict) else {}),
            "batch_id": batch_id,
            "item_id": item_id,
            "provider": "chutes",
            "selected_model": str(payload.get("model") or "z-image-turbo"),
        }
        async with semaphore:
            try:
                response = await _generate_images_for_body(payload, caller_skill=caller_skill)
                return {
                    "ok": True,
                    "item_id": item_id,
                    "index": index,
                    "model": response.get("model"),
                    "provider": response.get("provider"),
                    "data": response.get("data", []),
                    "scillm_metadata": payload["scillm_metadata"],
                    "latency_s": round(time.monotonic() - item_started, 2),
                }
            except ProxyError as exc:
                return {
                    "ok": False,
                    "item_id": item_id,
                    "index": index,
                    "error": exc.message,
                    "error_type": exc.error_type,
                    "status_code": exc.status_code,
                    "scillm_metadata": payload["scillm_metadata"],
                    "latency_s": round(time.monotonic() - item_started, 2),
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "item_id": item_id,
                    "index": index,
                    "error": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                    "scillm_metadata": payload["scillm_metadata"],
                    "latency_s": round(time.monotonic() - item_started, 2),
                }

    tasks = [asyncio.create_task(run_one(item, index)) for index, item in enumerate(items)]
    results = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    completed = sum(1 for result in results if result.get("ok"))
    failed = len(results) - completed
    return {
        "batch_id": batch_id,
        "object": "list",
        "provider": "chutes",
        "total": len(results),
        "completed": completed,
        "failed": failed,
        "elapsed_s": round(time.monotonic() - started, 2),
        "results": results,
        "ordered": False,
        "completion_order": "as_completed",
    }


@app.post("/v1/scillm/batch/completions")
async def scillm_batch_completions(request: Request):
    """Run a batch across a weighted provider pool using ``asyncio.as_completed``."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None or _router is None or _middleware_chain is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")

    body = await request.json()
    pool_name = str(body.get("model_pool") or body.get("model") or "qra-deepseek-pool")
    pool = _model_pool(pool_name)
    if pool is None:
        raise ProxyError(
            400,
            f"Unknown model_pool '{pool_name}'. Available: {', '.join(sorted(_DEFAULT_MODEL_POOLS))}.",
            "invalid_request_error",
        )

    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ProxyError(400, "items must be a non-empty list", "invalid_request_error")
    max_items = int(body.get("max_items") or 500)
    if len(items) > max_items:
        raise ProxyError(400, f"batch has {len(items)} items; max_items is {max_items}", "invalid_request_error")

    batch_id = str(body.get("batch_id") or f"batch-{uuid.uuid4().hex[:12]}")
    lane_offset = int(body.get("lane_offset") or body.get("start_index") or 0)
    lanes = pool["lanes"]
    concurrency_snapshot = {}
    if body.get("spill_to_available_lane", True):
        try:
            from chutes.middleware.concurrency_guard import get_concurrency_status
            concurrency_snapshot = get_concurrency_status()
        except ImportError:
            concurrency_snapshot = {}
    lane_semaphores = {
        lane["name"]: asyncio.Semaphore(int(lane.get("max_concurrency") or 1))
        for lane in lanes
    }
    auth_header = request.headers.get("authorization", "")
    caller_skill = request.headers.get("x-caller-skill", "scillm-batch-pool")
    shared_params = {key: body[key] for key in _BATCH_FORWARD_KEYS if key in body}
    shared_params["stream"] = False
    total_started = time.monotonic()

    async def run_one(client: httpx.AsyncClient, item: dict[str, Any], index: int) -> dict[str, Any]:
        lane = _lane_for_index_with_capacity(lanes, index + lane_offset, concurrency_snapshot)
        item_id = _item_id(item, index)
        started = time.monotonic()
        async with lane_semaphores[lane["name"]]:
            try:
                payload = {
                    **shared_params,
                    **{key: item[key] for key in _BATCH_FORWARD_KEYS if key in item},
                    "model": lane["model"],
                    "messages": _messages_for_batch_item(item),
                    "_scillm_pool": pool_name,
                    "_scillm_pool_lane": lane["name"],
                    "_scillm_require_hot_chutes": bool(lane.get("require_hot_chutes")),
                }
                if body.get("wait_for_hot_chutes") is not None:
                    payload["_scillm_wait_for_hot_chutes"] = body.get("wait_for_hot_chutes")
                if body.get("allow_cold_chutes") is not None:
                    payload["_scillm_allow_cold_chutes"] = body.get("allow_cold_chutes")
                item_metadata = item.get("scillm_metadata") if isinstance(item.get("scillm_metadata"), dict) else {}
                payload["scillm_metadata"] = {
                    **item_metadata,
                    "batch_id": batch_id,
                    "item_id": item_id,
                    "model_pool": pool_name,
                    "lane": lane["name"],
                    "selected_model": lane["model"],
                    "provider": lane.get("provider"),
                    "require_hot_chutes": bool(lane.get("require_hot_chutes")),
                }
                timeout = float(item.get("timeout") or body.get("timeout") or lane.get("timeout") or 300.0)
                payload["timeout"] = timeout
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": auth_header,
                        "X-Caller-Skill": caller_skill,
                        "X-Scillm-Batch-Id": batch_id,
                        "X-Scillm-Batch-Total": str(len(items)),
                        "X-Scillm-Call-Key": item_id,
                    },
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = _chat_completion_text_content(data)
                return {
                    "ok": bool(content.strip()),
                    "item_id": item_id,
                    "index": index,
                    "lane": lane["name"],
                    "provider": lane.get("provider"),
                    "model": lane["model"],
                    "served_model": data.get("model"),
                    "content": content,
                    "error": None if content.strip() else "empty_response_content",
                    "response": data,
                    "latency_s": round(time.monotonic() - started, 2),
                }
            except httpx.HTTPStatusError as exc:
                error = str(exc)
                try:
                    error_body = exc.response.json()
                    if isinstance(error_body, dict) and isinstance(error_body.get("error"), dict):
                        error = error_body["error"].get("message") or error
                except Exception:
                    error = exc.response.text or error
                return {
                    "ok": False,
                    "item_id": item_id,
                    "index": index,
                    "lane": lane["name"],
                    "provider": lane.get("provider"),
                    "model": lane["model"],
                    "error": error,
                    "status_code": exc.response.status_code,
                    "latency_s": round(time.monotonic() - started, 2),
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "item_id": item_id,
                    "index": index,
                    "lane": lane["name"],
                    "provider": lane.get("provider"),
                    "model": lane["model"],
                    "error": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                    "latency_s": round(time.monotonic() - started, 2),
                }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://scillm.internal") as client:
        tasks = [
            asyncio.create_task(run_one(client, item, index))
            for index, item in enumerate(items)
            if isinstance(item, dict)
        ]
        if len(tasks) != len(items):
            raise ProxyError(400, "each item must be an object", "invalid_request_error")
        results = []
        for task in asyncio.as_completed(tasks):
            results.append(await task)

    completed = sum(1 for result in results if result.get("ok"))
    failed = len(results) - completed
    return {
        "batch_id": batch_id,
        "model_pool": pool_name,
        "strategy": pool["strategy"],
        "lanes": lanes,
        "total": len(results),
        "completed": completed,
        "failed": failed,
        "elapsed_s": round(time.monotonic() - total_started, 2),
        "results": results,
        "ordered": False,
        "completion_order": "as_completed",
    }


@app.post("/v1/scillm/batch/completions/stream")
async def scillm_batch_completions_stream(request: Request):
    """Stream model-pool batch progress and item results as SSE events."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None or _router is None or _middleware_chain is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")

    body = await request.json()
    pool_name = str(body.get("model_pool") or body.get("model") or "qra-deepseek-pool")
    pool = _model_pool(pool_name)
    if pool is None:
        raise ProxyError(
            400,
            f"Unknown model_pool '{pool_name}'. Available: {', '.join(sorted(_DEFAULT_MODEL_POOLS))}.",
            "invalid_request_error",
        )

    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ProxyError(400, "items must be a non-empty list", "invalid_request_error")
    max_items = int(body.get("max_items") or 500)
    if len(items) > max_items:
        raise ProxyError(400, f"batch has {len(items)} items; max_items is {max_items}", "invalid_request_error")
    if any(not isinstance(item, dict) for item in items):
        raise ProxyError(400, "each item must be an object", "invalid_request_error")

    batch_id = str(body.get("batch_id") or f"batch-{uuid.uuid4().hex[:12]}")
    lane_offset = int(body.get("lane_offset") or body.get("start_index") or 0)
    lanes = pool["lanes"]
    concurrency_snapshot = {}
    if body.get("spill_to_available_lane", True):
        try:
            from chutes.middleware.concurrency_guard import get_concurrency_status
            concurrency_snapshot = get_concurrency_status()
        except ImportError:
            concurrency_snapshot = {}
    lane_semaphores = {
        lane["name"]: asyncio.Semaphore(int(lane.get("max_concurrency") or 1))
        for lane in lanes
    }
    auth_header = request.headers.get("authorization", "")
    caller_skill = request.headers.get("x-caller-skill", "scillm-batch-pool")
    heartbeat_s = float(body.get("stream_heartbeat_s") or body.get("heartbeat_interval_s") or DEFAULT_STREAM_HEARTBEAT_S)
    shared_params = {key: body[key] for key in _BATCH_FORWARD_KEYS if key in body}
    shared_params["stream"] = True
    shared_params["stream_progress_events"] = bool(body.get("stream_progress_events", True))
    shared_params["stream_heartbeat_s"] = heartbeat_s
    shared_params["stream_options"] = {"include_usage": True}
    total_started = time.monotonic()

    async def run_one(client: httpx.AsyncClient, item: dict[str, Any], index: int) -> dict[str, Any]:
        lane = _lane_for_index_with_capacity(lanes, index + lane_offset, concurrency_snapshot)
        item_id = _item_id(item, index)
        started = time.monotonic()
        async with lane_semaphores[lane["name"]]:
            try:
                payload = {
                    **shared_params,
                    **{key: item[key] for key in _BATCH_FORWARD_KEYS if key in item},
                    "model": lane["model"],
                    "messages": _messages_for_batch_item(item),
                    "_scillm_pool": pool_name,
                    "_scillm_pool_lane": lane["name"],
                    "_scillm_require_hot_chutes": bool(lane.get("require_hot_chutes")),
                }
                if body.get("wait_for_hot_chutes") is not None:
                    payload["_scillm_wait_for_hot_chutes"] = body.get("wait_for_hot_chutes")
                if body.get("allow_cold_chutes") is not None:
                    payload["_scillm_allow_cold_chutes"] = body.get("allow_cold_chutes")
                item_metadata = item.get("scillm_metadata") if isinstance(item.get("scillm_metadata"), dict) else {}
                payload["scillm_metadata"] = {
                    **item_metadata,
                    "batch_id": batch_id,
                    "item_id": item_id,
                    "model_pool": pool_name,
                    "lane": lane["name"],
                    "selected_model": lane["model"],
                    "provider": lane.get("provider"),
                    "require_hot_chutes": bool(lane.get("require_hot_chutes")),
                }
                timeout = float(item.get("timeout") or body.get("timeout") or lane.get("timeout") or 300.0)
                payload["timeout"] = timeout
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={
                        "Authorization": auth_header,
                        "X-Caller-Skill": caller_skill,
                        "X-Scillm-Batch-Id": batch_id,
                        "X-Scillm-Batch-Total": str(len(items)),
                        "X-Scillm-Call-Key": item_id,
                    },
                    json=payload,
                    timeout=timeout + 30.0,
                ) as response:
                    if response.status_code != 200:
                        text = (await response.aread()).decode("utf-8", errors="replace")
                        return {
                            "ok": False,
                            "item_id": item_id,
                            "index": index,
                            "lane": lane["name"],
                            "provider": lane.get("provider"),
                            "model": lane["model"],
                            "error": text,
                            "status_code": response.status_code,
                            "latency_s": round(time.monotonic() - started, 2),
                        }
                    data = await _collect_chat_sse_lines(response.aiter_lines(), requested_model=lane["model"])
                content = _chat_completion_text_content(data)
                return {
                    "ok": bool(content.strip()),
                    "item_id": item_id,
                    "index": index,
                    "lane": lane["name"],
                    "provider": lane.get("provider"),
                    "model": lane["model"],
                    "served_model": data.get("model"),
                    "content": content,
                    "error": None if content.strip() else "empty_response_content",
                    "response": data,
                    "latency_s": round(time.monotonic() - started, 2),
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "item_id": item_id,
                    "index": index,
                    "lane": lane["name"],
                    "provider": lane.get("provider"),
                    "model": lane["model"],
                    "error": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                    "latency_s": round(time.monotonic() - started, 2),
                }

    async def event_stream() -> AsyncIterator[str]:
        transport = httpx.ASGITransport(app=app)
        completed = 0
        failed = 0
        yield _batch_sse_event(
            "batch_started",
            {
                "batch_id": batch_id,
                "model_pool": pool_name,
                "strategy": pool["strategy"],
                "total": len(items),
                "stream": True,
                "stream_heartbeat_s": heartbeat_s,
            },
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://scillm.internal") as client:
            tasks: list[asyncio.Task[dict[str, Any]]] = []
            for index, item in enumerate(items):
                lane = _lane_for_index(lanes, index + lane_offset)
                item_id = _item_id(item, index)
                yield _batch_sse_event(
                    "item_started",
                    {
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "index": index,
                        "lane": lane["name"],
                        "provider": lane.get("provider"),
                        "model": lane["model"],
                    },
                )
                tasks.append(asyncio.create_task(run_one(client, item, index)))

            pending: set[asyncio.Task[dict[str, Any]]] = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, timeout=heartbeat_s, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    yield _batch_sse_event(
                        "heartbeat",
                        {
                            "batch_id": batch_id,
                            "model_pool": pool_name,
                            "elapsed_ms": int((time.monotonic() - total_started) * 1000),
                            "completed": completed,
                            "failed": failed,
                            "pending": len(pending),
                        },
                    )
                    continue
                for task in done:
                    result = await task
                    if result.get("ok"):
                        completed += 1
                        yield _batch_sse_event("item_completed", result)
                    else:
                        failed += 1
                        yield _batch_sse_event("item_failed", result)

        yield _batch_sse_event(
            "batch_done",
            {
                "batch_id": batch_id,
                "model_pool": pool_name,
                "strategy": pool["strategy"],
                "total": len(items),
                "completed": completed,
                "failed": failed,
                "elapsed_s": round(time.monotonic() - total_started, 2),
                "ordered": False,
                "completion_order": "as_completed",
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/v1/scillm/health")
async def scillm_health(request: Request):
    """Detailed health: router status, fallback config, uptime."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    if _config is None or _router is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")
    uptime = time.monotonic() - _start_time

    # Effective master-key provenance (issue #32): callers must be able to
    # tell WHICH key the live process accepts without guessing from 401s.
    import hashlib
    _mk = _config.general.master_key or ""
    master_key_status = {
        "fingerprint_sha256_12": hashlib.sha256(_mk.encode()).hexdigest()[:12] if _mk else None,
        "is_dev_default": _mk == "sk-dev-proxy-123",
        "source": "container env SCILLM_MASTER_KEY (compose env_file ../../.env is authoritative)",
    }

    # Concurrency status (optional)
    concurrency = {}
    try:
        from chutes.middleware.concurrency_guard import get_concurrency_status
        concurrency = get_concurrency_status()
    except ImportError:
        pass

    # Abuse guard status (optional)
    abuse_guard = {}
    try:
        from chutes.middleware.abuse_guard import get_abuse_status
        abuse_guard = get_abuse_status()
    except ImportError:
        pass

    model_pools = {
        name: _model_pool_status(name, pool, concurrency)
        for name, pool in _DEFAULT_MODEL_POOLS.items()
    }

    return {
        "master_key": master_key_status,
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "model_groups": list(_config.model_groups.keys()),
        "fallbacks": _config.fallbacks,
        "retry_policy": {
            "internal_server_error": _config.retry_policy.internal_server_error,
            "rate_limit_error": _config.retry_policy.rate_limit_error,
            "timeout_error": _config.retry_policy.timeout_error,
        },
        "routing_strategy": _config.routing_strategy,
        "circuit_breaker": _router.circuit_status(),
        "concurrency": concurrency,
        "model_pools": model_pools,
        "abuse_guard": abuse_guard,
    }


@app.get("/v1/scillm/active-calls")
async def scillm_active_calls(request: Request):
    """Return list of currently in-flight LLM calls for live monitoring."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.active_calls import get_active_calls, get_recent_stale_active_calls
        active = get_active_calls()
        stale = get_recent_stale_active_calls()
        return {
            "active": active,
            "live_in_flight": len(active),
            "stale_active_calls": len(stale),
            "stale": stale,
        }
    except ImportError:
        return {
            "active": [],
            "live_in_flight": 0,
            "stale_active_calls": 0,
            "stale": [],
            "error": "ActiveCallsMiddleware not loaded",
        }


@app.post("/v1/scillm/active-calls/purge")
async def scillm_active_calls_purge(
    request: Request,
    older_than_s: float | None = 600.0,
    caller: str | None = None,
    model_contains: str | None = None,
):
    """Purge stale active-call rows from the in-memory live registry."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.active_calls import clear_active_calls
        purged = clear_active_calls(
            older_than_s=older_than_s,
            caller=caller,
            model_contains=model_contains,
        )
        return {"purged": len(purged), "active": purged}
    except ImportError:
        return {"purged": 0, "active": [], "error": "ActiveCallsMiddleware not loaded"}


@app.get("/v1/scillm/activity")
async def scillm_activity(request: Request):
    """Return activity graph data (last 5 minutes, bucketed for charting)."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.active_calls import get_activity_graph
        return get_activity_graph()
    except ImportError:
        return {"buckets": [], "error": "ActiveCallsMiddleware not loaded"}


@app.get("/v1/scillm/concurrency")
async def scillm_concurrency(request: Request, model: str = ""):
    """Get concurrency info for a model (for batch sizing).

    Skills call this to determine optimal chunk_size for batch processing.
    Returns effective_limit accounting for adaptive backoff from 429s.

    Example: GET /v1/scillm/concurrency?model=Qwen/Qwen3.6-27B-TEE
    Response: {"model": "Qwen/Qwen3.6-27B-TEE", "provider": "chutes", "chunk_size": 4, ...}
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if not model:
        raise ProxyError(
            400,
            "model query parameter is required; Chutes calls must specify an exact live model ID",
            "invalid_request_error",
        )

    try:
        from chutes.middleware.concurrency_guard import get_concurrency_for_model
        return get_concurrency_for_model(model)
    except ImportError:
        # Fallback if concurrency guard not loaded
        return {
            "model": model,
            "provider": "unknown",
            "chunk_size": 4,  # Safe default
            "error": "ConcurrencyGuard not loaded",
        }


@app.post("/v1/scillm/concurrency/reset")
async def scillm_concurrency_reset(request: Request, provider: str = ""):
    """Reset concurrency state for a provider or all providers.

    Use when batch failures have corrupted state or queue is stuck.
    Clears in-flight counters, queue depth, pauses, and backoff state.

    Example: POST /v1/scillm/concurrency/reset?provider=chutes
    Example: POST /v1/scillm/concurrency/reset  (reset all)
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.concurrency_guard import reset_concurrency
        result = reset_concurrency(provider if provider else None)
        return {
            "ok": True,
            "message": f"Reset concurrency for {len(result['providers_reset'])} providers",
            **result,
        }
    except ImportError:
        return {"ok": False, "error": "ConcurrencyGuard not loaded"}


@app.post("/v1/scillm/abuse-guard/reset")
async def scillm_abuse_guard_reset(request: Request):
    """Reset abuse guard — unblock all clients and clear error history.

    Use when a failed batch has incorrectly blocked your API key.
    Abuse guard blocks clients after 5+ 4xx errors in 30 seconds.

    Example: POST /v1/scillm/abuse-guard/reset
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.abuse_guard import reset_abuse_guard
        result = reset_abuse_guard()
        return result
    except ImportError:
        return {"ok": False, "error": "AbuseGuard not loaded"}


@app.get("/v1/scillm/models")
async def scillm_models(
    request: Request,
    include_models_dev: bool = False,
    refresh_models_dev: bool = False,
    refresh_provider_models: bool = False,
):
    """List model groups, aliases, and UI-safe model selection contracts."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    if _config is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")
    groups = {}
    for name, group in _config.model_groups.items():
        groups[name] = {
            "deployments": len(group.deployments),
            "models": [d.model for d in group.deployments],
        }
    groups["embedding"] = {
        "deployments": 1,
        "models": ["all-MiniLM-L6-v2"],
        "endpoint": EMBEDDING_SERVICE_URL,
    }
    models = _build_model_catalog()
    provider_catalogs = {
        "claude": _provider_model_catalog("claude", refresh=refresh_provider_models),
        "codex": _provider_model_catalog("codex", refresh=refresh_provider_models),
        OPENCODE_GO_PROVIDER: _provider_model_catalog(OPENCODE_GO_PROVIDER, refresh=refresh_provider_models),
    }
    for provider, catalog in provider_catalogs.items():
        for row in catalog.get("models", []):
            if not isinstance(row, dict):
                continue
            model_id = row.get("id")
            if not isinstance(model_id, str) or model_id in models:
                continue
            models[model_id] = _model_catalog_entry(
                name=model_id,
                kind="provider_model",
                target=model_id,
                deployments=1,
                capabilities=_public_model_capabilities(model_id),
            )
            models[model_id]["provider"] = provider
            if "reasoning_efforts" in row:
                models[model_id]["reasoning_efforts"] = row["reasoning_efforts"]
    models_dev_error: str | None = None
    if include_models_dev:
        try:
            models_dev_catalog = await _fetch_models_dev_catalog(refresh=refresh_models_dev)
            for name, entry in models.items():
                advisory = _models_dev_advisory_for_name(models_dev_catalog, name, entry)
                if advisory is not None:
                    entry["models_dev"] = advisory
        except Exception as exc:
            models_dev_error = str(exc)
    selectable_models = [
        name for name, entry in models.items()
        if bool(entry.get("selectable"))
    ]
    review_fanout_models = [
        name for name in _REVIEW_FANOUT_MODEL_PREFERENCE
        if name in models and bool(models[name].get("selectable"))
    ]
    return {
        "groups": groups,
        "aliases": _config.aliases,
        "models": models,
        "provider_catalogs": provider_catalogs,
        "selectable_models": selectable_models,
        "review_fanout_models": review_fanout_models,
        "retired_model_prefixes": list(_RETIRED_MODEL_PREFIXES),
        "models_dev": {
            "included": include_models_dev,
            "advisory_only": True,
            "source": MODELS_DEV_API_URL,
            "error": models_dev_error,
        },
        "selection_contract": {
            "review_fanout": (
                "Use review_fanout_models for default scoped review nodes. Do not "
                "offer retired text-* aliases for review, "
                "prompt, or production DAG fanout nodes."
            ),
            "fallbacks": "Fallbacks must stay within the same provider family unless explicitly configured otherwise.",
        },
        "auto_providers": {
            "codex": {
                "model_prefixes": ["gpt-", "codex-", "o1", "o3", "o4"],
                "examples": _model_ids_from_catalog(provider_catalogs["codex"])[:5] or ["gpt-5.5"],
                "key_configured": _codex_oauth_available(),
            },
            "claude": {
                "model_prefixes": ["claude-"],
                "examples": _model_ids_from_catalog(provider_catalogs["claude"])[:5],
                "key_configured": _claude_oauth_available(),
            },
            OPENCODE_GO_PROVIDER: {
                "model_prefix": "opencode-go/",
                "models_endpoint": "/v1/scillm/opencode-go/models",
                "key_configured": bool(_config.opencode_go_api_key),
            }
        },
    }


@app.get("/v1/scillm/models-dev")
async def scillm_models_dev(
    request: Request,
    provider: str | None = None,
    model: str | None = None,
    refresh: bool = False,
):
    """Return advisory models.dev metadata without changing scillm routing."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        catalog = await _fetch_models_dev_catalog(refresh=refresh)
    except Exception as exc:
        raise ProxyError(
            502,
            f"models.dev catalog fetch failed: {exc}",
            "provider_error",
            details={"source": MODELS_DEV_API_URL, "advisory_only": True},
        )

    extracted = _models_dev_extract(catalog, provider=provider, model=model)
    extracted["source"] = MODELS_DEV_API_URL
    extracted["routing_source_of_truth"] = "/v1/scillm/models and provider-specific scillm capability endpoints"
    return extracted


@app.get("/v1/scillm/capabilities")
async def scillm_capabilities(request: Request):
    """Read-only capability facts for callers, adapters, pools, and policies."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None:
        raise ProxyError(503, "Proxy not ready", "service_unavailable")

    adapters = _load_adapter_manifests()
    model_groups: dict[str, dict[str, Any]] = {}
    for name, group in _config.model_groups.items():
        deployments = []
        support_items = []
        for dep in group.deployments:
            adapter_id = _adapter_id_for_deployment(dep, name)
            supports = _deployment_supports(dep, name, adapters)
            support_items.append(supports)
            deployments.append({
                "model": dep.model,
                "provider": adapter_id,
                "custom_llm_provider": dep.custom_llm_provider,
                "supports": supports,
                "timeout": dep.timeout,
            })
        model_groups[name] = {
            "supports": _merge_supports(support_items),
            "deployments": deployments,
            "fallbacks": _config.fallbacks.get(name, []),
        }

    pools = {}
    for name, pool in _DEFAULT_MODEL_POOLS.items():
        pools[name] = {
            "strategy": pool.get("strategy"),
            "description": pool.get("description"),
            "supports": {
                "batch": True,
                "text": True,
                "image": False,
                "pdf": False,
                "zip": False,
                "streaming": False,
                "tools": False,
            },
            "lanes": [dict(lane) for lane in pool.get("lanes", [])],
        }

    image_generation = {
        name: {
            "provider": config.get("provider", "chutes"),
            "endpoint": "/v1/images/generations",
            "batch_endpoint": "/v1/scillm/batch/images/generations",
            "model_slug": config.get("slug"),
            "supports": {
                "text": True,
                "image": False,
                "pdf": False,
                "zip": False,
                "streaming": False,
                "tools": False,
                "image_output": True,
                "image_generation": True,
                "batch": bool(config.get("supports_batch")),
            },
            "limits": config.get("limits", {}),
        }
        for name, config in _IMAGE_GENERATION_MODELS.items()
        if not config.get("alias_for")
    }

    try:
        from chutes.middleware.caller_policy import caller_profiles_for_capabilities
        caller_profiles = caller_profiles_for_capabilities(_config)
    except Exception:
        caller_profiles = {}

    return {
        "version": 1,
        "model_groups": model_groups,
        "aliases": _config.aliases,
        "fallbacks": _config.fallbacks,
        "model_pools": pools,
        "image_generation": image_generation,
        "adapters": adapters,
        "caller_profiles": caller_profiles,
    }


@app.get("/v1/scillm/opencode-go/models")
async def scillm_opencode_go_models(
    request: Request,
    source: str = "auto",
    refresh: bool = True,
    verbose: bool = False,
):
    """List OpenCode Go models available to scillm.

    ``source=auto`` defaults to live CLI refresh, then a running
    ``opencode serve`` instance, then the built-in registry.  ``source=server``,
    ``source=cli``, and ``source=static`` force one path.
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    if _config is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")

    source = source.lower()
    if source not in {"auto", "server", "cli", "static"}:
        raise ProxyError(400, "source must be one of: auto, server, cli, static", "invalid_request_error")

    errors: list[str] = []
    models: list[str] = []
    actual_source = "static"

    if source in {"auto", "cli"}:
        try:
            models = await list_opencode_go_models_from_cli(refresh=refresh, verbose=verbose)
            actual_source = "cli"
        except Exception as exc:
            errors.append(f"cli: {exc}")
            if source == "cli":
                raise ProxyError(502, f"opencode models failed: {exc}", "provider_error")

    if not models and source in {"auto", "server"}:
        try:
            models = await list_opencode_go_models_from_server()
            if models:
                actual_source = "server"
        except Exception as exc:
            errors.append(f"server: {exc}")
            if source == "server":
                raise ProxyError(502, f"OpenCode server model listing failed: {exc}", "provider_error")

    if not models:
        models = static_opencode_go_models()
        actual_source = "static"

    key_configured = bool(_config.opencode_go_api_key)
    return {
        "provider": OPENCODE_GO_PROVIDER,
        "source": actual_source,
        "refresh": refresh,
        "verbose": verbose,
        "key_configured": key_configured,
        "model_count": len(models),
        "models": [
            {
                **describe_opencode_go_model(model, key_configured=key_configured),
                "reasoning_efforts": [],
                "reasoning_source": "not_advertised",
            }
            for model in models
        ],
        "errors": errors,
    }


@app.get("/v1/models")
async def openai_models(request: Request):
    """OpenAI-compatible /v1/models endpoint."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    if _config is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")
    models = []
    for name in _config.model_groups:
        models.append({
            "id": name,
            "object": "model",
            "created": int(_start_time),
            "owned_by": "scillm",
        })
    for alias in _config.aliases:
        models.append({
            "id": alias,
            "object": "model",
            "created": int(_start_time),
            "owned_by": "scillm",
        })
    # Embedding model (served by local embedding service)
    models.append({
        "id": "embedding",
        "object": "model",
        "created": int(_start_time),
        "owned_by": "scillm",
    })
    for image_name, image_config in _IMAGE_GENERATION_MODELS.items():
        if image_config.get("alias_for"):
            continue
        models.append({
            "id": image_name,
            "object": "model",
            "created": int(_start_time),
            "owned_by": f"{image_config.get('provider', 'chutes')}-image-generation",
        })

    # Auto-routable providers — these don't need config entries
    from scillm.proxy.providers.auth import is_anthropic_available, is_codex_available
    auto_models = []
    if is_anthropic_available():
        for row in claude_catalog_payload()["models"]:
            auto_models.append({"id": row["id"], "object": "model", "created": int(_start_time), "owned_by": "anthropic-oauth"})
    if is_codex_available():
        discovered_codex = discover_codex_models()
        for m in [model.slug for model in discovered_codex] or ["gpt-5.5"]:
            auto_models.append({"id": m, "object": "model", "created": int(_start_time), "owned_by": "codex-oauth"})
    if _config.gemini_api_base:
        for m in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash-lite"]:
            auto_models.append({"id": m, "object": "model", "created": int(_start_time), "owned_by": "gemini"})
    if _config.ollama_api_base:
        auto_models.append({"id": "ollama:*", "object": "model", "created": int(_start_time), "owned_by": "ollama-auto"})
    if _config.chutes_api_base:
        auto_models.append({"id": "chutes:Org/Model", "object": "model", "created": int(_start_time), "owned_by": "chutes-auto"})
    if _config.opencode_go_api_key:
        opencode_catalog = _opencode_go_catalog_payload_for_validation()
        for model in _model_ids_from_catalog(opencode_catalog):
            auto_models.append({"id": model, "object": "model", "created": int(_start_time), "owned_by": OPENCODE_GO_PROVIDER})

    # Deduplicate (auto models might overlap with configured ones)
    existing_ids = {m["id"] for m in models}
    for m in auto_models:
        if m["id"] not in existing_ids:
            models.append(m)

    return {"object": "list", "data": models}


@app.get("/v1/scillm/providers")
async def scillm_providers(request: Request):
    """List all available providers and how to call them."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")
    if _config is None:
        raise ProxyError(503, "Proxy not ready", "service_unavailable")

    from scillm.proxy.providers.auth import is_anthropic_available, is_codex_available

    claude_catalog = _provider_model_catalog("claude")
    codex_catalog = _provider_model_catalog("codex")
    opencode_catalog = _provider_model_catalog(OPENCODE_GO_PROVIDER)
    codex_models = _model_ids_from_catalog(codex_catalog)
    providers = {
        "configured": {
            name: {
                "models": [d.model for d in group.deployments],
                "api_base": group.deployments[0].api_base if group.deployments else None,
            }
            for name, group in _config.model_groups.items()
        },
        "auto_routing": {
            "claude": {
                "available": is_anthropic_available(),
                "pattern": "claude-*",
                "examples": _model_ids_from_catalog(claude_catalog)[:5],
                "catalog": claude_catalog,
                "auth": "OAuth via ~/.claude/.credentials.json",
                "note": "System prompt locked to Claude Code prefix",
            },
            "codex": {
                "available": is_codex_available(),
                "pattern": "gpt-* | codex-*",
                "examples": codex_models[:5] or ["gpt-5.5"],
                "catalog": codex_catalog,
                "auth": "OAuth via ~/.codex/auth.json",
                "note": "temperature/max_tokens not supported",
            },
            OPENCODE_GO_PROVIDER: {
                "available": bool(_config.opencode_go_api_key),
                "pattern": "opencode-go/*",
                "examples": _model_ids_from_catalog(opencode_catalog)[:5],
                "catalog": opencode_catalog,
                "auth": "OPENCODE_GO_API_KEY",
                "note": "Use /v1/scillm/opencode-go/models?refresh=true for an explicit refresh.",
            },
            "gemini": {
                "available": bool(_config.gemini_api_base),
                "pattern": "gemini-*",
                "examples": ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-pro-preview"],
                "auth": "API key",
                "note": "Supports inlineData for PDFs/images/ZIP",
            },
            "chutes": {
                "available": bool(_config.chutes_api_base),
                "pattern": "Org/Model (contains /)",
                "examples": ["Qwen/Qwen3.6-27B-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"],
                "auth": "API key",
                "note": "No Chutes aliases; select exact live IDs with ops-chutes.",
            },
            "chutes_image_generation": {
                "available": bool(_config.chutes_api_key or os.environ.get("CHUTES_API_KEY") or os.environ.get("CHUTES_API_TOKEN")),
                "endpoint": "/v1/images/generations",
                "batch_endpoint": "/v1/scillm/batch/images/generations",
                "models": ["z-image-turbo"],
                "auth": "API key",
                "note": "Text-to-image output; do not route through vlm.",
            },
            "ollama": {
                "available": bool(_config.ollama_api_base),
                "pattern": "model:tag or unknown names",
                "examples": ["qwen2.5:7b", "qwen3:0.6b", "llama3:8b"],
                "auth": "none (local)",
                "note": "response_format auto-stripped",
            },
        },
        "fallback_chains": _config.fallbacks,
    }
    return providers


@app.get("/v1/scillm/logs")
async def scillm_auth(request: Request):
    """Check OAuth token health for Claude and Codex providers."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    from scillm.proxy.providers.auth import get_auth_status_snapshot

    return get_auth_status_snapshot()


@app.get("/v1/scillm/auth")
async def scillm_auth_endpoint(request: Request):
    """Check OAuth token health. Alias for the auth check."""
    return await scillm_auth(request)


@app.get("/v1/scillm/logs")
async def scillm_logs(request: Request, date: str = "", limit: int = 100):
    """Query request logs. Returns cost summary + recent records.

    Usage: GET /v1/scillm/logs?date=2026-03-13&limit=50
    Default: today's date, last 100 records.
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    from datetime import datetime, timezone
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        from chutes.middleware.request_log import get_cost_summary
        return await get_cost_summary(date)
    except (ImportError, Exception) as exc:
        return {"error": f"Request logging not available: {exc}"}


@app.get("/v1/budget")
async def budget_snapshot(request: Request):
    """Chutes budget snapshot (delegated to budget_guard if available)."""
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.budget_guard import get_budget_snapshot
        return get_budget_snapshot()
    except (ImportError, AttributeError):
        return {"status": "budget_guard_not_loaded"}


@app.get("/v1/scillm/debug/{call_id}")
async def debug_call_by_id(request: Request, call_id: str):
    """Analyze a specific call by its ID and return diagnosis with best practices.

    Returns structured JSON with:
    - analysis: LLM-generated diagnosis
    - call details (caller, model, status, timestamp)
    - best_practices_included: true

    Usage: GET /v1/scillm/debug/abc123
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.request_log import get_log_by_key, analyze_call

        log = await get_log_by_key(call_id)
        if not log:
            raise ProxyError(404, f"Call {call_id} not found", "not_found")

        return await analyze_call(log)
    except ProxyError:
        raise
    except Exception as exc:
        return {"success": False, "error": f"Debug failed: {exc}"}


@app.get("/v1/scillm/debug")
async def debug_recent_calls(request: Request, caller: str = "", limit: int = 1):
    """Analyze recent calls for a caller.

    Usage: GET /v1/scillm/debug?caller=pdf_oxide.clone&limit=3

    Returns analysis for the most recent `limit` calls from the specified caller.
    If caller is empty, returns instructions for usage.
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    if not caller:
        return {
            "usage": "GET /v1/scillm/debug?caller=YOUR_SKILL_NAME&limit=1",
            "example": "GET /v1/scillm/debug?caller=pdf_oxide.clone&limit=3",
            "or": "GET /v1/scillm/debug/{call_id} for specific call",
        }

    try:
        from chutes.middleware.request_log import get_recent_logs_by_caller, analyze_call

        logs = await get_recent_logs_by_caller(caller, limit=min(limit, 10))
        if not logs:
            return {"caller": caller, "message": "No recent calls found", "analyses": []}

        analyses = []
        for log in logs:
            result = await analyze_call(log)
            analyses.append(result)

        return {"caller": caller, "count": len(analyses), "analyses": analyses}
    except Exception as exc:
        return {"success": False, "error": f"Debug failed: {exc}"}


# ---------------------------------------------------------------------------
# Batch status (for resume/retry workflows)
# ---------------------------------------------------------------------------


@app.get("/v1/scillm/batch/{batch_id}")
async def batch_status(request: Request, batch_id: str):
    """Get status of a batch job: success/failure counts and failed item IDs.

    Usage: GET /v1/scillm/batch/create-qras-cwe-20260413-123456

    Returns:
    {
        "batch_id": "...",
        "total": 969,
        "success": 950,
        "failed": 19,
        "failed_items": ["CWE-79", "CWE-89", ...]
    }

    Callers can use failed_items to retry only failed work items.
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.batch_resume import get_batch_status
        status = await get_batch_status(batch_id)
        return status
    except ImportError:
        return {"error": "batch_resume module not available"}
    except Exception as exc:
        return {"error": f"Failed to get batch status: {exc}"}


@app.websocket("/v1/scillm/ws/batch/{batch_id}")
async def websocket_batch(websocket: WebSocket, batch_id: str):
    """WebSocket endpoint for real-time batch progress notifications.

    Clients subscribe to receive progress events as calls in the batch complete.

    Events sent to client:
    - subscribed: Initial connection with current progress
    - call_complete: Each call completion with status, duration, cost
    - batch_complete: Final summary when all calls finish
    - ping: Keep-alive (every 30s)

    Client can send:
    - {"type": "set_total", "total": N}: Set expected total calls

    Headers for HTTP requests in the batch:
    - X-Scillm-Batch-Id: Batch identifier (required)
    - X-Scillm-Call-Key: Unique key for each call (optional, defaults to model)
    - X-Scillm-Batch-Total: Total expected calls (optional)

    Example usage (Python):
        import asyncio
        import websockets
        import json

        async def monitor_batch(batch_id):
            uri = f"ws://localhost:4001/v1/scillm/ws/batch/{batch_id}"
            async with websockets.connect(uri) as ws:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    print(f"Progress: {data}")
                    if data.get("event") == "batch_complete":
                        break

        asyncio.run(monitor_batch("my-batch-123"))
    """
    # Note: WebSocket auth is optional — batch_id acts as the secret.
    # For production, consider requiring Bearer token in query param.
    try:
        from chutes.middleware.batch_ws import websocket_batch_handler
        await websocket_batch_handler(websocket, batch_id)
    except ImportError:
        await websocket.accept()
        await websocket.send_json({"error": "batch_ws module not available"})
        await websocket.close(code=1011, reason="module_not_available")


@app.get("/v1/scillm/ws/batch/{batch_id}/status")
async def websocket_batch_status(request: Request, batch_id: str):
    """REST endpoint to check batch progress (alternative to WebSocket).

    Returns current progress for a batch without subscribing.

    Usage: GET /v1/scillm/ws/batch/my-batch-123/status
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    try:
        from chutes.middleware.batch_ws import get_batch_tracker
        tracker = get_batch_tracker()
        status = await tracker.get_status(batch_id)
        if status is None:
            return {"batch_id": batch_id, "status": "not_found"}
        return status
    except ImportError:
        return {"error": "batch_ws module not available"}


# ---------------------------------------------------------------------------
# Embeddings (proxied to local embedding service)
# ---------------------------------------------------------------------------


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint.

    Translates to the local embedding service at EMBEDDING_SERVICE_URL
    and returns results in OpenAI format.
    """
    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    body = await request.json()
    raw_input = body.get("input")
    if raw_input is None:
        raise ProxyError(400, "input is required", "invalid_request_error")

    # Normalize input to list of strings (OpenAI spec allows string or list)
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = [str(t) for t in raw_input]
    else:
        raise ProxyError(400, "input must be a string or list of strings", "invalid_request_error")

    if not texts:
        raise ProxyError(400, "input must not be empty", "invalid_request_error")

    if _embedding_client is None:
        raise ProxyError(503, "Proxy not ready — startup incomplete", "service_unavailable")

    try:
        resp = await _embedding_client.post("/embed/batch", json={"texts": texts})
        resp.raise_for_status()
    except httpx.ConnectError:
        raise ProxyError(502, "Embedding service unreachable", "upstream_error")
    except httpx.HTTPStatusError as exc:
        raise ProxyError(
            502,
            f"Embedding service returned {exc.response.status_code}",
            "upstream_error",
        )
    except httpx.TimeoutException:
        raise ProxyError(502, "Embedding service timed out", "upstream_error")

    result = resp.json()
    vectors = result.get("vectors") or result.get("embeddings") or []
    model_name = result.get("model", "unknown")

    data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(vectors)
    ]

    return JSONResponse(content={
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


# ---------------------------------------------------------------------------
# Direct Chutes passthrough (no middleware)
# ---------------------------------------------------------------------------
# These routes bypass the entire middleware chain — ChutesRouter,
# ConcurrencyGuard, TimeoutEstimator, JsonGuard, etc. — and call Chutes
# directly via httpx. Same behavior as ``curl -X POST https://llm.chutes.ai``.
#
# POST /v1/scillm/chutes/completions  — single, supports stream
# POST /v1/scillm/chutes/batch          — multiple, SSE stream via as_completed


@app.post("/v1/scillm/chutes/completions")
async def chutes_direct_completion(request: Request):
    """Direct Chutes completion — no middleware, one httpx call.

    Same body format as ``/v1/chat/completions``. Supports streaming.
    ``max_tokens`` is silently stripped (see chutes_direct.py).
    """
    from scillm.proxy.chutes_direct import direct_completion
    from scillm.proxy.middleware import MiddlewareReject

    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    if not model:
        raise ProxyError(400, "model is required", "invalid_request_error")
    if not messages:
        raise ProxyError(400, "messages is required", "invalid_request_error")

    try:
        result = await direct_completion(
            model=model,
            messages=messages,
            stream=stream,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            stop=body.get("stop"),
            response_format=body.get("response_format"),
            seed=body.get("seed"),
        )
    except MiddlewareReject as exc:
        raise ProxyError(exc.status_code, str(exc), "model_unavailable")
    except ProxyError:
        raise
    except Exception as exc:
        raise ProxyError(502, f"Chutes call failed: {exc}", "upstream_error")

    if stream:
        raw_resp = result["_stream_response"]
        model_served = result["model_served"]

        async def _stream():
            try:
                yield f'data: {{"object":"chat.completion.chunk","model":"{model_served}"}}\n\n'
                done_seen = False
                async for line in raw_resp.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            done_seen = True
                            yield "data: [DONE]\n\n"
                            continue
                        yield line + "\n"
                if not done_seen:
                    yield "data: [DONE]\n\n"
            finally:
                await raw_resp.aclose()
                slot = getattr(raw_resp, "_scillm_chutes_slot", None)
                if slot is not None:
                    await slot.release()

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"x-model-served": model_served},
        )

    return JSONResponse(content=result, headers={"x-model-served": result.get("model_served", model)})


@app.post("/v1/scillm/chutes/batch")
async def chutes_direct_batch(request: Request):
    """Direct Chutes batch — semaphore + retry + as_completed, SSE stream.

    Body::
        {
            "requests": [{"model": "...", "messages": [...]}, ...],
            "concurrency": 4,
            "wall_time_s": 600
        }

    Yields SSE events as items complete (not in input order).
    """
    from scillm.proxy.chutes_direct import (
        batch_completions,
        run_full_prompt_payload_probe,
        run_prompt_preflight,
    )

    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    body = await request.json()
    requests_list = body.get("requests", [])
    if not requests_list:
        raise ProxyError(400, "requests is required", "invalid_request_error")

    concurrency = int(body.get("concurrency", 4))
    wall_time_s = float(body.get("wall_time_s", 600))
    provider_stream_default = bool(body.get("provider_stream", body.get("stream", True)))
    normalized_requests: list[dict[str, Any]] = []
    for item in requests_list:
        req_item = dict(item or {})
        req_item.setdefault("stream", provider_stream_default)
        normalized_requests.append(req_item)
    requests_list = normalized_requests
    progress_heartbeat_s_raw = body.get("progress_heartbeat_s", body.get("heartbeat_s"))
    progress_heartbeat_s = (
        float(progress_heartbeat_s_raw)
        if progress_heartbeat_s_raw is not None
        else None
    )
    prompt_preflight = body.get("prompt_preflight")
    require_prompt_preflight = bool(body.get("require_prompt_preflight", False))
    if require_prompt_preflight and not prompt_preflight:
        raise ProxyError(
            400,
            "prompt_preflight is required when require_prompt_preflight is true",
            "invalid_request_error",
        )

    async def _event_stream():
        yielded = 0
        if prompt_preflight:
            prompt_preflight_obj = prompt_preflight if isinstance(prompt_preflight, dict) else {}
            first_request = requests_list[0] or {}
            probe_model = (
                prompt_preflight_obj.get("probe_model")
                or prompt_preflight_obj.get("full_prompt_payload", {}).get("model")
                or first_request.get("model")
            )
            probe_response_format = (
                prompt_preflight_obj.get("full_prompt_payload", {}).get("response_format")
                or first_request.get("response_format")
            )
            yielded += 1
            yield f"data: {json.dumps({'type': 'prompt_full_payload_probe_started', 'ok': True, 'model': probe_model})}\n\n"
            probe_wall_time_s = min(wall_time_s, float(prompt_preflight_obj.get("full_prompt_probe_wall_time_s", 300)))
            full_payload_probe: dict[str, Any] | None = None
            async for progress_event in _await_with_phase_progress(
                run_full_prompt_payload_probe(
                    prompt_preflight,
                    model=probe_model,
                    response_format=probe_response_format,
                    wall_time_s=probe_wall_time_s,
                ),
                phase="prompt_full_payload_probe",
                model=probe_model,
                heartbeat_s=progress_heartbeat_s,
                wall_time_s=probe_wall_time_s,
            ):
                if progress_event.get("type") == "prompt_phase_progress":
                    yielded += 1
                    yield f"data: {json.dumps(progress_event)}\n\n"
                    continue
                if progress_event.get("ok"):
                    full_payload_probe = progress_event.get("result") or {}
                else:
                    full_payload_probe = {
                        "ok": False,
                        "error": progress_event.get("error") or "prompt_full_payload_probe_failed",
                        "probe_transport": "scillm_chutes_batch",
                        "probe_model": probe_model,
                    }
            if full_payload_probe is None:
                full_payload_probe = {
                    "ok": False,
                    "error": "prompt_full_payload_probe_missing_result",
                    "probe_transport": "scillm_chutes_batch",
                    "probe_model": probe_model,
                }
            if not full_payload_probe.get("ok"):
                yielded += 1
                event = {
                    "type": "prompt_full_payload_probe",
                    "ok": False,
                    "error": full_payload_probe.get("error"),
                    "missing": full_payload_probe.get("missing"),
                    "probe_transport": full_payload_probe.get("probe_transport"),
                    "probe_model": full_payload_probe.get("probe_model") or probe_model,
                }
                yield f"data: {json.dumps(event)}\n\n"
                return
            yielded += 1
            yield f"data: {json.dumps({'type': 'prompt_full_payload_probe', 'ok': True, 'probe_transport': full_payload_probe.get('probe_transport'), 'probe_model': full_payload_probe.get('probe_model'), 'parsed_keys': sorted((full_payload_probe.get('parsed') or {}).keys())})}\n\n"

            reviewer_model = (
                body.get("prompt_reviewer_model")
                or prompt_preflight_obj.get("reviewer_model")
                or first_request.get("model")
            )
            yielded += 1
            yield f"data: {json.dumps({'type': 'prompt_preflight_started', 'ok': True, 'reviewer_model': reviewer_model})}\n\n"
            reviewer_wall_time_s = min(wall_time_s, float(prompt_preflight_obj.get("wall_time_s", 300)))
            preflight_result: dict[str, Any] | None = None
            async for progress_event in _await_with_phase_progress(
                run_prompt_preflight(
                    prompt_preflight,
                    reviewer_model=reviewer_model,
                    wall_time_s=reviewer_wall_time_s,
                ),
                phase="prompt_preflight",
                model=reviewer_model,
                heartbeat_s=progress_heartbeat_s,
                wall_time_s=reviewer_wall_time_s,
            ):
                if progress_event.get("type") == "prompt_phase_progress":
                    yielded += 1
                    yield f"data: {json.dumps(progress_event)}\n\n"
                    continue
                if progress_event.get("ok"):
                    preflight_result = progress_event.get("result") or {}
                else:
                    preflight_result = {
                        "ok": False,
                        "error": progress_event.get("error") or "prompt_reviewer_batch_failed",
                        "reviewer_transport": "scillm_chutes_batch",
                        "reviewer_model": reviewer_model,
                    }
            if preflight_result is None:
                preflight_result = {
                    "ok": False,
                    "error": "prompt_preflight_missing_result",
                    "reviewer_transport": "scillm_chutes_batch",
                    "reviewer_model": reviewer_model,
                }
            if not preflight_result.get("ok"):
                yielded += 1
                event = {
                    "type": "prompt_preflight",
                    "ok": False,
                    "error": preflight_result.get("error"),
                    "reviewer_transport": preflight_result.get("reviewer_transport"),
                    "reviewer_model": preflight_result.get("reviewer_model") or reviewer_model,
                    "prompt_review": (preflight_result.get("artifact") or {}).get("prompt_review"),
                }
                yield f"data: {json.dumps(event)}\n\n"
                return
            yielded += 1
            event = {
                "type": "prompt_preflight",
                "ok": True,
                "reviewer_transport": preflight_result.get("reviewer_transport"),
                "reviewer_model": preflight_result.get("reviewer_model"),
                "prompt_review": (preflight_result.get("artifact") or {}).get("prompt_review"),
            }
            yield f"data: {json.dumps(event)}\n\n"
        async for item in batch_completions(
            requests_list,
            concurrency=concurrency,
            wall_time_s=wall_time_s,
            progress_heartbeat_s=progress_heartbeat_s,
        ):
            yielded += 1
            yield f"data: {json.dumps(item)}\n\n"
        if yielded == 0:
            yield 'data: {"error":"no_items_completed"}\n\n'

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "x-request-id": request.headers.get("x-request-id", str(uuid.uuid4())[:8]),
        },
    )


@app.get("/v1/scillm/chutes/models")
async def chutes_direct_models(request: Request):
    """List available Chutes models and Scillm Chutes config drift."""
    from scillm.proxy.chutes_direct import _get_available_models

    auth_err = _check_auth(request)
    if auth_err:
        raise ProxyError(401, auth_err, "authentication_error")

    models = await _get_available_models()
    inventory = _chutes_config_inventory(models)
    return {
        "object": "list",
        "data": [{"id": m, "available": True} for m in models],
        **inventory,
        "advice": (
            "Configured Chutes fallbacks or deployments reference models missing from "
            "the live provider inventory. Update local/proxy_server_config.yaml and "
            "rerun ops-chutes models --query <family> --json."
            if inventory["status"] == "config_drift"
            else "Configured Chutes models match the live provider inventory."
        ),
    }


# ---------------------------------------------------------------------------
# Prometheus /metrics endpoint
# ---------------------------------------------------------------------------


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not _prom_available:
        raise ProxyError(404, "prometheus_client not installed", "not_found_error")
    from starlette.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Catch-all for unknown routes
# ---------------------------------------------------------------------------


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str):
    """Return 404 for unrecognized routes."""
    raise ProxyError(404, f"Unknown endpoint: /{path}", "not_found_error")
