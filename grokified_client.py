"""OpenAI-compatible HTTP client for the Grokified API (https://grokified.com).

Deliberately dependency-light: it only uses ``requests``, which ComfyUI already
ships, so the node works without installing the ``openai`` package.

Endpoints used:
    GET  {base_url}/models               -> model catalogue (for the combo widget)
    POST {base_url}/chat/completions     -> chat completion (streaming or not)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import requests

DEFAULT_BASE_URL = "https://api.grokified.com/v1"
DEFAULT_MODEL = "grok-4.7"
API_KEY_ENV = "GROKIFIED_API_KEY"

# Used only when /models cannot be reached (offline, wrong key, ComfyUI starting).
FALLBACK_MODELS = [
    "grok-4.7",
    "grok-4.6",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
    "grok-build-0.1",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-opus-4-8",
]

# Values Grokified rejects for grok-4.7 (there is no "none"/"off" effort).
REASONING_EFFORTS = ["default", "low", "medium", "high", "xhigh"]
_NO_EFFORT = {"", "default", "none", "off", "auto", "disabled"}


class GrokifiedError(RuntimeError):
    """Non-recoverable Grokified API problem (bad request, server error...)."""


class GrokifiedAuthError(GrokifiedError):
    """Missing or rejected API key."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_api_key(value: str | None = None) -> str:
    """Widget value first, then the GROKIFIED_API_KEY environment variable."""
    key = (value or "").strip()
    if key:
        return key
    return (os.environ.get(API_KEY_ENV) or "").strip()


def normalize_base_url(value: str | None = None) -> str:
    url = (value or "").strip() or DEFAULT_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    url = url.rstrip("/")
    parts = urlsplit(url)
    if parts.path in ("", "/"):
        url = urlunsplit((parts.scheme, parts.netloc, "/v1", "", ""))
    return url


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _as_text(value: Any) -> str:
    """Content may be a string or a list of {type, text} parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for part in value:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(_as_text(part.get("text") or part.get("content")))
        return "".join(out)
    return str(value)


def _error_text(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return (response.text or "").strip()[:800]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err)
        return str(err or data.get("message") or json.dumps(data))[:800]
    return str(data)[:800]


def mask_key(key: str) -> str:
    if not key:
        return "(none)"
    if len(key) <= 12:
        return key[:4] + "..."
    return f"{key[:8]}...{key[-4:]}"


# --------------------------------------------------------------------------- #
# model catalogue
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, Any] = {"key": None, "ts": 0.0, "models": [], "error": None}
MODEL_CACHE_TTL = 300.0
MODEL_CACHE_FAIL_TTL = 45.0


def list_models(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    timeout: float = 8.0,
    force: bool = False,
) -> list[str]:
    """Model ids from GET /models, cached; falls back to a static list."""
    base = normalize_base_url(base_url)
    key = resolve_api_key(api_key)
    cache_key = (base, key)
    now = time.time()

    cache = _MODEL_CACHE
    if not force and cache["key"] == cache_key and cache["models"]:
        ttl = MODEL_CACHE_FAIL_TTL if cache["error"] else MODEL_CACHE_TTL
        if now - cache["ts"] < ttl:
            return list(cache["models"])

    models: list[str] = []
    error: str | None = None
    if not key:
        error = f"no API key (widget empty and {API_KEY_ENV} unset)"
    else:
        try:
            response = requests.get(f"{base}/models", headers=_headers(key), timeout=timeout)
            if response.status_code in (401, 403):
                error = f"HTTP {response.status_code}: {_error_text(response)}"
            else:
                response.raise_for_status()
                for item in response.json().get("data") or []:
                    model_id = item.get("id") if isinstance(item, dict) else item
                    if isinstance(model_id, str) and model_id and model_id not in models:
                        models.append(model_id)
                if not models:
                    error = "empty model list returned by the API"
        except Exception as exc:  # noqa: BLE001 - never break node registration
            error = f"{type(exc).__name__}: {exc}"

    if not models:
        models = list(FALLBACK_MODELS)

    _MODEL_CACHE.update(key=cache_key, ts=now, models=models, error=error)
    return list(models)


# --------------------------------------------------------------------------- #
# chat completions
# --------------------------------------------------------------------------- #
def _downgrade(body: dict[str, Any], message: str) -> str | None:
    """Remove/convert the single parameter the server complained about.

    Grokified's catalogue mixes model families (grok-*, claude-*) that do not
    all accept the same sampling/reasoning parameters, so a rejected parameter
    is dropped and the request retried instead of failing the whole node.
    """
    text = (message or "").lower()

    if "max_completion_tokens" in text and "max_completion_tokens" in body:
        body["max_tokens"] = body.pop("max_completion_tokens")
        return "server rejected max_completion_tokens -> resent as max_tokens"
    if "reasoning_effort" in text and "reasoning_effort" in body:
        body.pop("reasoning_effort")
        return "server rejected reasoning_effort -> omitted"
    if "seed" in text and "seed" in body:
        body.pop("seed")
        return "server rejected seed -> omitted"
    if "temperature" in text and "temperature" in body:
        body.pop("temperature")
        return "server rejected temperature -> omitted"
    if "top_p" in text and "top_p" in body:
        body.pop("top_p")
        return "server rejected top_p -> omitted"
    if "stop" in text and "stop" in body:
        body.pop("stop")
        return "server rejected stop -> omitted"
    if "detail" in text:
        removed = False
        for msg in body.get("messages") or []:
            for part in msg.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("image_url"), dict):
                    if part["image_url"].pop("detail", None) is not None:
                        removed = True
        if removed:
            return "server rejected image detail -> omitted"
    if "max_tokens" in text and "max_tokens" in body:
        body.pop("max_tokens")
        return "server rejected max_tokens -> omitted"
    if "stream" in text and body.get("stream"):
        body["stream"] = False
        return "server rejected stream -> retried without streaming"
    return None


def _parse_completion(data: dict[str, Any]) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    return {
        "content": _as_text(message.get("content")),
        "reasoning": _as_text(message.get("reasoning_content") or message.get("reasoning")),
        "refusal": message.get("refusal"),
        "usage": data.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
        "model": data.get("model"),
        "streamed": False,
    }


def _parse_stream(response: requests.Response, on_delta: Callable[[str], None] | None) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    refusal = None
    usage: dict[str, Any] = {}
    finish_reason = None
    model = None

    for raw in response.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("[DONE]", ""):
            if payload == "[DONE]":
                break
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(chunk.get("model"), str):
            model = chunk["model"]
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or choice.get("message") or {}

        piece = _as_text(delta.get("content"))
        if piece:
            content.append(piece)
            if on_delta:
                on_delta("content")
        piece = _as_text(delta.get("reasoning_content") or delta.get("reasoning"))
        if piece:
            reasoning.append(piece)
            if on_delta:
                on_delta("reasoning")
        if delta.get("refusal"):
            refusal = (refusal or "") + _as_text(delta["refusal"])
            if on_delta:
                on_delta("refusal")

    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "refusal": refusal,
        "usage": usage,
        "finish_reason": finish_reason,
        "model": model,
        "streamed": True,
    }


def chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 300.0,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
    max_attempts: int = 5,
    on_delta: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """POST /chat/completions with retries and parameter auto-downgrade."""
    base = normalize_base_url(base_url)
    key = resolve_api_key(api_key)
    if not key:
        raise GrokifiedAuthError(
            "No Grokified API key. Put it in the node's api_key field or set the "
            f"{API_KEY_ENV} environment variable."
        )

    body: dict[str, Any] = {"model": model, "messages": messages, "stream": bool(stream)}
    if max_tokens:
        body["max_completion_tokens"] = int(max_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if seed:
        body["seed"] = int(seed)
    effort = str(reasoning_effort or "").strip().lower()
    if effort and effort not in _NO_EFFORT:
        body["reasoning_effort"] = effort
    if extra_body:
        body.update(extra_body)

    notes: list[str] = []
    started = time.perf_counter()
    last_transport_error: Exception | None = None

    for attempt in range(max_attempts):
        streaming = bool(body.get("stream"))
        request_started = time.perf_counter()
        try:
            response = requests.post(
                f"{base}/chat/completions",
                headers=_headers(key),
                json=body,
                timeout=(20.0, float(timeout)),
                stream=streaming,
            )
        except requests.RequestException as exc:
            last_transport_error = exc
            if attempt < max_attempts - 1:
                time.sleep(min(2.0**attempt, 8.0))
                continue
            raise GrokifiedError(f"Could not reach {base}: {exc}") from exc

        try:
            if response.status_code == 200:
                if streaming:
                    result = _parse_stream(response, on_delta)
                else:
                    result = _parse_completion(response.json())
                result["elapsed"] = time.perf_counter() - started
                result["request_seconds"] = time.perf_counter() - request_started
                result["notes"] = notes
                result["request"] = {
                    "model": model,
                    "base_url": base,
                    "stream": streaming,
                    "params": sorted(k for k in body if k not in ("messages", "model")),
                }
                return result

            detail = _error_text(response)

            if response.status_code in (401, 403):
                raise GrokifiedAuthError(
                    f"Grokified rejected the API key ({mask_key(key)}): HTTP "
                    f"{response.status_code} {detail}"
                )

            if response.status_code == 400:
                note = _downgrade(body, detail)
                if note:
                    notes.append(note)
                    continue
                raise GrokifiedError(f"HTTP 400 from Grokified for model '{model}': {detail}")

            if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                time.sleep(min(2.0**attempt, 8.0))
                last_transport_error = GrokifiedError(f"HTTP {response.status_code}: {detail}")
                continue

            raise GrokifiedError(f"HTTP {response.status_code} from Grokified: {detail}")
        finally:
            response.close()

    raise GrokifiedError(
        f"Grokified request failed after {max_attempts} attempts"
        + (f": {last_transport_error}" if last_transport_error else "")
    )
