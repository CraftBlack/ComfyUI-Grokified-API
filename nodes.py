"""ComfyUI node: Grokified (OpenAI-compatible) LLM text processor.

Drop-in replacement for the local "LLM Text Processor" node
(ComfyUI-LLM-text-processor) for people who want a cloud model instead of
llama.cpp: same RESPONSE / REASONING / PERF outputs, same optional IMAGE input,
same system-prompt presets folder (ComfyUI/models/LLM/prompts).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .grokified_client import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    REASONING_EFFORTS,
    GrokifiedError,
    chat_completion,
    list_models,
    resolve_api_key,
)

LOGGER = logging.getLogger("GrokifiedAPI")

NO_SYSTEM_PROMPT = "none"
PROMPT_FOLDER_KEY = "llm_text_processor_prompts"
MAX_INLINE_IMAGE_BYTES = 18 * 1024 * 1024  # Grokified docs: 20 MiB per image


# --------------------------------------------------------------------------- #
# system prompt presets (shared with ComfyUI-LLM-text-processor)
# --------------------------------------------------------------------------- #
def _prompt_root() -> Path | None:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "LLM" / "prompts"
    except Exception:  # noqa: BLE001 - folder_paths always exists inside ComfyUI
        return None


def system_prompt_options() -> list[str]:
    """Preset .txt files from the local LLM node's prompts folder."""
    names: list[str] = []
    try:
        import folder_paths

        if PROMPT_FOLDER_KEY in folder_paths.folder_names_and_paths:
            names = [
                name
                for name in folder_paths.get_filename_list(PROMPT_FOLDER_KEY)
                if os.sep not in name and "/" not in name
            ]
    except Exception:  # noqa: BLE001
        names = []

    if not names:
        root = _prompt_root()
        if root is not None and root.is_dir():
            names = sorted(p.name for p in root.glob("*.txt"))

    return [NO_SYSTEM_PROMPT] + names


def _read_preset(name: str) -> tuple[str | None, str | None]:
    """Return (text, warning) for a preset name; never raises."""
    candidates: list[Path] = []
    try:
        import folder_paths

        if PROMPT_FOLDER_KEY in folder_paths.folder_names_and_paths:
            full = folder_paths.get_full_path(PROMPT_FOLDER_KEY, name)
            if full:
                candidates.append(Path(full))
    except Exception:  # noqa: BLE001
        pass

    root = _prompt_root()
    if root is not None:
        candidates.append(root / name)
        candidates.append(root / f"{name}.txt")
    candidates.append(Path(name))

    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace"), None
        except OSError:
            continue

    where = str(root) if root is not None else "ComfyUI/models/LLM/prompts"
    return None, (
        f"system prompt preset '{name}' was not found in {where}; continuing without a "
        "system prompt (paste the text into system_prompt_text instead)"
    )


def build_system_prompt(preset: str, inline_text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    inline = (inline_text or "").strip()
    if inline:
        return inline, warnings
    preset = (preset or "").strip()
    if not preset or preset == NO_SYSTEM_PROMPT:
        return "", warnings
    text, warning = _read_preset(preset)
    if warning:
        warnings.append(warning)
    return (text or "").strip(), warnings


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def _encode_image(pil_image: Image.Image, max_bytes: int) -> tuple[str, bytes]:
    """PNG when it fits, otherwise JPEG (Grokified vision accepts jpg/png)."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    blob = buffer.getvalue()
    if len(blob) <= max_bytes:
        return "image/png", blob

    quality = 92
    while True:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
        blob = buffer.getvalue()
        if len(blob) <= max_bytes or quality <= 60:
            break
        quality -= 10

    scale = 0.75
    while len(blob) > max_bytes and scale > 0.2:
        size = (max(1, int(pil_image.width * scale)), max(1, int(pil_image.height * scale)))
        buffer = io.BytesIO()
        pil_image.resize(size, Image.LANCZOS).save(
            buffer, format="JPEG", quality=85, subsampling=0, optimize=True
        )
        blob = buffer.getvalue()
        scale -= 0.2

    return "image/jpeg", blob


def image_to_parts(
    image,
    *,
    detail: str = "auto",
    max_side: int = 2048,
    max_images: int = 3,
    max_bytes: int = MAX_INLINE_IMAGE_BYTES,
) -> tuple[list[dict], list[str]]:
    """ComfyUI IMAGE tensor [B,H,W,C] float 0..1 -> OpenAI image_url parts."""
    if image is None:
        return [], []

    tensor = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    tensor = np.asarray(tensor)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim == 2:
        tensor = tensor[None, ..., None]

    warnings: list[str] = []
    total = int(tensor.shape[0])
    if max_images and total > max_images:
        warnings.append(f"{total} images were supplied; only the first {max_images} were sent")
        tensor = tensor[:max_images]

    parts: list[dict] = []
    for frame in tensor:
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.dtype == np.uint8:
            array = np.ascontiguousarray(frame)
        else:
            array = np.clip(frame.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(array)
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        if pil_image.mode == "L":
            pil_image = pil_image.convert("RGB")
        if max_side and max(pil_image.size) > max_side:
            pil_image.thumbnail((max_side, max_side), Image.LANCZOS)

        mime, blob = _encode_image(pil_image, max_bytes)
        uri = f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")
        image_url: dict = {"url": uri}
        if detail and detail != "auto":
            image_url["detail"] = detail
        elif detail == "auto":
            image_url["detail"] = "auto"
        parts.append({"type": "image_url", "image_url": image_url})

    return parts, warnings


# --------------------------------------------------------------------------- #
# usage formatting
# --------------------------------------------------------------------------- #
def _usage_numbers(usage: dict) -> dict:
    details = usage.get("completion_tokens_details") or {}
    grokified = usage.get("grokified") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "charged_usd": grokified.get("charged_usd"),
        "list_usd": grokified.get("list_usd"),
        "balance_usd": grokified.get("balance_usd"),
    }


def _format_perf(model: str, numbers: dict, elapsed: float, streamed: bool) -> str:
    parts = [model or "?"]
    prompt_tokens = numbers.get("prompt_tokens")
    completion_tokens = numbers.get("completion_tokens")
    reasoning_tokens = numbers.get("reasoning_tokens")
    if prompt_tokens is not None or completion_tokens is not None:
        token_text = f"{prompt_tokens or 0} in / {completion_tokens or 0} out"
        if reasoning_tokens:
            token_text += f" (reasoning {reasoning_tokens})"
        parts.append(token_text)
    parts.append(f"{elapsed:.2f}s" + (" stream" if streamed else ""))
    cost = numbers.get("charged_usd")
    if cost is None:
        cost = numbers.get("cost_usd")
    if cost is not None:
        parts.append(f"${float(cost):.6f}")
    if numbers.get("balance_usd") is not None:
        parts.append(f"balance ${float(numbers['balance_usd']):.2f}")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #
class GrokifiedLLMTextProcessor:
    """Text/vision LLM call against the Grokified API."""

    CATEGORY = "LLM/Grokified"
    FUNCTION = "generate"
    TITLE = "LLM Text Processor (Grokified API)"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "REASONING", "PERF", "USAGE_JSON")
    OUTPUT_TOOLTIPS = (
        "Final model response (reasoning removed).",
        "Reasoning text returned by the model, when it exposes any.",
        "Model, token counts, latency and cost of the last call.",
        "Raw usage metadata as JSON, for logging or cost tracking.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = list_models(DEFAULT_BASE_URL)
        except Exception:  # noqa: BLE001 - never break node registration
            models = [DEFAULT_MODEL]
        if DEFAULT_MODEL in models:
            models = [DEFAULT_MODEL] + [m for m in models if m != DEFAULT_MODEL]

        return {
            "required": {
                "model": (models, {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Model id from the Grokified catalogue (GET /v1/models).",
                }),
                "system_prompt": (system_prompt_options(), {
                    "default": NO_SYSTEM_PROMPT,
                    "tooltip": "System prompt preset from ComfyUI/models/LLM/prompts (same folder as the local LLM node).",
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "User prompt sent to the model.",
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 131072,
                    "tooltip": "Maximum output tokens (sent as max_completion_tokens).",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. Reasoning models ignore it.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold.",
                }),
                "reasoning_effort": (REASONING_EFFORTS, {
                    "default": "high",
                    "tooltip": "Grokified reasoning effort. grok-4.7 supports low/medium/high/xhigh; 'default' omits the field.",
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2**31 - 1,
                    "control_after_generate": True,
                    "tooltip": "Sent only when > 0. Grokified rejects seed = 0.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 3600,
                    "tooltip": "Maximum time to wait for the API response.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input. A single image or ComfyUI batch is sent to the vision model.",
                }),
                "system_prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Inline system prompt. Overrides the preset above when not empty.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": f"Grokified API key. Empty = use the {API_KEY_ENV} environment variable.",
                    "advanced": True,
                }),
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "multiline": False,
                    "tooltip": "OpenAI-compatible base URL. Any compatible gateway works.",
                    "advanced": True,
                }),
                "image_detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "Vision detail hint sent with every image.",
                    "advanced": True,
                }),
                "max_image_side": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Downscale images whose longest side exceeds this. 0 disables resizing.",
                    "advanced": True,
                }),
                "max_images": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Maximum number of batch frames to send.",
                    "advanced": True,
                }),
                "stream": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stream tokens: shows progress and avoids long-request timeouts.",
                    "advanced": True,
                }),
                "extra_body_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": 'Optional JSON merged into the request body, e.g. {"stop": ["\\n\\n"]}.',
                    "advanced": True,
                }),
                "enable_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When disabled, the input prompt is forwarded to RESPONSE without calling the API.",
                    "advanced": True,
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Presets/models may disappear between runs; the node reports that itself.
        return True

    def generate(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
        seed: int,
        timeout_seconds: int,
        image=None,
        system_prompt_text: str = "",
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        image_detail: str = "auto",
        max_image_side: int = 2048,
        max_images: int = 3,
        stream: bool = True,
        extra_body_json: str = "",
        enable_processing: bool = True,
    ):
        if not enable_processing:
            return (prompt, "", "skipped (enable_processing = false)", "{}")

        key = resolve_api_key(api_key)
        if not key:
            raise GrokifiedError(
                "No Grokified API key. Paste it in the node's api_key field or set the "
                f"{API_KEY_ENV} environment variable, then retry."
            )

        system_text, warnings = build_system_prompt(system_prompt, system_prompt_text)
        for warning in warnings:
            LOGGER.warning("[Grokified] %s", warning)

        parts, image_warnings = image_to_parts(
            image, detail=image_detail, max_side=max_image_side, max_images=max_images
        )
        for warning in image_warnings:
            LOGGER.warning("[Grokified] %s", warning)

        messages: list[dict] = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        user_content: list[dict] = [{"type": "text", "text": prompt}]
        user_content.extend(parts)
        messages.append({"role": "user", "content": user_content})

        extra_body = None
        if (extra_body_json or "").strip():
            try:
                extra_body = json.loads(extra_body_json)
            except json.JSONDecodeError as exc:
                raise GrokifiedError(f"extra_body_json is not valid JSON: {exc}") from exc
            if not isinstance(extra_body, dict):
                raise GrokifiedError("extra_body_json must be a JSON object")

        progress = None
        progress_step = 0
        if stream:
            try:
                import comfy.utils

                progress = comfy.utils.ProgressBar(int(max_tokens))
            except Exception:  # noqa: BLE001 - progress is cosmetic
                progress = None

        def on_delta(_kind: str) -> None:
            nonlocal progress_step
            if progress is None:
                return
            progress_step += 1
            if progress_step % 4 == 0:
                progress.update(4)  # update() is relative; ProgressBar clamps to total

        LOGGER.info(
            "[Grokified] %s -> %s | %d image(s) | max_tokens=%s effort=%s stream=%s",
            model,
            base_url,
            len(parts),
            max_tokens,
            reasoning_effort,
            bool(stream),
        )

        started = time.perf_counter()
        try:
            result = chat_completion(
                messages=messages,
                model=model,
                base_url=base_url,
                api_key=key,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                reasoning_effort=reasoning_effort,
                timeout=float(timeout_seconds),
                stream=bool(stream),
                extra_body=extra_body,
                on_delta=on_delta if progress is not None else None,
            )
        except GrokifiedError as exc:
            LOGGER.error("[Grokified] %s", exc)
            raise
        elapsed = time.perf_counter() - started

        for note in result.get("notes") or []:
            LOGGER.warning("[Grokified] %s", note)

        response_text = result.get("content") or ""
        reasoning_text = result.get("reasoning") or ""
        if not response_text and result.get("refusal"):
            response_text = str(result["refusal"])
            LOGGER.warning("[Grokified] model refused the request; returned the refusal text")

        if result.get("finish_reason") == "length":
            LOGGER.warning(
                "[Grokified] output hit max_tokens=%s; raise it if the response looks cut off",
                max_tokens,
            )

        numbers = _usage_numbers(result.get("usage") or {})
        usage_json = json.dumps(
            {
                "model": result.get("model") or model,
                "finish_reason": result.get("finish_reason"),
                "streamed": result.get("streamed"),
                "elapsed_seconds": round(elapsed, 3),
                "images_sent": len(parts),
                "notes": result.get("notes") or [],
                **numbers,
            },
            ensure_ascii=False,
        )
        perf = _format_perf(result.get("model") or model, numbers, elapsed, bool(result.get("streamed")))

        if response_text:
            LOGGER.info(
                "[Grokified] done in %.2fs | %s chars out | %s chars reasoning",
                elapsed,
                len(response_text),
                len(reasoning_text),
            )
        else:
            LOGGER.warning("[Grokified] the model returned an empty response (%s)", perf)

        return (response_text, reasoning_text, perf, usage_json)


NODE_CLASS_MAPPINGS = {"GrokifiedLLMTextProcessor": GrokifiedLLMTextProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"GrokifiedLLMTextProcessor": "LLM Text Processor (Grokified API)"}
