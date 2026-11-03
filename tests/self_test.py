"""Self-test for ComfyUI-Grokified-API.

Runs the real node class outside ComfyUI (folder_paths / comfy.utils are
stubbed) against the live Grokified API.

    set GROKIFIED_API_KEY=gk_live_...
    C:\\ComfyUI\\venv\\Scripts\\python.exe tests\\self_test.py [path\\to\\ComfyUI-Grokified-API]
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"C:\ComfyUI"))

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def stub_comfy() -> None:
    """Minimal folder_paths / comfy.utils so nodes.py imports standalone."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_filename_list = lambda key: []
    folder_paths.get_full_path = lambda key, name: None
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.current = 0
            self.updates = 0

        def update(self, value):
            self.current = min(self.total, self.current + value)
            self.updates += 1

    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils


def load_package(package_dir: Path):
    sys.path.insert(0, str(package_dir.parent))
    return importlib.import_module(package_dir.name)


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    package_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here
    print(f"package: {package_dir}")
    stub_comfy()
    module = load_package(package_dir)

    node_cls = module.NODE_CLASS_MAPPINGS["GrokifiedLLMTextProcessor"]
    node = node_cls()

    # ---------------------------------------------------------------- widgets
    spec = node_cls.INPUT_TYPES()
    widgets = list(spec["required"]) + list(spec["optional"])
    print("widget/socket order:", widgets)
    models = spec["required"]["model"][0]
    fallback = set(module.grokified_client.FALLBACK_MODELS)
    live_only = [m for m in models if m not in fallback]
    check("model list from API (grok-4.7 first)", models[0] == "grok-4.7" and len(models) > 5, f"{len(models)} models")
    check("model list is live (not the offline fallback)", bool(live_only), f"e.g. {live_only[:3]}")
    check("outputs are RESPONSE/REASONING/PERF/USAGE_JSON",
          node_cls.RETURN_NAMES == ("RESPONSE", "REASONING", "PERF", "USAGE_JSON"))

    def call(**overrides):
        kwargs = dict(
            model="grok-4.7",
            system_prompt="none",
            prompt="Reply with exactly: OK",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            reasoning_effort="high",
            seed=1,
            timeout_seconds=300,
            image=None,
            system_prompt_text="",
            api_key="",
            base_url="https://api.grokified.com/v1",
            image_detail="auto",
            max_image_side=2048,
            max_images=3,
            stream=True,
            extra_body_json="",
            enable_processing=True,
        )
        kwargs.update(overrides)
        return node.generate(**kwargs)

    # ------------------------------------------------------- 1. text streaming
    started = time.perf_counter()
    response, reasoning, perf, usage_json = call(
        system_prompt_text="You are terse.", prompt="Name the capital of France.", max_tokens=512
    )
    print(f"  RESPONSE={response[:120]!r}\n  REASONING={reasoning[:80]!r}...\n  PERF={perf}")
    check("text call (stream=True) returns text", bool(response.strip()), f"{time.perf_counter()-started:.1f}s")
    check("USAGE_JSON parses", isinstance(json.loads(usage_json), dict))
    check("usage has cost + balance", "cost_usd" in usage_json and "balance_usd" in usage_json)

    # ------------------------------------------------- 2. seed + params sanity
    response, _, perf, _ = call(prompt="Say A", max_tokens=256, seed=1, stream=False)
    check("non-streaming call works with seed=1", bool(response.strip()), perf)

    # ------------------------------------------------------------- 3. vision
    import torch

    image = torch.zeros((1, 96, 96, 3), dtype=torch.float32)
    image[..., 0] = 0.85  # red square
    response, reasoning, perf, usage_json = call(
        prompt="What colour is this image? Answer with one word.",
        image=image,
        max_tokens=512,
        stream=True,
    )
    usage = json.loads(usage_json)
    print(f"  vision RESPONSE={response[:80]!r}  images_sent={usage.get('images_sent')}")
    check("vision call sees the image", "red" in response.lower(), response[:60])
    check("image tokens billed", (usage.get("prompt_tokens") or 0) > 0)

    # -------------------------------------------- 4. batch + downscale + detail
    batch = torch.rand((5, 512, 300, 3))
    response, _, perf, usage_json = call(
        prompt="Reply with exactly: OK", image=batch, max_images=2, max_image_side=256,
        image_detail="low", max_tokens=256,
    )
    usage = json.loads(usage_json)
    check("batch limited to max_images", usage.get("images_sent") == 2, str(usage.get("images_sent")))

    # ------------------------------------------------- 5. preset + warnings
    response, _, _, _ = call(
        system_prompt="H3_NSFW_Physics_Engine.txt",  # known-missing preset
        prompt="Reply with exactly: OK",
        max_tokens=256,
    )
    check("missing preset warns but still answers", bool(response.strip()))

    # -------------------------------------------------- 6. passthrough mode
    response, reasoning, perf, usage_json = call(enable_processing=False, prompt="passthrough text")
    check("enable_processing=False forwards the prompt",
          response == "passthrough text" and reasoning == "" and json.loads(usage_json) == {})

    # --------------------------------------------------------- 7. bad inputs
    try:
        call(extra_body_json="{not json")
        check("invalid extra_body_json raises", False)
    except Exception as exc:  # noqa: BLE001
        check("invalid extra_body_json raises", "not valid JSON" in str(exc), type(exc).__name__)

    try:
        call(api_key="gk_live_invalid_key_for_test", prompt="hi", max_tokens=64)
        check("bad API key raises GrokifiedAuthError", False, "no exception raised")
    except Exception as exc:  # noqa: BLE001
        check("bad API key raises GrokifiedAuthError", "Auth" in type(exc).__name__, type(exc).__name__)

    # env-var fallback must not be shadowed by an empty widget value
    saved = os.environ.get("GROKIFIED_API_KEY")
    os.environ["GROKIFIED_API_KEY"] = "gk_live_env_fallback_probe"
    resolved = module.grokified_client.resolve_api_key("")
    check("empty api_key widget falls back to env var", resolved == "gk_live_env_fallback_probe")
    if saved is None:
        os.environ.pop("GROKIFIED_API_KEY", None)
    else:
        os.environ["GROKIFIED_API_KEY"] = saved

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
