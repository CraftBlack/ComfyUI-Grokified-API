"""Probe Grokified API compatibility with the params ComfyUI-OpenAI-API sends."""
import base64
import io
import json
import os
import sys

import requests

BASE = "https://api.grokified.com/v1"
KEY = os.environ["GROKIFIED_API_KEY"]
MODEL = "grok-4.7"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def call(name, payload, timeout=120):
    body = {"model": MODEL, "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}]}
    body.update(payload)
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=H, json=body, timeout=timeout)
        if r.status_code != 200:
            print(f"[FAIL {r.status_code}] {name}: {r.text[:400]}")
            return None
        d = r.json()
        msg = d["choices"][0]["message"]
        rc = msg.get("reasoning_content")
        print(f"[ OK ] {name}: content={msg.get('content')!r} keys={sorted(msg.keys())} "
              f"reasoning_len={len(rc) if rc else 0} usage={d['usage'].get('completion_tokens')}tok")
        return d
    except Exception as e:  # noqa: BLE001
        print(f"[ERR ] {name}: {type(e).__name__}: {e}")
        return None


# 1. exactly what ComfyUI-OpenAI-API sends by default (seed + max_tokens)
call("node-defaults: seed+max_tokens+temperature", {"seed": 0, "max_tokens": 64, "temperature": 1.0})
# 2. isolate each param
call("max_tokens only", {"max_tokens": 64})
call("max_completion_tokens only", {"max_completion_tokens": 64})
call("seed only", {"seed": 1234})
call("temperature only", {"temperature": 0.7})
call("top_p only", {"top_p": 0.9})
call("stop only", {"stop": ["\n\n"]})
# 3. reasoning
call("reasoning_effort=low", {"reasoning_effort": "low", "max_completion_tokens": 200})
call("reasoning_effort=xhigh", {"reasoning_effort": "xhigh", "max_completion_tokens": 200})
call("reasoning_effort=none", {"reasoning_effort": "none", "max_completion_tokens": 50})
call("reasoning_effort=medium (docs: not valid?)", {"reasoning_effort": "medium", "max_completion_tokens": 50})
# 4. reasoning split across the two wire formats
call("reasoning:{effort:low}", {"reasoning": {"effort": "low"}, "max_completion_tokens": 100})
call("thinking:{type:enabled}", {"thinking": {"type": "enabled"}, "max_completion_tokens": 100})

# 5. vision: data-URI png made on the fly
from PIL import Image  # noqa: E402

img = Image.new("RGB", (64, 64), (200, 30, 30))
buf = io.BytesIO()
img.save(buf, format="PNG")
uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
call("vision data-uri png", {"max_completion_tokens": 60, "messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "What color is this image? One word."},
        {"type": "image_url", "image_url": {"url": uri}},
    ]}]})

# 6. system message as array-of-parts (node sends system content as a list)
call("system prompt as parts array", {"max_completion_tokens": 40, "messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
    {"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}]})
call("system prompt as plain string", {"max_completion_tokens": 40, "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Reply with exactly: OK"}]})

print("\n--- streaming, look for reasoning_content deltas ---")
r = requests.post(f"{BASE}/chat/completions", headers=H, json={
    "model": MODEL, "stream": True, "reasoning_effort": "low", "max_completion_tokens": 120,
    "messages": [{"role": "user", "content": "What is 17*23? Think briefly."}]}, timeout=120, stream=True)
seen = set()
first = []
for line in r.iter_lines():
    if not line or not line.startswith(b"data: "):
        continue
    chunk = line[6:]
    if chunk == b"[DONE]":
        break
    d = json.loads(chunk)
    delta = d["choices"][0].get("delta", {})
    seen.update(delta.keys())
    if len(first) < 6:
        first.append({k: (v[:80] if isinstance(v, str) else v) for k, v in delta.items()})
print("delta keys seen:", sorted(seen))
print("first deltas:", json.dumps(first, indent=1)[:800])
