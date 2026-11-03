"""Compare streaming vs non-streaming reasoning_content from Grokified."""
import json
import os
import sys

import requests

BASE = "https://api.grokified.com/v1"
KEY = os.environ["GROKIFIED_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
PROMPT = "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball? Think it through."


def non_streaming():
    body = {"model": "grok-4.7", "messages": [{"role": "user", "content": PROMPT}],
            "reasoning_effort": "xhigh", "max_completion_tokens": 800}
    d = requests.post(f"{BASE}/chat/completions", headers=H, json=body, timeout=180).json()
    msg = d["choices"][0]["message"]
    print("NON-STREAM message keys:", sorted(msg.keys()))
    print("  content  :", repr(msg.get("content"))[:120])
    print("  reasoning:", repr(msg.get("reasoning_content"))[:160])


def streaming():
    body = {"model": "grok-4.7", "messages": [{"role": "user", "content": PROMPT}],
            "reasoning_effort": "xhigh", "max_completion_tokens": 800, "stream": True}
    r = requests.post(f"{BASE}/chat/completions", headers=H, json=body, timeout=180, stream=True)
    keys_seen, samples, n_chunks, n_empty = set(), [], 0, 0
    reasoning_parts, content_parts = [], []
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        n_chunks += 1
        choices = chunk.get("choices") or []
        if not choices:
            n_empty += 1
            if n_empty == 1:
                print("  (chunk without choices, keys:", sorted(chunk.keys()), ")")
            continue
        delta = choices[0].get("delta") or {}
        keys_seen.update(delta.keys())
        if len(samples) < 4:
            samples.append({k: (v[:60] if isinstance(v, str) else v) for k, v in delta.items()})
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])
    print(f"STREAM chunks={n_chunks} empty={n_empty} delta_keys={sorted(keys_seen)}")
    print("  samples:", json.dumps(samples)[:400])
    print("  reasoning chars:", len("".join(reasoning_parts)), repr("".join(reasoning_parts))[:140])
    print("  content chars  :", len("".join(content_parts)), repr("".join(content_parts))[:120])


non_streaming()
print()
streaming()
