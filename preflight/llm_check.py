"""Agent4 preflight: one small request to every LLM named in README 6.7, from this machine.

Secrets are read from ~/api.env and are never printed or written. The prompts contain no data.
usage: python llm_check.py [gemini] [hf] [carbon-repeat N]
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "results"
PROMPT = "Reply with exactly one word: READY"


def load_env(path="~/api.env"):
    env = {}
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.replace("export ", "").strip()] = value.strip().strip('"').strip("'")
    return env


ENV = load_env()


def scrub(text):
    text = str(text)
    for value in ENV.values():
        if value:
            text = text.replace(value, "<redacted>")
    return text


def gemini(model, timeout=120, thinking_level=None):
    """Legacy generateContent endpoint. Google's docs call it legacy but fully supported."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT}]}]}
    if thinking_level:
        body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": thinking_level}}
    start = time.time()
    row = dict(provider="gemini", model=model, timeout=timeout, thinking_level=thinking_level,
               started=time.strftime("%H:%M:%S"))
    try:
        r = requests.post(url, headers={"x-goog-api-key": ENV["GEMINI_API_KEY"]}, json=body, timeout=(15, timeout))
        row.update(status=r.status_code, seconds=round(time.time() - start, 1))
        if r.status_code == 200:
            d = r.json()
            parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            usage = d.get("usageMetadata", {})
            row.update(reply="".join(p.get("text", "") for p in parts).strip()[:40],
                       tokens={k: usage.get(k) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")})
        else:
            row.update(error=scrub(r.text)[:300])
    except Exception as exc:
        row.update(status="timeout" if "Timeout" in type(exc).__name__ else "exception",
                   seconds=round(time.time() - start, 1), error=scrub(repr(exc))[:200])
    return row


def hf_router(model, timeout=180):
    start = time.time()
    row = dict(provider="hf-router", model=model, timeout=timeout, started=time.strftime("%H:%M:%S"))
    try:
        r = requests.post("https://router.huggingface.co/v1/chat/completions",
                          headers={"Authorization": f"Bearer {ENV['HF_TOKEN']}"}, timeout=(15, timeout),
                          json={"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 2048})
        row.update(status=r.status_code, seconds=round(time.time() - start, 1))
        if r.status_code == 200:
            d = r.json()
            message = (d.get("choices") or [{}])[0].get("message", {})
            usage = d.get("usage") or {}
            row.update(reply=(message.get("content") or "").strip()[:40], served_model=d.get("model"),
                       returns_reasoning=bool(message.get("reasoning_content") or message.get("reasoning")),
                       tokens={k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}, cost_usd=usage.get("cost"))
        else:
            row.update(error=scrub(r.text)[:300])
    except Exception as exc:
        row.update(status="timeout" if "Timeout" in type(exc).__name__ else "exception",
                   seconds=round(time.time() - start, 1), error=scrub(repr(exc))[:200])
    return row


def main():
    args = sys.argv[1:] or ["gemini", "hf"]
    rows = []

    def record(row):
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if "gemini" in args:
        for model in ["gemini-4-carbon", "gemini-4-neon", "gemini-3.8-flash"]:
            record(gemini(model))
    if "hf" in args:
        for model in ["Qwen/Qwen3.8-2.4T-A95B", "deepseek-ai/DeepSeek-V4-Pro"]:
            record(hf_router(model))
    if "carbon-repeat" in args:
        n = int(args[args.index("carbon-repeat") + 1])
        for i in range(n):  # strictly one at a time, each allowed to run to a long timeout
            record(gemini("gemini-4-carbon", timeout=400, thinking_level="low" if i % 2 else None))
            time.sleep(5)
    OUT.mkdir(exist_ok=True)
    name = "llm_carbon_repeat.json" if "carbon-repeat" in args else "llm_check.json"
    (OUT / name).write_text(json.dumps(dict(host=os.uname().nodename, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), calls=rows),
                                       indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
