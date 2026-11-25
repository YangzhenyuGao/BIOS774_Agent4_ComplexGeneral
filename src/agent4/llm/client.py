"""One client for every LLM call (README 6.7).

Roles
  generator  Gemini 4 Carbon only. One attempt may take at most 6 minutes; on timeout or server error the
             same request is sent again until an answer arrives. No fallback model.
  judge      Claude Opus 5.5 through the Claude Code command line (login on this machine), then, only if
             that keeps failing, the reasoning models on the Hugging Face router.

Every answer is cached under llm_cache/ keyed by a hash of (role, model, system, prompt). With a warm cache
the whole pipeline replays without keys, login or network. Secrets are read from ~/api.env and never
logged. Callers must pass summaries, plan text and evidence only - never patient- or cell-level records.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

from ..io import ROOT, now, read_json, sha256_text, write_json

GENERATOR_MODEL = "gemini-4-carbon"
GENERATOR_TIMEOUT = 360
JUDGE_MODEL = "claude-opus-5-5"
JUDGE_FALLBACKS = ("Qwen/Qwen3.8-2.4T-A95B", "deepseek-ai/DeepSeek-V4-Pro")
CACHE = ROOT / "llm_cache"
LOG = ROOT / "logs" / "llm_calls.jsonl"
_log_lock = threading.Lock()


class CacheMiss(RuntimeError):
    pass


class PermanentError(RuntimeError):
    """A request the service rejects as malformed or unauthorised; retrying the same request cannot help."""


def _env():
    env, path = {}, Path(os.path.expanduser("~/api.env"))
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.replace("export ", "").strip()] = v.strip().strip('"').strip("'")
    return env


def _scrub(text, env):
    text = str(text)
    for v in env.values():
        if v:
            text = text.replace(v, "<redacted>")
    return text


def _log(record):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json(text):
    """Parse the first JSON object or array in a model reply (tolerates code fences and surrounding prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found in reply")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def _gemini_once(system, prompt, env, timeout, want_json):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATOR_MODEL}:generateContent"
    config = {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": "high"}}
    if want_json:
        config["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(url, headers={"x-goog-api-key": env["GEMINI_API_KEY"]}, json=body, timeout=(20, timeout))
    if r.status_code in (400, 401, 403, 404):
        raise PermanentError(f"HTTP {r.status_code}: {_scrub(r.text, env)[:400]}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_scrub(r.text, env)[:300]}")
    d = r.json()
    cand = (d.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    if not text.strip():
        raise RuntimeError(f"empty reply (finishReason={cand.get('finishReason')})")
    u = d.get("usageMetadata", {})
    return text, dict(input_tokens=u.get("promptTokenCount"), output_tokens=u.get("candidatesTokenCount"),
                      thinking_tokens=u.get("thoughtsTokenCount"), finish_reason=cand.get("finishReason"))


def _claude_once(system, prompt, effort="high", timeout=600):
    wrapper = ROOT / "scripts" / "claude_cli.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(system)
        sys_path = f.name
    try:
        p = subprocess.run(["bash", str(wrapper), sys_path, JUDGE_MODEL, effort], input=prompt, text=True,
                           capture_output=True, timeout=timeout)
    finally:
        os.unlink(sys_path)
    if p.returncode != 0:
        raise RuntimeError(f"claude cli exit {p.returncode}: {p.stderr[-300:]}")
    d = json.loads(p.stdout)
    if d.get("is_error") or not d.get("result"):
        raise RuntimeError(f"claude cli error: {str(d.get('result'))[:300]}")
    u = d.get("usage", {})
    return d["result"], dict(input_tokens=(u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                             + (u.get("cache_read_input_tokens") or 0), output_tokens=u.get("output_tokens"),
                             thinking_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens"),
                             list_price_usd=d.get("total_cost_usd"), api_seconds=(d.get("duration_api_ms") or 0) / 1000)


def _hf_once(model, system, prompt, env, timeout=300):
    r = requests.post("https://router.huggingface.co/v1/chat/completions", timeout=(20, timeout),
                      headers={"Authorization": f"Bearer {env['HF_TOKEN']}"},
                      json={"model": model, "max_tokens": 8192,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_scrub(r.text, env)[:300]}")
    d = r.json()
    u = d.get("usage") or {}
    return d["choices"][0]["message"]["content"], dict(input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                                                         list_price_usd=u.get("cost"))


def call(role, system, prompt, tag="", want_json=True, replay_only=None, validate=None):
    """Return dict(text, json, model, usage, seconds, attempts, cached).

    validate: optional function(parsed_json) that raises on an unusable answer. For the judge an invalid
    answer is retried (and is not cached); for the generator the caller decides what to do.
    """
    replay_only = bool(int(os.environ.get("A4_REPLAY_ONLY", "0"))) if replay_only is None else replay_only
    model = GENERATOR_MODEL if role == "generator" else JUDGE_MODEL
    key = sha256_text(json.dumps([role, model, system, prompt], ensure_ascii=False))
    path = CACHE / role / f"{key[:32]}.json"
    if path.exists():
        rec = read_json(path)
        rec["cached"] = True
        _log(dict(time=now(), tag=tag, role=role, model=rec["model"], cached=True, key=key[:32]))
        return rec
    if replay_only:
        raise CacheMiss(f"no cached answer for {role} call '{tag}' ({key[:32]}); replay-only mode")
    env = _env()
    attempts, t_start = [], time.time()
    while True:
        t0 = time.time()
        used = model
        try:
            if role == "generator":
                text, usage = _gemini_once(system, prompt, env, GENERATOR_TIMEOUT, want_json)
            else:
                n_fail = len(attempts)
                if n_fail < 3:
                    text, usage = _claude_once(system, prompt)
                else:                       # README 6.7: fallback judges only after repeated failure
                    used = JUDGE_FALLBACKS[min(n_fail - 3, len(JUDGE_FALLBACKS) - 1)]
                    text, usage = _hf_once(used, system, prompt, env)
            parsed = extract_json(text) if want_json else None
            if validate is not None:
                validate(parsed)
            attempts.append(dict(model=used, ok=True, seconds=round(time.time() - t0, 1)))
            break
        except PermanentError:
            raise
        except Exception as exc:  # timeout, server error, malformed or invalid answer
            kind = "timeout" if "Timeout" in type(exc).__name__ or "timed out" in str(exc) else "error"
            attempts.append(dict(model=used, ok=False, kind=kind, seconds=round(time.time() - t0, 1),
                                 error=_scrub(repr(exc), env)[:240]))
            _log({**attempts[-1], "time": now(), "tag": tag, "role": role, "cached": False})
            if role == "judge" and len(attempts) >= 3 + 2 * len(JUDGE_FALLBACKS):
                raise RuntimeError(f"judge call '{tag}' failed on every model: {attempts[-1]['error']}")
            time.sleep(10 if role == "generator" else 3)
    rec = dict(key=key[:32], tag=tag, role=role, model=used, created=now(), system=system, prompt=prompt, text=text,
               json=parsed, usage=usage, seconds=round(time.time() - t_start, 1), attempts=attempts, cached=False)
    write_json(path, rec)
    _log(dict(time=now(), tag=tag, role=role, model=used, cached=False, ok=True, seconds=rec["seconds"],
              n_attempts=len(attempts), usage=usage, key=key[:32]))
    return rec
