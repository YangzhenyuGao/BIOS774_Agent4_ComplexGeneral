#!/usr/bin/env python
"""Build the submission archives (README 9.3, 11).

    submission/source_code.zip        code, configuration, prompts, cached LLM answers, tests, documentation, reports
    submission/generated_outputs.zip  what the run produced: objectives, plans, evidence cards, tournament records,
                                      audit records, benchmark summaries, report inputs and figures
    submission/manifest.json          every archived file with size and SHA256

Never archived: raw data, encoder weights, encoder features (the task views), the evaluator's hidden data, and
large prediction arrays. Secrets are searched for and the build stops if one is found.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from agent4.io import now, sha256_file  # noqa: E402

SOURCE = [".gitignore", "README.md", "Makefile", "pyproject.toml", "config", "knowhow", "templates", "src", "bench", "scripts", "tests", "docs",
          "llm_cache", "preflight", "reports"]
OUTPUT_DIRS = ["shared_inputs", "plans", "pilot", "tournament", "full", "audit", "benchmark1", "report", "alt_question"]
SKIP_SUFFIX = (".npy", ".npz", ".pyc", ".parquet", ".h5", ".tmp")
KEEP_PARQUET = False
MAX_BYTES = 20 * 1024 * 1024
SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|hf_[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9\-]{10,}")


def wanted(path: Path) -> bool:
    if any(part in ("__pycache__", ".cache", ".pytest_cache") for part in path.parts):
        return False
    if path.suffix in SKIP_SUFFIX or path.stat().st_size > MAX_BYTES:
        return False
    return True


def collect(paths):
    files = []
    for rel in paths:
        p = ROOT / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file() and wanted(f)]
    return files


def scan_for_secrets(files):
    hits = []
    for f in files:
        if f.suffix in (".pdf", ".png", ".zip"):
            continue
        try:
            if SECRET.search(f.read_text(errors="ignore")):
                hits.append(str(f.relative_to(ROOT)))
        except OSError:
            pass
    return hits


def build(name, files, manifest):
    out = ROOT / "submission" / name
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = str(f.relative_to(ROOT))
            z.write(f, rel)
            manifest.append(dict(archive=name, path=rel, bytes=f.stat().st_size, sha256=sha256_file(f)))
    print(f"{name}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")


def main():
    source = collect(SOURCE)
    outputs = []
    for task in sorted(p.name for p in (ROOT / "outputs").iterdir() if p.is_dir() and p.name.startswith("task_")):
        outputs += collect([f"outputs/{task}/{d}" for d in OUTPUT_DIRS] + [f"outputs/{task}/costs.json", f"outputs/{task}/index.json"])
    outputs += collect(["outputs/prompt_freeze.json", "outputs/replay_check.json", "logs/llm_calls.jsonl"])
    hits = scan_for_secrets(source + outputs)
    if hits:
        raise SystemExit(f"possible secrets found, nothing archived: {hits}")
    forbidden = [str(f.relative_to(ROOT)) for f in source + outputs if "agent_view" in f.parts or "bench_private" in f.parts]
    assert not forbidden, forbidden
    manifest = []
    build("source_code.zip", source, manifest)
    build("generated_outputs.zip", outputs, manifest)
    (ROOT / "submission" / "manifest.json").write_text(json.dumps(dict(
        built=now(), excluded="raw data, encoder weights, encoder features (task views), evaluator-only data, arrays and tables above 20 MB",
        n_files=len(manifest), files=manifest), indent=1))


if __name__ == "__main__":
    main()
