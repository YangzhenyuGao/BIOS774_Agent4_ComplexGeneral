"""Small file helpers shared by the agent and the evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGENT4_ROOT", Path(__file__).resolve().parents[2]))


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return super().default(o)


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, cls=_Encoder, ensure_ascii=False))
    tmp.replace(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(path) -> str:
    """Path relative to the project root, for records that must stay valid if the project moves."""
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def write_index(directory, note: str = "") -> Path:
    """Write index.json listing every file under `directory` with size and SHA256 (README 5.5)."""
    directory = Path(directory)
    files = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != "index.json" and not p.name.endswith(".tmp"):
            files.append(dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size, sha256=sha256_file(p)))
    return write_json(directory / "index.json", dict(created=now(), note=note, n_files=len(files), files=files))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 2)
