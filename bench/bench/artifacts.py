"""JSONL artifact writer + resume-aware reader.

One record per benchmark cell. Appended atomically so a crash mid-night
leaves a valid file (every line that exists is a complete JSON object).

Record schema is documented in the plan at
~/.claude/plans/lexical-toasting-kahan.md.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def _atomic_append(path: Path, line: str) -> None:
    """Append a single line and fsync so interrupted writes don't corrupt.

    Single-line append + fsync is sufficient for correctness: readers
    parse line-by-line and tolerate a trailing partial line, but fsync
    guarantees that any line we wrote before a crash is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_record(path: Path, record: dict) -> None:
    """Append a single JSON record to the JSONL file."""
    _atomic_append(path, json.dumps(record, default=_json_default))


def _json_default(o):
    # numpy / torch scalars that slipped through
    try:
        import numpy as np  # local import
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    try:
        import torch  # local import
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def iter_records(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Ignore a trailing partial line from a crash; the
                # harness will re-run that cell.
                continue


def read_completed_record_ids(path: Path) -> Dict[str, dict]:
    """Map record_id → record for cells already recorded as completed.

    Used by --resume to skip cells that already have an ok, skipped,
    error, or timeout record. Retrying timeout/error cells with the
    same timeout is pointless — they'll just fail again.
    """
    completed: Dict[str, dict] = {}
    for rec in iter_records(path):
        if rec.get("status") in ("ok", "skipped", "error", "timeout"):
            rid = rec.get("record_id")
            if rid:
                completed[rid] = rec
    return completed
