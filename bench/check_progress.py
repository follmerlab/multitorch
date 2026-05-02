#!/usr/bin/env python3
"""Quick progress checker for bench full run. Usage: python check_progress.py"""
import json
from collections import Counter
from pathlib import Path

JSONL = Path(__file__).parent / "results" / "full.jsonl"
TOTAL = 1032

with open(JSONL) as f:
    records = [json.loads(l) for l in f]

by_status = Counter(r["status"] for r in records)
done = len(records)
print(f"\n{'='*60}")
print(f"Progress: {done}/{TOTAL}  ({100*done/TOTAL:.0f}%)")
print(f"  ok={by_status.get('ok',0)}  skipped={by_status.get('skipped',0)}  "
      f"error={by_status.get('error',0)}  timeout={by_status.get('timeout',0)}")

c = records[-1]["config"]
print(f"\nLatest: {c['impl']}  {c['fixture']}  batch={c['batch_size']}  "
      f"device={c['device']}  -> {records[-1]['status']}")

print(f"\nFixtures:")
fx = Counter(r["config"]["fixture"] for r in records)
for f, n in sorted(fx.items()):
    print(f"  {f}: {n}")

fortran_ok = sum(1 for r in records
                 if r["config"]["impl"] in ("pyctm","pyttmult","ttmult_raw")
                 and r["status"] == "ok")
print(f"\nFortran OK cells: {fortran_ok}")
print(f"{'='*60}\n")
