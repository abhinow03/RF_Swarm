"""
Progress reporting for long-running eval/training scripts (battery runs,
masked-loss measurement, QLoRA training loops) -- so a 60-minute run isn't a
blind wait. Writes evaluation/progress/<task>.json periodically and prints a
one-line stdout summary; a second terminal can `watch`/`tail -f` that file.

Usage:
    from swarm_intent.progress import Reporter

    with Reporter("degradation_v3b", total=540, rate_hint=0.26) as r:
        for case in cases:
            ... do one unit of work ...
            r.update(1, item=case["name"])
    # __exit__ writes status="done" on success, status="failed" + the last
    # traceback line on an uncaught exception (then re-raises).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROGRESS_DIR = REPO_ROOT / "evaluation" / "progress"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Reporter:
    def __init__(self, task: str, total: int, rate_hint: Optional[float] = None,
                 progress_dir: Optional[Path] = None):
        self.task = task
        self.total = total
        self.done = 0
        self.started_at = time.time()
        self.last_write = 0.0
        self.last_item = None
        self.status = "running"
        self.error_line = None
        self._first_unit_times: list[float] = []

        self.dir = Path(progress_dir) if progress_dir else DEFAULT_PROGRESS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{task}.json"

        if rate_hint and rate_hint > 0:
            eta_s = total / rate_hint
            print(f"[{task}] {total} units @ {rate_hint:.3f}/s (--rate-hint) -> "
                  f"estimated total runtime {_fmt_duration(eta_s)}", flush=True)
        else:
            print(f"[{task}] starting, {total} units -- runtime estimate after "
                  f"the first 10 complete", flush=True)
        print(f"[{task}] follow progress in another terminal with:\n"
              f"    watch -n 2 cat {self.path}", flush=True)
        self._write()

    def update(self, n: int = 1, item: Optional[str] = None):
        self.done += n
        if item is not None:
            self.last_item = item
        now = time.time()

        if len(self._first_unit_times) < 10:
            self._first_unit_times.append(now)
            if len(self._first_unit_times) == 10:
                elapsed10 = self._first_unit_times[-1] - self.started_at
                rate = 10 / elapsed10 if elapsed10 > 0 else 0.0
                if rate > 0:
                    eta_s = (self.total - self.done) / rate
                    print(f"[{self.task}] measured rate from first 10 units: "
                          f"{rate:.3f}/s -> estimated total runtime "
                          f"{_fmt_duration(self.total / rate)}, ETA {_fmt_duration(eta_s)} "
                          f"from now", flush=True)

        due = (self.done % 10 == 0) or (now - self.last_write >= 30) or (self.done >= self.total)
        if due:
            self._write()

    def _write(self):
        now = time.time()
        elapsed = now - self.started_at
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - self.done, 0)
        eta_s = remaining / rate if rate > 0 else None
        eta_clock = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + eta_s))
                    if eta_s is not None else None)
        pct = 100.0 * self.done / self.total if self.total else 100.0

        payload = {
            "task": self.task,
            "done": self.done,
            "total": self.total,
            "pct": round(pct, 1),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "elapsed_s": round(elapsed, 1),
            "rate_per_s": round(rate, 4),
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "eta_clock_time": eta_clock,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "status": self.status,
            "current_item": self.last_item,
        }
        if self.error_line:
            payload["error"] = self.error_line
        self.path.write_text(json.dumps(payload, indent=2))
        self.last_write = now

        line = (f"[{self.task}] {self.done}/{self.total} ({pct:.1f}%) "
               f"rate={rate:.3f}/s elapsed={_fmt_duration(elapsed)} "
               f"eta={_fmt_duration(eta_s)}")
        if self.last_item:
            line += f" | {self.last_item}"
        print(line, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.status = "failed"
            self.error_line = traceback.format_exception_only(exc_type, exc)[-1].strip()
            self._write()
            return False  # propagate the exception
        self.status = "done"
        self._write()
        return False
