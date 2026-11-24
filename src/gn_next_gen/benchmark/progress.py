"""Resumable, observable long runs.

The approved plan forbids this session from launching anything over ~15 minutes,
because a disconnect kills it: the PI launches the grid detached instead.  That
only works if a run can be interrupted and resumed without losing hours, and if
its state is readable without tailing a log.  So every long script here writes a
``progress.json`` beside its log and skips cells that are already on disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Progress:
    """Cell counter that persists itself after every update."""

    path: Path
    total: int
    label: str = ""
    done: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    pid: int = field(default_factory=os.getpid)
    #: Identifies which invocation of the grid this file belongs to.  The
    #: orchestrator exports ``GN_RUN_ID``; a script run by hand gets its own.
    run_id: str = field(default_factory=lambda: os.environ.get("GN_RUN_ID", "") or str(os.getpid()))

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write()

    def _elapsed(self) -> float:
        return time.time() - self.started

    def write(self) -> None:
        finished = self.done + self.skipped + self.failed
        rate = finished / self._elapsed() if self._elapsed() > 0 and finished else 0.0
        remaining = (self.total - finished) / rate if rate > 0 else float("nan")
        self.path.write_text(
            json.dumps(
                {
                    "label": self.label,
                    "pid": self.pid,
                    "run_id": self.run_id,
                    # A previous run's progress file sits here until this stage
                    # overwrites it, and a reader cannot otherwise tell the two
                    # apart -- which already misled one operator into thinking a
                    # 459-cell grid was an 18-cell one.  ``run_id`` settles it
                    # exactly; an mtime gap cannot, because an early stage
                    # legitimately finishes long before a late one.
                    "started_utc": datetime.fromtimestamp(self.started, UTC).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "total": self.total,
                    "done": self.done,
                    "skipped": self.skipped,
                    "failed": self.failed,
                    "failures": self.failures[-20:],
                    "elapsed_s": round(self._elapsed(), 1),
                    "eta_s": None if remaining != remaining else round(remaining, 1),
                },
                indent=2,
            )
            + "\n"
        )

    def record(self, *, done: int = 0, skipped: int = 0, failure: str | None = None) -> None:
        self.done += done
        self.skipped += skipped
        if failure is not None:
            self.failed += 1
            self.failures.append(failure)
        self.write()

    def summary(self) -> str:
        return (
            f"{self.label}: {self.done} done, {self.skipped} skipped, "
            f"{self.failed} failed of {self.total} in {self._elapsed() / 60:.1f} min"
        )
