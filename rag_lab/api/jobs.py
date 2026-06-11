"""Tiny in-process job manager: runs one heavy job at a time in a worker thread
and exposes live progress so the Vue app can show a progress bar / log."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from typing import Callable, Optional


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"  # running | done | error
        self.progress = -1.0      # 0..1, or -1 for indeterminate
        self.stage = ""
        self.log: deque[str] = deque(maxlen=200)
        self.error: Optional[str] = None
        self.result: Optional[dict] = None
        self.started = time.time()
        self.finished: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "log": list(self.log),
            "error": self.error,
            "result": self.result,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1),
        }


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.current: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, kind: str, fn: Callable[[Job], dict]) -> Job:
        if self.is_busy():
            raise RuntimeError(f"A job ({self.current.kind}) is already running.")
        job = Job(kind)
        self.current = job

        def runner():
            try:
                job.result = fn(job)
                job.status = "done"
                job.progress = 1.0
                job.stage = "done"
            except Exception as e:  # noqa
                job.status = "error"
                job.error = f"{e}\n{traceback.format_exc()[-1500:]}"
                job.log.append(f"ERROR: {e}")
            finally:
                job.finished = time.time()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return job

    def status(self) -> Optional[dict]:
        return self.current.to_dict() if self.current else None


MANAGER = JobManager()
