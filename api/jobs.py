# api/jobs.py
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    job_id: str
    out_dir: Path
    status: str = "QUEUED"  # QUEUED | RUNNING | DONE | ERROR
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _utc_now()


class JobStore:
    """
    Simple in-memory job store with background thread execution.
    Suitable for a local/dev server (single process).
    """

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def create_job(self) -> Job:
        job_id = uuid.uuid4().hex
        out_dir = self.runs_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        job = Job(job_id=job_id, out_dir=out_dir)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.error = error
            job.touch()

    def _set_artifacts_from_dir(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.out_dir.exists():
                files = sorted([p.name for p in job.out_dir.iterdir() if p.is_file()])
            else:
                files = []
            job.artifacts = files
            job.touch()

    def run_in_thread(self, job_id: str, fn: Callable[[], None]) -> None:
        """
        Run fn() in a daemon thread and update job status/artifacts.
        """
        def _runner():
            self._set_status(job_id, "RUNNING")
            try:
                fn()
                self._set_artifacts_from_dir(job_id)
                self._set_status(job_id, "DONE")
            except Exception as e:
                self._set_artifacts_from_dir(job_id)
                self._set_status(job_id, "ERROR", error=str(e))

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
