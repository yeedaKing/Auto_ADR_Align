# api/schemas.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class AlignCreateResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


class ArtifactInfo(BaseModel):
    name: str
    url: str
    size_bytes: Optional[int] = None


class JobInfo(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None
    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class JobList(BaseModel):
    jobs: List[JobInfo]
