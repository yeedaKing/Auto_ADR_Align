# uvicorn api.main:app --reload --port 8000

# api/main.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.jobs import JobStore
from api.schemas import AlignCreateResponse, JobInfo, JobList, ArtifactInfo

# Import pipeline runner
# Requires bin/ to be a package (bin/__init__.py)
from bin.adr_align import run_align
from core.features import FeatureConfig
from core.dtw_map import DTWConfig
from bin.adr_align import run_align, GuardrailConfig


def _iso(dt) -> str:
    return dt.isoformat()


def _safe_write_upload(dst: Path, up: UploadFile) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _artifact_info(base_url: str, job_id: str, path: Path) -> ArtifactInfo:
    size = None
    try:
        size = path.stat().st_size
    except OSError:
        pass
    return ArtifactInfo(
        name=path.name,
        url=f"{base_url}/jobs/{job_id}/artifact/{path.name}",
        size_bytes=size,
    )


RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs_api")).resolve()
store = JobStore(RUNS_DIR)

app = FastAPI(title="Auto-ADR Align API", version="0.1")

# For eventual React UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Auto-ADR Align API",
        "endpoints": ["/health", "/docs", "/align", "/jobs"]
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/align", response_model=AlignCreateResponse)
async def align(
    guide: UploadFile = File(..., description="Guide audio file (wav/flac)"),
    adr: UploadFile = File(..., description="ADR audio file (wav/flac)"),

    # Outputs
    render: bool = Form(False),
    qc: bool = Form(False),
    fade_ms: float = Form(20.0),

    # DTW options (defaults match your current setup)
    band: float = Form(0.15),
    step_penalty: float = Form(1e-3),
    anchor_every: int = Form(5),
    simplify_eps: float = Form(0.020),
    slope_min: float = Form(0.0),
    slope_max: float = Form(3.0),

    # Feature options (keep minimal)
    frame_ms: float = Form(25.0),
    hop_ms: float = Form(10.0),
    n_mfcc: int = Form(20),
    n_mels: int = Form(40),

    # QC options
    qc_cost_percentile: float = Form(95.0),

    # Optional
    write_path: bool = Form(True),
    segment_guide: bool = Form(False),
    seg_min_silence: float = Form(0.35),
    seg_rel_db: float = Form(20.0),

    # Guardrails
    render_cost_max: float = Form(0.25),
    qc_on_render_skip: bool = Form(True),
    write_summary: bool = Form(True),
    plateau_warn_s: float = Form(2.0),
):
    job = store.create_job()

    # Save uploads into job folder
    in_dir = job.out_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    guide_path = in_dir / f"guide{Path(guide.filename or 'guide').suffix or '.wav'}"
    adr_path = in_dir / f"adr{Path(adr.filename or 'adr').suffix or '.wav'}"

    _safe_write_upload(guide_path, guide)
    _safe_write_upload(adr_path, adr)

    # Build configs
    feat_cfg = FeatureConfig(
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
    )
    dtw_cfg = DTWConfig(
        band_frac=band,
        step_penalty=step_penalty,
        anchor_every=anchor_every,
        simplify_eps_s=simplify_eps,
        slope_min=slope_min,
        slope_max=slope_max,
    )
    guardrails = GuardrailConfig(
        render_cost_max=render_cost_max,
        qc_on_render_skip=qc_on_render_skip,
        write_summary=write_summary,
        plateau_warn_s=plateau_warn_s,
    )

    def _do_work():
        # Put artifacts directly in job.out_dir (not inside inputs/)
        run_align(
            guide_path=str(guide_path),
            adr_path=str(adr_path),
            out_dir=job.out_dir,
            feat_cfg=feat_cfg,
            dtw_cfg=dtw_cfg,
            write_path=write_path,
            segment_guide=segment_guide,
            seg_min_silence=seg_min_silence,
            seg_rel_db=seg_rel_db,
            render=render,
            render_out=str(job.out_dir / "adr_conformed.wav") if render else None,
            fade_ms=fade_ms,
            qc=qc,
            qc_cost_percentile=qc_cost_percentile,
            qc_out=str(job.out_dir / "qc_segments.csv") if qc else None,
            guardrails=guardrails,
        )

    store.run_in_thread(job.job_id, _do_work)

    base = ""  # keep relative; clients can prefix host
    return AlignCreateResponse(
        job_id=job.job_id,
        status="QUEUED",
        status_url=f"{base}/jobs/{job.job_id}",
    )


@app.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Build artifact list from current directory state
    artifacts = []
    base = ""
    if job.out_dir.exists():
        for p in sorted(job.out_dir.iterdir()):
            if p.is_file():
                artifacts.append(_artifact_info(base, job.job_id, p))

    return JobInfo(
        job_id=job.job_id,
        status=job.status,
        created_at=_iso(job.created_at),
        updated_at=_iso(job.updated_at),
        error=job.error,
        artifacts=artifacts,
    )


@app.get("/jobs", response_model=JobList)
def list_jobs():
    jobs = store.list_jobs()
    base = ""
    out = []
    for j in jobs:
        artifacts = []
        if j.out_dir.exists():
            for p in sorted(j.out_dir.iterdir()):
                if p.is_file():
                    artifacts.append(_artifact_info(base, j.job_id, p))
        out.append(JobInfo(
            job_id=j.job_id,
            status=j.status,
            created_at=_iso(j.created_at),
            updated_at=_iso(j.updated_at),
            error=j.error,
            artifacts=artifacts,
        ))
    return JobList(jobs=out)


@app.get("/jobs/{job_id}/artifact/{name}")
def get_artifact(job_id: str, name: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    path = (job.out_dir / name).resolve()

    # Security: ensure artifact is within the job directory
    if job.out_dir.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(str(path), filename=path.name)
