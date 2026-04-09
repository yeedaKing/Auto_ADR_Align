# Auto-ADR Align

Automatic alignment and conforming of ADR dialogue to a guide track using DTW-based time mapping.

Auto-ADR Align is a tool that aligns an ADR (automated dialogue replacement) recording to a guide track.  
It computes a time map using dynamic time warping (DTW), then optionally renders a conformed audio file that matches the guide timing.

This project focuses on practical alignment accuracy for speech, with guardrails and parameter tuning for stable results.

## Demo

1. Upload a guide audio file and ADR take
2. Adjust parameters (optional)
3. Run alignment
4. Download the conformed ADR output

Example output:
- `adr_conformed.wav` (aligned to guide timing)

## Features

- DTW-based alignment between guide and ADR audio
- Anchor-based time mapping
- Optional audio rendering with WSOLA-style time stretching
- Quality control (QC) output for alignment validation
- Parameter tuning for alignment and rendering behavior

## How to Run

### Backend

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Open in Browser
http://localhost:8000/ui

---

## Project Structure

- `api/` – FastAPI backend (job handling, endpoints)
- `core/`
  - `features.py` – feature extraction
  - `dtw_map.py` – DTW alignment + anchor generation
  - `render.py` – audio conforming
  - `export.py` - CSV export
  - `io_utils.py` - Audio I/O utilities
  - `qc.py` - Quality control heuristics
  - `segment.py` - Phrase segmentation
- `bin/adr_align.py` – main pipeline
- `web/` – frontend UI

## Key Insight

A slope-clamping step in the time map generation was initially used to constrain local timing changes.  
However, this caused underestimation of cumulative delay in later segments, resulting in insufficient spacing in the rendered output.

Disabling slope clamping improved alignment accuracy by preserving the true DTW-derived timing structure.

## Recommended Parameters

- Fade: 40 ms
- Anchor every: 5 frames
- Simplify eps: 0.040-0.080 sec
- Step penalty: 0.005-0.01
- Render cost max: 0.30
- Band: 0.15
- Slope clamping (min/max slope): disabled

## Limitations

- Performance depends on audio quality and similarity
- Extreme timing differences may require parameter tuning
- Currently optimized for speech, not music or noisy recordings

## Future Work

- Improved handling of quick bursts of speech in one file but not the other
- Better time-map regularization without losing alignment accuracy
- Improved UI visualization of alignment path