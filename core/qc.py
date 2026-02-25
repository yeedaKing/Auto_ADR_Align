# core/qc.py
"""
QC (quality control) heuristics for Auto-ADR Align.

No ASR required. Uses:
- anchors (t_adr -> t_guide)
- DTW path + features to compute local cosine-distance costs

Produces:
- qc_segments.csv with suspicious regions and reasons.

Typical usage (standalone):
  python3 -m core.qc --guide ... --adr ... --out outputs/qc_run

Or call from bin/adr_align.py after you compute:
  - FeatureBatch for adr/guide
  - DTWResult (path, anchors, stats)
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from core.features import FeatureBatch, cosine_normalize


@dataclass(frozen=True)
class QCConfig:
    # minimum duration for a reported segment
    min_segment_s: float = 0.20

    # plateau detection (guide time barely changes while ADR time advances)
    plateau_slope_max: float = 0.03     # slope <= this considered "plateau-ish"
    plateau_min_s: float = 0.30         # require at least this long

    # max-speed detection (slope near slope_max)
    maxspeed_tol: float = 0.03          # flag if slope >= slope_max - tol
    maxspeed_min_s: float = 0.20

    # high-cost detection on DTW path (cosine distance along path)
    cost_percentile: float = 95.0       # flag above this percentile
    cost_min_s: float = 0.20

    # severity thresholds (seconds)
    sev2_s: float = 0.50
    sev3_s: float = 1.50

    eps: float = 1e-12


@dataclass(frozen=True)
class QCSegment:
    start_adr_sec: float
    end_adr_sec: float
    start_guide_sec: float
    end_guide_sec: float
    reason: str                 # PLATEAU / MAXSPEED / HIGH_COST
    severity: int               # 1..3
    value: float                # slope or cost threshold metric


# ----------------------------
# Small utilities
# ----------------------------

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _atomic_write_csv(path: str, rows: List[List[object]]) -> None:
    _ensure_parent_dir(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _interp_fwd_from_anchors(anchors: np.ndarray):
    """
    Returns callable f(t_adr)->t_guide using linear interpolation on anchors.
    """
    a = np.asarray(anchors, dtype=np.float64)
    tA = np.maximum.accumulate(a[:, 0])
    tG = np.maximum.accumulate(a[:, 1])

    def f(t_adr: np.ndarray) -> np.ndarray:
        return np.interp(t_adr, tA, tG, left=tG[0], right=tG[-1])

    return f


def _merge_time_segments(segs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/adjacent segments (assumes segs in seconds)."""
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(a), float(b)) for a, b in out]


def _duration_to_severity(d: float, cfg: QCConfig) -> int:
    if d >= cfg.sev3_s:
        return 3
    if d >= cfg.sev2_s:
        return 2
    return 1


# ----------------------------
# QC computations
# ----------------------------

def qc_from_anchors(
    anchors: np.ndarray,
    slope_max: float,
    cfg: QCConfig = QCConfig(),
) -> List[QCSegment]:
    """
    QC segments from anchor slopes:
    - PLATEAU: slope <= plateau_slope_max
    - MAXSPEED: slope >= slope_max - tol
    """
    a = np.asarray(anchors, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        return []

    fwd = _interp_fwd_from_anchors(a)

    tA0 = a[:-1, 0]
    tA1 = a[1:, 0]
    tG0 = a[:-1, 1]
    tG1 = a[1:, 1]

    dx = tA1 - tA0
    dy = tG1 - tG0
    slope = np.zeros_like(dx)
    m = dx > cfg.eps
    slope[m] = dy[m] / dx[m]

    segs: List[QCSegment] = []

    # PLATEAU segments
    plateau_mask = (dx > cfg.eps) & (slope <= cfg.plateau_slope_max)
    segs += _mask_to_qc_segments(
        tA0, tA1, plateau_mask, fwd,
        reason="PLATEAU",
        value_arr=slope,
        min_len_s=cfg.plateau_min_s,
        cfg=cfg,
    )

    # MAXSPEED segments (near slope_max)
    max_mask = (dx > cfg.eps) & (slope >= (slope_max - cfg.maxspeed_tol))
    segs += _mask_to_qc_segments(
        tA0, tA1, max_mask, fwd,
        reason="MAXSPEED",
        value_arr=slope,
        min_len_s=cfg.maxspeed_min_s,
        cfg=cfg,
    )

    return segs


def qc_from_dtw_cost(
    adr_fb: FeatureBatch,
    guide_fb: FeatureBatch,
    path: List[Tuple[int, int]],
    anchors: np.ndarray,
    cfg: QCConfig = QCConfig(),
) -> List[QCSegment]:
    """
    Compute cosine-distance along DTW path and flag high-cost ADR-time regions.

    Implementation:
    - For each ADR frame i, collect all path costs for that i and take median.
    - Flag frames above percentile threshold.
    - Merge into time segments.
    """
    if not path:
        return []

    A = cosine_normalize(adr_fb.feat)
    G = cosine_normalize(guide_fb.feat)

    Ta = A.shape[0]
    costs_per_i: List[List[float]] = [[] for _ in range(Ta)]

    for ia, ig in path:
        if 0 <= ia < Ta and 0 <= ig < G.shape[0]:
            dot = float(np.dot(A[ia], G[ig]))
            dot = float(np.clip(dot, -1.0, 1.0))
            c = 1.0 - dot
            costs_per_i[ia].append(c)

    # median cost per ADR frame, fill missing by forward-fill
    cost_i = np.zeros((Ta,), dtype=np.float64)
    last = 0.0
    for i in range(Ta):
        if costs_per_i[i]:
            last = float(np.median(costs_per_i[i]))
        cost_i[i] = last

    # threshold by percentile
    thr = float(np.percentile(cost_i, cfg.cost_percentile))
    high = cost_i >= thr

    # convert frame mask -> time segments
    hop = adr_fb.hop_s
    segs_raw: List[Tuple[float, float]] = []
    i = 0
    while i < Ta:
        if not high[i]:
            i += 1
            continue
        s = i
        i += 1
        while i < Ta and high[i]:
            i += 1
        e = i  # exclusive
        start_t = s * hop
        end_t = e * hop
        if (end_t - start_t) >= cfg.cost_min_s:
            segs_raw.append((start_t, end_t))

    segs_raw = _merge_time_segments(segs_raw)

    fwd = _interp_fwd_from_anchors(anchors)
    out: List[QCSegment] = []
    for s, e in segs_raw:
        if (e - s) < cfg.min_segment_s:
            continue
        sev = _duration_to_severity(e - s, cfg)
        # representative value: max cost within this region
        i0 = int(max(0, min(Ta - 1, round(s / hop))))
        i1 = int(max(0, min(Ta, round(e / hop))))
        val = float(np.max(cost_i[i0:i1])) if i1 > i0 else float(cost_i[i0])
        gs = float(fwd(np.asarray([s], dtype=np.float64))[0])
        ge = float(fwd(np.asarray([e], dtype=np.float64))[0])
        out.append(QCSegment(s, e, gs, ge, "HIGH_COST", sev, val))

    # include threshold in value? (optional) — keep value as max cost; threshold can be written in header/logs.
    return out


def _mask_to_qc_segments(
    tA0: np.ndarray,
    tA1: np.ndarray,
    mask: np.ndarray,
    fwd_map,
    reason: str,
    value_arr: np.ndarray,
    min_len_s: float,
    cfg: QCConfig,
) -> List[QCSegment]:
    """
    Convert interval mask on anchor segments into merged QC segments.
    """
    segs: List[Tuple[float, float, float]] = []  # (start_adr, end_adr, value)
    for k in range(mask.size):
        if not mask[k]:
            continue
        s = float(tA0[k])
        e = float(tA1[k])
        if e - s <= 0:
            continue
        segs.append((s, e, float(value_arr[k])))

    if not segs:
        return []

    # merge contiguous/overlapping by ADR time
    segs_sorted = sorted(segs, key=lambda x: x[0])
    merged: List[Tuple[float, float, float]] = []
    cur_s, cur_e, cur_v = segs_sorted[0]
    for s, e, v in segs_sorted[1:]:
        if s <= cur_e + 1e-9:
            cur_e = max(cur_e, e)
            # keep worst/most indicative value
            if reason == "PLATEAU":
                cur_v = min(cur_v, v)  # lower slope is worse
            else:
                cur_v = max(cur_v, v)
        else:
            merged.append((cur_s, cur_e, cur_v))
            cur_s, cur_e, cur_v = s, e, v
    merged.append((cur_s, cur_e, cur_v))

    out: List[QCSegment] = []
    for s, e, v in merged:
        d = e - s
        if d < min_len_s or d < cfg.min_segment_s:
            continue
        sev = _duration_to_severity(d, cfg)
        gs = float(fwd_map(np.asarray([s], dtype=np.float64))[0])
        ge = float(fwd_map(np.asarray([e], dtype=np.float64))[0])
        out.append(QCSegment(s, e, gs, ge, reason, sev, v))

    return out


def compute_qc_segments(
    adr_fb: FeatureBatch,
    guide_fb: FeatureBatch,
    path: List[Tuple[int, int]],
    anchors: np.ndarray,
    slope_max: float,
    cfg: QCConfig = QCConfig(),
) -> List[QCSegment]:
    """
    Combine multiple QC detectors and return merged list (sorted).
    """
    segs: List[QCSegment] = []
    segs.extend(qc_from_anchors(anchors, slope_max=slope_max, cfg=cfg))
    segs.extend(qc_from_dtw_cost(adr_fb, guide_fb, path, anchors, cfg=cfg))

    # Sort by ADR time then reason
    segs.sort(key=lambda s: (s.start_adr_sec, s.reason))
    return segs


def write_qc_segments_csv(segs: List[QCSegment], out_path: str, precision: int = 6) -> None:
    fmt = f"{{:.{precision}f}}"
    rows: List[List[object]] = [[
        "start_adr_sec", "end_adr_sec",
        "start_guide_sec", "end_guide_sec",
        "duration_sec", "reason", "severity", "value"
    ]]
    for s in segs:
        rows.append([
            fmt.format(s.start_adr_sec),
            fmt.format(s.end_adr_sec),
            fmt.format(s.start_guide_sec),
            fmt.format(s.end_guide_sec),
            fmt.format(s.end_adr_sec - s.start_adr_sec),
            s.reason,
            int(s.severity),
            fmt.format(float(s.value)),
        ])
    _atomic_write_csv(out_path, rows)


# ----------------------------
# Standalone CLI for testing
# ----------------------------

"""
python3 -m core.qc \
  --guide playground/mono48k.wav \
  --adr playground/mono48kadr.wav \
  --out outputs/test_qc \
  --slope_max 3.0 \
  --cost_percentile 95
"""
if __name__ == "__main__":
    import argparse
    from core.io_utils import ensure_mono_48k
    from core.features import extract_mfcc, FeatureConfig
    from core.dtw_map import align_feature_batches, DTWConfig
    from core.export import write_anchors_csv, write_dtw_path_csv, write_stats_csv

    p = argparse.ArgumentParser(description="QC for Auto-ADR Align (no ASR)")
    p.add_argument("--guide", required=True, help="Guide wav/flac")
    p.add_argument("--adr", required=True, help="ADR wav/flac")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--band", type=float, default=0.15)
    p.add_argument("--slope_max", type=float, default=3.0)
    p.add_argument("--cost_percentile", type=float, default=95.0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # align
    g = ensure_mono_48k(args.guide)
    a = ensure_mono_48k(args.adr)
    fb_g = extract_mfcc(g.samples, g.sr, FeatureConfig())
    fb_a = extract_mfcc(a.samples, a.sr, FeatureConfig())

    dtw_cfg = DTWConfig(band_frac=args.band, slope_max=args.slope_max)
    res = align_feature_batches(fb_a, fb_g, dtw_cfg)

    # write alignment artifacts (handy when debugging QC)
    write_anchors_csv(res.anchors, os.path.join(args.out, "anchors.csv"))
    write_dtw_path_csv(res.path, fb_a.hop_s, fb_g.hop_s, os.path.join(args.out, "path.csv"))
    write_stats_csv(res.stats, os.path.join(args.out, "stats.csv"))

    # QC
    qc_cfg = QCConfig(cost_percentile=args.cost_percentile)
    segs = compute_qc_segments(fb_a, fb_g, res.path, res.anchors, slope_max=args.slope_max, cfg=qc_cfg)
    out_csv = os.path.join(args.out, "qc_segments.csv")
    write_qc_segments_csv(segs, out_csv)
    print(f"[qc] wrote: {out_csv}  (n={len(segs)})")
