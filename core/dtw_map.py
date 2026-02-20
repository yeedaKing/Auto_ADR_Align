# core/dtw_map.py
"""
DTW alignment + time map building for Auto-ADR Align (no librosa).

Inputs:
- FeatureBatch for ADR and guide (from core.features.extract_mfcc)
  each has feat shape (T, D) and hop_s

Outputs:
- DTWResult:
    path: list[(i_adr, i_guide)] frame indices along optimal path
    tmap: np.ndarray shape (K, 2) of (t_adr_sec, t_guide_sec) anchors (monotone)
    stats: summary dict

Notes:
- Uses banded DTW with cosine distance.
- Intended to run phrase-wise (segments), not hours-long audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

from core.features import FeatureBatch, cosine_normalize


@dataclass(frozen=True)
class DTWConfig:
    band_frac: float = 0.15      # Sakoe-Chiba band as fraction of max(Ta, Tg)
    step_penalty: float = 1e-3
    slope_min: float = 0.00 # 0.60      # allowed local slope dt_guide/dt_adr
    slope_max: float = 3.00 # 1.60
    # Time map sampling
    anchor_every: int = 5        # keep every Nth path point as anchor before simplification
    simplify_eps_s: float = 0.020  # simplify anchors: max time error (sec) in guide time
    # Safety / numeric
    eps: float = 1e-8


@dataclass
class DTWResult:
    path: List[Tuple[int, int]]         # (i_adr, i_guide) frames along DTW path
    anchors: np.ndarray                 # (K, 2): (t_adr, t_guide) in seconds
    stats: Dict[str, float]


def _anchors_from_path_by_i(path, hop_a, hop_g, every_i: int) -> np.ndarray:
    if not path:
        raise ValueError("Empty path")

    Ta = path[-1][0] + 1
    buckets = [[] for _ in range(Ta)]
    for i, j in path:
        buckets[i].append(j)

    # pick a representative j for each i (median is stable)
    j_of_i = np.zeros(Ta, dtype=np.float64)
    last = 0.0
    for i in range(Ta):
        if buckets[i]:
            val = float(np.median(buckets[i]))
        else:
            val = last
        # enforce monotone nondecreasing
        val = max(val, last)
        j_of_i[i] = val
        last = val

    every_i = max(1, int(every_i))
    idx = np.arange(0, Ta, every_i, dtype=int)
    if idx[-1] != Ta - 1:
        idx = np.append(idx, Ta - 1)

    anchors = np.column_stack((idx.astype(np.float64) * hop_a,
                               j_of_i[idx] * hop_g))
    return anchors



def _cosine_distance_matrix(A: np.ndarray, B: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute pairwise cosine distance between A (Ta,D) and B (Tg,D).
    Returns (Ta, Tg) matrix with values in [0, 2].
    """
    # Normalize rows
    A2 = cosine_normalize(A, eps=eps)
    B2 = cosine_normalize(B, eps=eps)
    # cosine similarity = dot
    sim = A2 @ B2.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    return dist.astype(np.float32)


def _band_limits(Ta: int, Tg: int, band_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each i in [0,Ta), define j range [j_lo[i], j_hi[i]] inclusive
    around the diagonal mapping i/Ta ~ j/Tg.

    Returns arrays j_lo, j_hi of length Ta.
    """
    if Ta <= 0 or Tg <= 0:
        raise ValueError("Empty sequence length")

    band = int(np.ceil(band_frac * max(Ta, Tg)))
    j_lo = np.zeros(Ta, dtype=np.int32)
    j_hi = np.zeros(Ta, dtype=np.int32)

    for i in range(Ta):
        # expected diagonal j
        j_center = int(round((i * (Tg - 1)) / max(1, (Ta - 1))))
        lo = max(0, j_center - band)
        hi = min(Tg - 1, j_center + band)
        j_lo[i] = lo
        j_hi[i] = hi

    return j_lo, j_hi


def banded_dtw_path(cost: np.ndarray, j_lo: np.ndarray, j_hi: np.ndarray, step_penalty: float) -> List[Tuple[int, int]]:
    """
    Compute optimal DTW path with 3 allowed steps:
      (i-1, j)   vertical
      (i,   j-1) horizontal
      (i-1, j-1) diagonal

    cost: (Ta, Tg)
    band defined by j_lo/j_hi.

    Returns path as list of (i, j) from start->end.
    """
    Ta, Tg = cost.shape

    INF = 1e18
    dp = np.full((Ta, Tg), INF, dtype=np.float64)
    bt = np.full((Ta, Tg, 2), -1, dtype=np.int32)  # backtrace prev (pi,pj)

    # Initialize start
    dp[0, 0] = float(cost[0, 0])
    bt[0, 0] = (-1, -1)

    # Fill DP within band
    for i in range(Ta):
        lo = int(j_lo[i])
        hi = int(j_hi[i])
        for j in range(lo, hi + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            prev = (-1, -1)

           # from (i-1, j) vertical
            if i - 1 >= 0 and j_lo[i-1] <= j <= j_hi[i-1]:
                cand = dp[i-1, j] + step_penalty
                if cand < best:
                    best = cand; prev = (i-1, j)

            # from (i, j-1) horizontal
            if j - 1 >= 0 and lo <= (j - 1) <= hi:
                cand = dp[i, j-1] + step_penalty
                if cand < best:
                    best = cand; prev = (i, j-1)

            # from (i-1, j-1) diagonal
            if i - 1 >= 0 and j - 1 >= 0 and j_lo[i-1] <= (j-1) <= j_hi[i-1]:
                cand = dp[i-1, j-1]  # no penalty
                if cand < best:
                    best = cand; prev = (i-1, j-1)

            dp[i, j] = best + float(cost[i, j])
            bt[i, j] = prev

    # Choose end point: typically (Ta-1, Tg-1) if reachable, else best in last row/col within band.
    end = (Ta - 1, Tg - 1)
    if not np.isfinite(dp[end]):
        # fallback: best in last row within band
        i = Ta - 1
        lo, hi = int(j_lo[i]), int(j_hi[i])
        j_best = lo + int(np.argmin(dp[i, lo:hi + 1]))
        end = (i, j_best)
        if not np.isfinite(dp[end]):
            raise RuntimeError("DTW failed: end not reachable within band. Try increasing band_frac.")

    # Backtrace
    path_rev: List[Tuple[int, int]] = []
    i, j = end
    while i >= 0 and j >= 0:
        path_rev.append((i, j))
        pi, pj = bt[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = int(pi), int(pj)

    path = list(reversed(path_rev))

    # Basic monotonicity check
    for k in range(1, len(path)):
        if path[k][0] < path[k - 1][0] or path[k][1] < path[k - 1][1]:
            raise AssertionError("Non-monotone DTW path detected")

    return path


def _rdp_simplify_xy(points: np.ndarray, eps_y: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker simplification, where error is measured on y-axis only.
    points: (K,2) with x monotone
    eps_y: max allowed absolute y error
    """
    if points.shape[0] <= 2:
        return points

    x = points[:, 0]
    y = points[:, 1]

    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, points.shape[0] - 1)]
    while stack:
        a, b = stack.pop()
        xa, xb = x[a], x[b]
        ya, yb = y[a], y[b]
        if xb <= xa + 1e-12:
            continue

        # line y = ya + t*(yb-ya)
        idx = np.arange(a + 1, b)
        t = (x[idx] - xa) / (xb - xa)
        y_hat = ya + t * (yb - ya)
        err = np.abs(y[idx] - y_hat)
        m = int(np.argmax(err)) if err.size else -1
        if err.size and float(err[m]) > eps_y:
            i = int(idx[m])
            keep[i] = True
            stack.append((a, i))
            stack.append((i, b))

    return points[keep]


def _clamp_slopes(anchors: np.ndarray, slope_min: float, slope_max: float) -> Tuple[np.ndarray, int]:
    """
    Enforce slope bounds on consecutive anchor segments by projecting y-values
    into feasible intervals. Two-pass (forward/backward) keeps constraints consistent.

    Guarantees (for all k):
        slope_min <= (y[k+1]-y[k])/(x[k+1]-x[k]) <= slope_max
    assuming x is nondecreasing and dx>0 for consecutive points.
    """
    x = anchors[:, 0].astype(np.float64).copy()
    y = anchors[:, 1].astype(np.float64).copy()

    K = len(x)
    if K <= 1:
        return anchors.astype(np.float64), 0

    clamps = 0

    # Ensure monotone x (should already be true)
    x = np.maximum.accumulate(x)

    # Forward pass: constrain y[k] based on y[k-1]
    for k in range(1, K):
        dx = x[k] - x[k - 1]
        if dx <= 1e-12:
            # collapse duplicate x by keeping y monotone
            if y[k] < y[k - 1]:
                y[k] = y[k - 1]
                clamps += 1
            continue

        lo = y[k - 1] + slope_min * dx
        hi = y[k - 1] + slope_max * dx
        y_new = min(max(y[k], lo), hi)
        if abs(y_new - y[k]) > 1e-12:
            clamps += 1
            y[k] = y_new

        # Also enforce monotone y
        if y[k] < y[k - 1]:
            y[k] = y[k - 1]
            clamps += 1

    # Backward pass: constrain y[k] based on y[k+1] (keeps future feasibility)
    for k in range(K - 2, -1, -1):
        dx = x[k + 1] - x[k]
        if dx <= 1e-12:
            # ensure monotone backwards (should already hold)
            if y[k] > y[k + 1]:
                y[k] = y[k + 1]
                clamps += 1
            continue

        lo = y[k + 1] - slope_max * dx
        hi = y[k + 1] - slope_min * dx
        y_new = min(max(y[k], lo), hi)
        if abs(y_new - y[k]) > 1e-12:
            clamps += 1
            y[k] = y_new

        if y[k] > y[k + 1]:
            y[k] = y[k + 1]
            clamps += 1

    out = np.column_stack((x, y)).astype(np.float64)
    out[:, 0] = np.maximum.accumulate(out[:, 0])
    out[:, 1] = np.maximum.accumulate(out[:, 1])
    return out, clamps


def align_feature_batches(adr: FeatureBatch, guide: FeatureBatch, cfg: DTWConfig = DTWConfig()) -> DTWResult:
    """
    Align ADR to guide using MFCC features + banded DTW.

    Returns DTWResult with path + simplified anchors + stats.
    """
    A = adr.feat
    G = guide.feat
    Ta, Tg = A.shape[0], G.shape[0]
    if Ta == 0 or Tg == 0:
        raise ValueError("Empty feature sequence")

    # cost matrix (Ta, Tg)
    cost = _cosine_distance_matrix(A, G, eps=cfg.eps)

    # band limits per i in ADR
    j_lo, j_hi = _band_limits(Ta, Tg, cfg.band_frac)

    # DTW path
    path = banded_dtw_path(cost, j_lo, j_hi, step_penalty=cfg.step_penalty)

    # anchors from path -> simplify -> clamp slopes
    anchors = _anchors_from_path_by_i(path, adr.hop_s, guide.hop_s, every_i=cfg.anchor_every)
    anchors = _rdp_simplify_xy(anchors, eps_y=cfg.simplify_eps_s)
    anchors, clamps = _clamp_slopes(anchors, cfg.slope_min, cfg.slope_max)

    # stats
    path_cost = float(np.mean([cost[i, j] for (i, j) in path]))
    # estimate max slope on anchors
    dx = np.diff(anchors[:, 0])
    dy = np.diff(anchors[:, 1])
    mask = dx > 1e-9
    slopes = dy[mask] / dx[mask] if np.any(mask) else np.array([], dtype=np.float64)

    # debug
    dx0 = np.diff(anchors[:, 0])
    dy0 = np.diff(anchors[:, 1])
    m = dx0 > 1e-9
    sl0 = dy0[m] / dx0[m]
    worst = np.argsort(sl0)[-5:]
    print("[dtw] worst slopes:", sl0[worst])

    stats = {
        "Ta": float(Ta),
        "Tg": float(Tg),
        "path_len": float(len(path)),
        "mean_cost": path_cost,
        "min_slope": float(np.min(slopes)) if slopes.size else 0.0,
        "max_slope": float(np.max(slopes)) if slopes.size else 0.0,
        "num_clamps": float(clamps),
        "anchors_k": float(anchors.shape[0]),
        "band_frac": float(cfg.band_frac),
    }

    return DTWResult(path=path, anchors=anchors.astype(np.float64), stats=stats)


# ----------------------------
# Smoke test
# ----------------------------

# python3 -c "from core.dtw_map import _smoke_test; _smoke_test('playground/mono48k.wav','playground/mono48kadr.wav')"
def _smoke_test(guide_path: str, adr_path: str) -> None:
    from core.io_utils import ensure_mono_48k
    from core.features import extract_mfcc, FeatureConfig

    g = ensure_mono_48k(guide_path)
    a = ensure_mono_48k(adr_path)

    fb_g = extract_mfcc(g.samples, g.sr, FeatureConfig())
    fb_a = extract_mfcc(a.samples, a.sr, FeatureConfig())

    res = align_feature_batches(fb_a, fb_g, DTWConfig())

    print("[dtw] guide:", guide_path)
    print("[dtw] adr  :", adr_path)
    print("[dtw] stats:", {k: round(v, 4) for k, v in res.stats.items()})
    print("[dtw] anchors (first 5):")
    for row in res.anchors[:5]:
        print(f"  t_adr={row[0]:.3f}s -> t_guide={row[1]:.3f}s")
    print("[dtw] anchors (last 3):")
    for row in res.anchors[-3:]:
        print(f"  t_adr={row[0]:.3f}s -> t_guide={row[1]:.3f}s")


# python3 -m core.dtw_map --guide playground/mono48k.wav --adr playground/mono48kadr.wav
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DTW alignment smoke test")
    parser.add_argument("--guide", required=True, help="Guide wav/flac")
    parser.add_argument("--adr", required=True, help="ADR wav/flac")
    parser.add_argument("--band", type=float, default=0.15, help="Band fraction (0.05-0.30 typical)")
    args = parser.parse_args()

    # Run with custom band if provided
    from core.io_utils import ensure_mono_48k
    from core.features import extract_mfcc, FeatureConfig

    g = ensure_mono_48k(args.guide)
    a = ensure_mono_48k(args.adr)
    fb_g = extract_mfcc(g.samples, g.sr, FeatureConfig())
    fb_a = extract_mfcc(a.samples, a.sr, FeatureConfig())

    cfg = DTWConfig(band_frac=args.band)
    res = align_feature_batches(fb_a, fb_g, cfg)

    print("[dtw] stats:", {k: round(v, 4) for k, v in res.stats.items()})
    print("[dtw] anchors_k:", int(res.stats["anchors_k"]))
