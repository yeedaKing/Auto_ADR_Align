# core/render.py
"""
Render (conform) ADR audio to match guide timing using an anchor time map.

Inputs:
- guide audio file (used mainly for output length / sample rate)
- adr audio file
- anchors: (K,2) array of [t_adr_sec, t_guide_sec] (monotone nondecreasing)

Outputs:
- adr_conformed.wav: audio aligned to guide timeline (same duration as guide)

Approach:
- For each anchor interval:
    ADR segment [tA0, tA1] -> Guide interval [tG0, tG1]
  time-stretch/compress ADR segment to match guide interval length.
- Use overlap-add + WSOLA-style search for pitch-preserving-ish speech stretch.
- Apply short fades at segment boundaries and accumulate with weights.

Notes:
- Handles "flat" guide regions (tG1==tG0): we skip audio (effectively delete ADR time).
- Handles "flat" ADR regions (tA1==tA0): we insert silence for guide duration.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from core.io_utils import ensure_mono_48k, write_audio, Audio


@dataclass(frozen=True)
class RenderConfig:
    sr: int = 48000
    fade_ms: float = 20.0

    # WSOLA params (good speech defaults)
    wsola_win: int = 1024          # ~21 ms at 48k
    wsola_overlap: int = 512       # 50% overlap
    wsola_search: int = 256        # search radius for best overlap match

    # Fallback threshold
    min_wsola_samples: int = 4096  # below this, use linear resample


def read_anchors_csv(path: str) -> np.ndarray:
    """Read anchors.csv with header: t_adr_sec,t_guide_sec"""
    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for line in r:
            if not line or len(line) < 2:
                continue
            rows.append((float(line[0]), float(line[1])))
    if not rows:
        raise ValueError(f"No anchors found in {path}")
    a = np.asarray(rows, dtype=np.float64)
    # enforce monotone (safety)
    a[:, 0] = np.maximum.accumulate(a[:, 0])
    a[:, 1] = np.maximum.accumulate(a[:, 1])
    return a


def _linear_resample(x: np.ndarray, out_len: int) -> np.ndarray:
    """Fast fallback: time-scale by linear interpolation (NOT pitch-preserving)."""
    if out_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == 0:
        return np.zeros((out_len,), dtype=np.float32)
    if out_len == x.size:
        return x.astype(np.float32, copy=False)

    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=True, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=out_len, endpoint=True, dtype=np.float64)
    y = np.interp(xq, xp, x.astype(np.float64))
    return y.astype(np.float32, copy=False)


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def _normxcorr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized correlation between two 1D arrays of same length."""
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    if aa <= eps or bb <= eps:
        return -1e9
    return float(np.dot(a, b) / np.sqrt(aa * bb + eps))


def _wsola_time_stretch(
    x: np.ndarray,
    out_len: int,
    cfg: RenderConfig,
) -> np.ndarray:
    """
    WSOLA-like time-scaling to produce exactly out_len samples.

    This is designed for speech; it preserves pitch better than naive resampling.
    """
    if out_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    in_len = x.size
    if in_len == 0:
        return np.zeros((out_len,), dtype=np.float32)
    if in_len == out_len:
        return x.copy()

    N = int(cfg.wsola_win)
    O = int(cfg.wsola_overlap)
    S = int(cfg.wsola_search)
    if N <= 2 or O <= 0 or O >= N:
        return _linear_resample(x, out_len)

    Hs = N - O
    alpha = out_len / float(in_len)  # output/input time-scale
    Ha = max(1, int(round(Hs / max(alpha, 1e-9))))

    win = _hann(N)

    # Pad input so we can safely slice frames
    pad = N + S + 2
    xpad = np.pad(x, (0, pad), mode="constant")

    y = np.zeros((out_len + N + 2,), dtype=np.float32)

    y_pos = 0
    x_pos = 0

    # First frame
    frame0 = xpad[x_pos:x_pos + N]
    y[y_pos:y_pos + N] += frame0 * win
    y_pos += Hs
    x_pos += Ha

    # Subsequent frames
    while y_pos < out_len and x_pos < in_len:
        # Reference overlap in current output
        ref_start = max(0, y_pos)
        ref = y[ref_start:ref_start + O].copy()

        # Search best match around predicted x_pos
        best_off = 0
        best_score = -1e18

        base = x_pos
        lo = max(0, base - S)
        hi = base + S

        # If ref is near-silent, skip searching (avoid noise-driven instability)
        ref_energy = float(np.dot(ref, ref))
        do_search = ref_energy > 1e-6

        if do_search:
            for cand in range(lo, hi + 1):
                seg = xpad[cand:cand + O]
                score = _normxcorr(ref, seg)
                if score > best_score:
                    best_score = score
                    best_off = cand - base
        else:
            best_off = 0

        cand_pos = base + best_off
        frame = xpad[cand_pos:cand_pos + N]

        y[y_pos:y_pos + N] += frame * win

        # advance
        y_pos += Hs
        x_pos = cand_pos + Ha

    # Trim exact output length
    return y[:out_len].astype(np.float32, copy=False)


def _apply_fades(seg: np.ndarray, fade_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (seg_weighted, weight) with fade-in/out applied.
    We add into an accumulation buffer with weights, then normalize.
    """
    n = seg.size
    if n == 0:
        return seg, np.zeros((0,), dtype=np.float32)

    w = np.ones((n,), dtype=np.float32)

    f = min(fade_len, n // 2)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, num=f, endpoint=True, dtype=np.float32)
        w[:f] *= ramp
        w[-f:] *= ramp[::-1]

    return seg * w, w


def warp_adr_to_guide(
    adr_samples: np.ndarray,
    guide_len_samples: int,
    anchors: np.ndarray,
    cfg: RenderConfig,
) -> np.ndarray:
    """
    Render ADR onto the guide timeline using anchors.
    Output length == guide_len_samples.
    """
    sr = cfg.sr
    out = np.zeros((guide_len_samples,), dtype=np.float32)
    wsum = np.zeros((guide_len_samples,), dtype=np.float32)

    fade_len = int(round(cfg.fade_ms * 1e-3 * sr))

    # Ensure anchors include start at (0,0)
    a = anchors.astype(np.float64, copy=False)
    if a[0, 0] > 1e-6 or a[0, 1] > 1e-6:
        a = np.vstack(([0.0, 0.0], a))
    a[:, 0] = np.maximum.accumulate(a[:, 0])
    a[:, 1] = np.maximum.accumulate(a[:, 1])

    # Iterate anchor intervals
    for k in range(len(a) - 1):
        tA0, tG0 = float(a[k, 0]), float(a[k, 1])
        tA1, tG1 = float(a[k + 1, 0]), float(a[k + 1, 1])

        in_s = int(round(tA0 * sr))
        in_e = int(round(tA1 * sr))
        out_s = int(round(tG0 * sr))
        out_e = int(round(tG1 * sr))

        in_s = max(0, min(in_s, adr_samples.size))
        in_e = max(0, min(in_e, adr_samples.size))
        out_s = max(0, min(out_s, guide_len_samples))
        out_e = max(0, min(out_e, guide_len_samples))

        out_len = out_e - out_s
        in_len = in_e - in_s

        if out_len <= 0:
            # guide interval is zero: delete adr time here
            continue

        if in_len <= 0:
            # no adr content: insert silence
            seg = np.zeros((out_len,), dtype=np.float32)
        else:
            xseg = adr_samples[in_s:in_e]

            # Choose method
            if (in_len >= cfg.min_wsola_samples) and (out_len >= cfg.min_wsola_samples):
                seg = _wsola_time_stretch(xseg, out_len=out_len, cfg=cfg)
            else:
                seg = _linear_resample(xseg, out_len)

        seg_w, w = _apply_fades(seg, fade_len)
        out[out_s:out_e] += seg_w
        wsum[out_s:out_e] += w

    # Normalize overlaps
    mask = wsum > 1e-6
    out[mask] /= wsum[mask]
    return out


def render_conformed_wav(
    guide_path: str,
    adr_path: str,
    anchors_csv: str,
    out_wav: str,
    cfg: RenderConfig = RenderConfig(),
) -> None:
    guide = ensure_mono_48k(guide_path, target_sr=cfg.sr)
    adr = ensure_mono_48k(adr_path, target_sr=cfg.sr)
    anchors = read_anchors_csv(anchors_csv)

    y = warp_adr_to_guide(
        adr_samples=adr.samples,
        guide_len_samples=guide.samples.shape[0],
        anchors=anchors,
        cfg=cfg,
    )

    write_audio(out_wav, Audio(samples=y, sr=cfg.sr), subtype="PCM_16")


# ----------------------------
# CLI smoke test
# ----------------------------

"""
python3 -m core.render \
  --guide playground/mono48k.wav \
  --adr playground/mono48kadr.wav \
  --anchors outputs/test1/anchors.csv \
  --out outputs/test1/adr_conformed.wav
  --fade_ms 40
"""
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render conformed ADR from anchors.csv (WSOLA)")
    p.add_argument("--guide", required=True, help="Guide wav/flac (timeline reference)")
    p.add_argument("--adr", required=True, help="ADR wav/flac (to be conformed)")
    p.add_argument("--anchors", required=True, help="anchors.csv (t_adr_sec,t_guide_sec)")
    p.add_argument("--out", required=True, help="Output WAV path (conformed ADR)")
    p.add_argument("--fade_ms", type=float, default=20.0, help="Fade at segment boundaries (ms)")
    args = p.parse_args()

    cfg = RenderConfig(fade_ms=args.fade_ms)
    render_conformed_wav(args.guide, args.adr, args.anchors, args.out, cfg)
    print(f"[render] wrote: {args.out}")
