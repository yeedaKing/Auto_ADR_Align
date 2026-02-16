# core/segment.py
"""
Phrase segmentation for Auto-ADR Align.

Goal:
- Split audio into phrase-like regions separated by silence.
- Use a simple energy envelope + hangover logic (robust enough for MVP).

Outputs:
- List of segments as (start_sec, end_sec)

Design notes:
- Operates on mono float32 audio in [-1, 1].
- You should pass audio already resampled to 48 kHz mono (use core.io_utils.ensure_mono_48k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class SegmenterConfig:
    frame_ms: float = 20.0          # frame size for energy computation
    hop_ms: float = 10.0            # hop between frames
    min_silence_s: float = 0.35     # silence duration that triggers a split
    min_segment_s: float = 0.25     # discard segments shorter than this
    pad_s: float = 0.02             # pad segment edges (helps preserve consonants)
    # Thresholding
    rel_db_above_noise: float = 20.0  # speech threshold relative to estimated noise floor (dB)
    abs_db_floor: float = -55.0       # do not set threshold below this absolute dBFS
    # Smoothing / hangover
    smooth_frames: int = 5            # moving average over energy frames
    hangover_frames: int = 3          # keep "speech" true for this many frames after dip
    fallback_to_full: bool = False


def _rms_db(x: np.ndarray, eps: float = 1e-10) -> float:
    rms = float(np.sqrt(np.mean(x * x) + eps))
    return 20.0 * np.log10(max(rms, eps))


def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    # 'same' keeps length; edges are slightly biased but fine for segmentation
    return np.convolve(y, kernel, mode="same")


def _frames_from_audio(x: np.ndarray, sr: int, frame_len: int, hop_len: int) -> np.ndarray:
    """
    Returns shape (n_frames, frame_len) using zero-padding at end if needed.
    """
    n = x.shape[0]
    if n < frame_len:
        pad = frame_len - n
        x = np.pad(x, (0, pad), mode="constant")
        n = x.shape[0]

    n_frames = 1 + (n - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    )
    return frames


def segment_phrases(
    samples: np.ndarray,
    sr: int,
    cfg: SegmenterConfig = SegmenterConfig(),
) -> List[Tuple[float, float]]:
    """
    Segment mono audio into phrase regions.

    Args:
        samples: mono float array shape (n,)
        sr: sample rate
        cfg: segmentation configuration

    Returns:
        List of (start_sec, end_sec) segments in seconds.
    """
    if samples.ndim != 1:
        raise ValueError("segment_phrases expects mono samples with shape (n,)")

    x = samples.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    print("peak:", peak)
    assert -1 <= peak <= 1

    frame_len = max(1, int(round(cfg.frame_ms * 1e-3 * sr)))
    hop_len = max(1, int(round(cfg.hop_ms * 1e-3 * sr)))

    frames = _frames_from_audio(x, sr, frame_len, hop_len)
    # frame-wise RMS in dBFS
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10)).astype(np.float32)

    # Smooth envelope
    rms_db_s = _moving_average(rms_db, cfg.smooth_frames)

    # Estimate noise floor as lower percentile of smoothed RMS
    noise_floor_db = float(np.percentile(rms_db_s, 20))
    # Speech threshold: relative to noise floor but never below abs_db_floor
    thr_db = max(cfg.abs_db_floor, noise_floor_db + cfg.rel_db_above_noise)

    print("noise_floor_db", noise_floor_db, "thr_db", thr_db,
      "rms_db_s min/med/max", float(rms_db_s.min()), float(np.median(rms_db_s)), float(rms_db_s.max()))

    speech = rms_db_s >= thr_db
    max_sil = 0
    cur = 0
    for v in speech:
        if not v:
            cur += 1
            max_sil = max(max_sil, cur)
        else:
            cur = 0
    print("max_silence_run_frames", max_sil, "=", max_sil * hop_len / sr, "sec")
    print("speech_true_frames", int(np.sum(speech)), "of", len(speech))

    # Hangover (safe): extend speech forward by N frames without propagating across long silence
    if cfg.hangover_frames > 0:
        hang = int(cfg.hangover_frames)
        speech2 = speech.copy()
        for k in range(1, hang + 1):
            speech2[k:] |= speech[:-k]   # if any of the previous k frames were speech, keep this on
        speech = speech2

    # Convert speech mask into segments with silence gaps >= min_silence_s
    min_sil_frames = max(1, int(round(cfg.min_silence_s / (hop_len / sr))))
    min_seg_frames = max(1, int(round(cfg.min_segment_s / (hop_len / sr))))

    segments_frames: List[Tuple[int, int]] = []
    nF = len(speech)

    min_sil_frames = max(1, int(round(cfg.min_silence_s / (hop_len / sr))))
    min_seg_frames = max(1, int(round(cfg.min_segment_s / (hop_len / sr))))

    i = 0
    while i < nF:
        # Find next speech start
        while i < nF and not speech[i]:
            i += 1
        if i >= nF:
            break
        start = i

        # Walk forward until we observe a silence run >= min_sil_frames
        sil = 0
        i += 1
        while i < nF:
            if speech[i]:
                sil = 0
            else:
                sil += 1
                if sil >= min_sil_frames:
                    break
            i += 1

        if i >= nF:
            end = nF
        else:
            # i is inside the silence run; compute end as the first silent frame index (exclusive)
            end = i - sil + 1

            # Skip the rest of the silence run so next segment begins cleanly
            while i < nF and not speech[i]:
                i += 1

        if (end - start) >= min_seg_frames:
            segments_frames.append((start, end))

    print("segments_frames:", segments_frames[:10], "count", len(segments_frames))
    
    # Convert frame segments to seconds, apply padding, clamp to [0, duration]
    dur_s = x.shape[0] / float(sr)
    hop_s = hop_len / float(sr)

    out: List[Tuple[float, float]] = []
    for sF, eF in segments_frames:
        s = sF * hop_s - cfg.pad_s
        e = eF * hop_s + cfg.pad_s
        s = max(0.0, s)
        e = min(dur_s, e)
        if e - s >= cfg.min_segment_s:
            out.append((s, e))

    # If nothing found, fall back to whole clip (useful for very clean/short lines)
    if not out and dur_s > 0 and cfg.fallback_to_full:
        out = [(0.0, dur_s)]

    return out


def segments_to_sample_ranges(
    segments_s: List[Tuple[float, float]],
    sr: int,
) -> List[Tuple[int, int]]:
    """Convert (start_sec, end_sec) to (start_sample, end_sample)."""
    ranges = []
    for s, e in segments_s:
        a = int(round(s * sr))
        b = int(round(e * sr))
        a = max(0, a)
        b = max(a, b)
        ranges.append((a, b))
    return ranges


def _format_segments(segments: List[Tuple[float, float]]) -> str:
    lines = []
    for i, (s, e) in enumerate(segments, 1):
        lines.append(f"{i:02d}: {s:7.3f}s → {e:7.3f}s   ({(e-s):.3f}s)")
    return "\n".join(lines)


# python3 -m core.segment --in playground/mono48k.flac --rel_db 10 --min_silence 0.3
if __name__ == "__main__":
    import argparse
    from core.io_utils import ensure_mono_48k

    parser = argparse.ArgumentParser(description="Phrase segmentation smoke test")
    parser.add_argument("--in", dest="in_path", required=True, help="Input audio file")
    parser.add_argument("--min_silence", type=float, default=0.35, help="Min silence to split (sec)")
    parser.add_argument("--rel_db", type=float, default=20.0, help="dB above noise floor threshold")
    args = parser.parse_args()

    audio = ensure_mono_48k(args.in_path)
    cfg = SegmenterConfig(min_silence_s=args.min_silence, rel_db_above_noise=args.rel_db)
    segs = segment_phrases(audio.samples, audio.sr, cfg)

    print(f"[segment] File: {args.in_path}")
    print(f"[segment] Duration: {audio.duration:.3f}s | Segments: {len(segs)}")
    print(_format_segments(segs))
