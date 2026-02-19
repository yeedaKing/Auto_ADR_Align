# core/features.py
"""
Feature extraction for Auto-ADR Align.

Implements MFCC (+ optional delta, delta-delta) using only numpy + scipy.

Pipeline:
1) Frame audio
2) Window (Hann)
3) Power spectrum via rFFT
4) Mel filterbank energies
5) Log-mel
6) DCT-II -> MFCC
7) Optional deltas + CMVN

Output:
- FeatureBatch(feat=(T,D), hop_length, win_length, sr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import rfft, dct  # scipy>=1.4 usually
# If your SciPy is old and lacks scipy.fft, replace with:
# from scipy.fftpack import dct
# from numpy.fft import rfft


@dataclass(frozen=True)
class FeatureConfig:
    n_mfcc: int = 20
    n_mels: int = 40
    fmin: float = 50.0
    fmax: Optional[float] = None      # if None, uses sr/2
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    include_deltas: bool = True
    include_delta2: bool = True
    cmvn: bool = True
    # MFCC details
    preemph: float = 0.97
    power: float = 2.0                # 1.0 for magnitude, 2.0 for power
    eps: float = 1e-10
    delta_N: int = 2                  # regression window for delta features


@dataclass(frozen=True)
class FeatureBatch:
    feat: np.ndarray   # (T, D)
    sr: int
    hop_length: int
    win_length: int
    n_fft: int

    @property
    def hop_s(self) -> float:
        return self.hop_length / float(self.sr)

    @property
    def frame_s(self) -> float:
        return self.win_length / float(self.sr)

    @property
    def n_frames(self) -> int:
        return int(self.feat.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.feat.shape[1])

    def frame_to_time(self, frame_idx: int) -> float:
        return frame_idx * self.hop_s

    def time_to_frame(self, t_s: float) -> int:
        return int(round(t_s / self.hop_s))


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _preemphasis(x: np.ndarray, a: float) -> np.ndarray:
    if a <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def _frame_audio(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    Returns frames with shape (T, frame_len).
    Pads at end with zeros so we include the tail.
    """
    if x.ndim != 1:
        raise ValueError("Expected mono audio (n,)")

    n = x.shape[0]
    if n < frame_len:
        x = np.pad(x, (0, frame_len - n), mode="constant")
        n = x.shape[0]

    # Number of frames so that last frame starts before end (with padding)
    n_frames = 1 + int(np.ceil((n - frame_len) / hop_len))
    total_len = (n_frames - 1) * hop_len + frame_len
    if total_len > n:
        x = np.pad(x, (0, total_len - n), mode="constant")

    # Stride trick
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    )
    return frames


def _hz_to_mel(f_hz: np.ndarray) -> np.ndarray:
    # HTK mel scale
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def _mel_to_hz(m_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m_mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Returns mel filterbank matrix of shape (n_mels, n_bins),
    where n_bins = n_fft//2 + 1 (rFFT bins).
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)

    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = _mel_to_hz(m_points)

    # Convert Hz to FFT bin indices
    bin_points = np.floor((n_fft + 1) * f_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        if center == left:
            center = min(left + 1, n_bins - 1)
        if right == center:
            right = min(center + 1, n_bins - 1)

        # Rising slope
        for k in range(left, center):
            fb[m, k] = (k - left) / float(center - left)
        # Falling slope
        for k in range(center, right):
            fb[m, k] = (right - k) / float(right - center)

    # Normalize filters (optional; common in MFCC pipelines)
    # Prevents scale differences across mel bands
    enorm = 2.0 / (f_points[2:n_mels + 2] - f_points[:n_mels])
    fb *= enorm[:, None].astype(np.float32)

    return fb


def _cmvn(feat: np.ndarray, eps: float) -> np.ndarray:
    """
    CMVN across time per dimension.
    Input feat: (T, D)
    """
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True)
    return (feat - mu) / np.maximum(sd, eps)


def _delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Regression deltas along time.
    feat: (T, D)
    """
    if N <= 0:
        return np.zeros_like(feat)
    T, D = feat.shape
    denom = 2.0 * sum(n * n for n in range(1, N + 1))
    # Pad by edge values
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    out = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        num = 0.0
        for n in range(1, N + 1):
            num += n * (padded[t + N + n] - padded[t + N - n])
        out[t] = num / denom
    return out


def extract_mfcc(samples: np.ndarray, sr: int, cfg: FeatureConfig = FeatureConfig()) -> FeatureBatch:
    """
    Compute MFCC features (plus optional deltas) for DTW alignment.
    Returns FeatureBatch with feat shape (T, D).
    """
    if samples.ndim != 1:
        raise ValueError("extract_mfcc expects mono samples (n,)")

    x = samples.astype(np.float32, copy=False)
    if x.size == 0:
        raise ValueError("Empty audio")

    # Pre-emphasis (helps speech MFCC stability)
    x = _preemphasis(x, cfg.preemph)

    win_length = max(1, int(round(cfg.frame_ms * 1e-3 * sr)))
    hop_length = max(1, int(round(cfg.hop_ms * 1e-3 * sr)))
    n_fft = _next_pow2(win_length)

    frames = _frame_audio(x, win_length, hop_length)  # (T, win_length)
    window = np.hanning(win_length).astype(np.float32)
    frames_w = frames * window[None, :]

    # rFFT -> power spectrum
    spec = rfft(frames_w, n=n_fft, axis=1)  # (T, n_bins)
    mag = np.abs(spec).astype(np.float32)
    if cfg.power == 1.0:
        powspec = mag
    else:
        powspec = mag ** cfg.power

    # Mel filterbank
    fmax = cfg.fmax if cfg.fmax is not None else sr / 2.0
    fb = _mel_filterbank(sr, n_fft, cfg.n_mels, cfg.fmin, fmax)  # (n_mels, n_bins)

    melE = (powspec @ fb.T).astype(np.float32)  # (T, n_mels)
    melE = np.maximum(melE, cfg.eps)
    log_mel = np.log(melE)  # natural log is fine for DTW

    # DCT-II along mel axis -> MFCC
    # dct(x, type=2, norm='ortho') yields common MFCC scaling
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, : cfg.n_mfcc].astype(np.float32)  # (T, n_mfcc)

    feats = [mfcc]
    if cfg.include_deltas:
        feats.append(_delta(mfcc, N=cfg.delta_N))
    if cfg.include_delta2:
        feats.append(_delta(_delta(mfcc, N=cfg.delta_N), N=cfg.delta_N))

    F = np.concatenate(feats, axis=1).astype(np.float32, copy=False)  # (T, D)

    if cfg.cmvn:
        F = _cmvn(F, eps=cfg.eps).astype(np.float32, copy=False)

    return FeatureBatch(feat=F, sr=sr, hop_length=hop_length, win_length=win_length, n_fft=n_fft)


def cosine_normalize(feat_TD: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize each time frame to unit norm (good before cosine distance DTW).
    Input: (T, D)
    """
    norms = np.linalg.norm(feat_TD, axis=1, keepdims=True)
    return feat_TD / np.maximum(norms, eps)


# ----------------------------
# Smoke test
# ----------------------------

def _smoke_test(in_path: str) -> None:
    from core.io_utils import ensure_mono_48k

    audio = ensure_mono_48k(in_path)
    fb = extract_mfcc(audio.samples, audio.sr)

    print(f"[features] file: {in_path}")
    print(f"[features] duration: {audio.duration:.3f}s | sr: {audio.sr}")
    print(f"[features] feat shape: {fb.feat.shape} (T, D)")
    print(f"[features] hop: {fb.hop_s*1000:.2f} ms | frame: {fb.frame_s*1000:.2f} ms | n_fft: {fb.n_fft}")
    approx = fb.n_frames * fb.hop_s
    print(f"[features] approx span: {approx:.3f}s (T*hop)")
    print(f"[features] mean/std: {fb.feat.mean():.4f} / {fb.feat.std():.4f}")

# python3 -m core.features --in playground/mono48k.flac
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFCC feature extraction smoke test (no librosa)")
    parser.add_argument("--in", dest="in_path", required=True, help="Input audio file")
    args = parser.parse_args()

    _smoke_test(args.in_path)
