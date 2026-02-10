# core/io_utils.py
"""
Audio I/O utilities for Auto-ADR Align.

Responsibilities
- Load/save WAV/FLAC with consistent dtype.
- Convert to mono and resample to a target sample rate (default 48 kHz).
- Basic helpers: safe mkdir, hashing, duration, chunking.

Dependencies
- soundfile (pysoundfile)
- numpy
- scipy (signal.resample_poly)
- tqdm (optional; only used in __main__ smoke test)
"""

from __future__ import annotations

import io
import os
import math
import hashlib
import contextlib
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 48_000
DEFAULT_DTYPE = np.float32


# ----------------------------
# Filesystem helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ----------------------------
# Audio core
# ----------------------------

@dataclass
class Audio:
    """Simple audio container."""
    samples: np.ndarray  # shape: (n,) mono or (n, ch)
    sr: int

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sr)


def _to_float32(x: np.ndarray) -> np.ndarray:
    """Cast to float32, scaling ints to [-1, 1] if needed."""
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(DEFAULT_DTYPE, copy=False)
    # Integer PCM → float in [-1, 1]
    info = np.iinfo(x.dtype)
    return (x.astype(np.float64) / max(1, info.max)).astype(DEFAULT_DTYPE)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    """Float [-1, 1] to int16 PCM with clipping."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def read_audio(path: str, dtype: np.dtype = DEFAULT_DTYPE) -> Audio:
    """
    Read audio from disk using soundfile.
    Returns float32 (default) array in shape (n,) or (n, ch) and sample rate.
    """
    data, sr = sf.read(path, always_2d=False)
    data = _to_float32(np.asarray(data))
    if dtype != DEFAULT_DTYPE:
        data = data.astype(dtype)
    return Audio(samples=data, sr=sr)


def write_audio(path: str, audio: Audio, subtype: Optional[str] = None) -> None:
    """
    Write audio to disk using soundfile.
    - If subtype is None, choose sensible default by extension.
    - Supports float WAV/FLAC.  For int16 WAV, set subtype="PCM_16".
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    ext = os.path.splitext(path)[1].lower()

    # Choose default subtype
    if subtype is None:
        if ext in [".wav", ".aif", ".aiff"]:
            subtype = "PCM_16"  # widely compatible; change to "FLOAT" if preferred
        elif ext == ".flac":
            subtype = None  # FLAC ignores subtype; uses lossless compression
        else:
            subtype = "PCM_16"

    # Convert dtype for PCM_16
    data = audio.samples
    if subtype == "PCM_16":
        if data.ndim == 1:
            out = _to_pcm16(data)
        else:
            out = np.stack([_to_pcm16(data[:, ch]) for ch in range(data.shape[1])], axis=1)
    else:
        out = data.astype(np.float32, copy=False)

    sf.write(path, out, audio.sr, subtype=subtype)


def to_mono(audio: Audio, method: str = "mean") -> Audio:
    """
    Convert to mono.
    - method="mean": average channels (default).
    - method="left"/"right": pick a channel.
    """
    x = audio.samples
    if x.ndim == 1 or x.shape[1] == 1:
        return audio  # already mono

    if method == "mean":
        mono = np.mean(x, axis=1)
    elif method == "left":
        mono = x[:, 0]
    elif method == "right":
        mono = x[:, -1]
    else:
        raise ValueError(f"Unknown mono method: {method}")
    return Audio(samples=mono.astype(DEFAULT_DTYPE, copy=False), sr=audio.sr)


def resample_to_sr(audio: Audio, target_sr: int = TARGET_SR) -> Audio:
    """
    High-quality resample using polyphase filtering.
    """
    if audio.sr == target_sr:
        return audio

    # Use integer up/down factors for resample_poly
    from math import gcd
    g = gcd(audio.sr, target_sr)
    up = target_sr // g
    down = audio.sr // g

    x = audio.samples
    if x.ndim == 1:
        y = resample_poly(x, up, down).astype(DEFAULT_DTYPE, copy=False)
    else:
        # Channel-wise resample
        y_ch = []
        for ch in range(x.shape[1]):
            y_ch.append(resample_poly(x[:, ch], up, down))
        y = np.stack(y_ch, axis=1).astype(DEFAULT_DTYPE, copy=False)

    return Audio(samples=y, sr=target_sr)


def ensure_mono_48k(path: str,
                    mono_method: str = "mean",
                    target_sr: int = TARGET_SR) -> Audio:
    """
    Load an audio file, convert to mono, and resample to 48 kHz.
    """
    a = read_audio(path)
    a = to_mono(a, method=mono_method)
    a = resample_to_sr(a, target_sr=target_sr)
    return a


# ----------------------------
# Convenience helpers
# ----------------------------

def seconds_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def samples_to_seconds(samples: int, sr: int) -> float:
    return float(samples) / float(sr)


def frame_count(n_samples: int, hop_length: int) -> int:
    """Number of frames for a given hop, using librosa's convention (center=False)."""
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")
    return max(0, 1 + (n_samples - 1) // hop_length)


def iter_chunks(audio: Audio, chunk_seconds: float) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Iterate over fixed-size chunks of audio (mono or multi-channel).
    Yields: (start_sample, end_sample, array_view)
    """
    step = seconds_to_samples(chunk_seconds, audio.sr)
    n = audio.n_samples
    start = 0
    while start < n:
        end = min(n, start + step)
        yield start, end, audio.samples[start:end]
        start = end


@contextlib.contextmanager
def temp_wav_buffer(sr: int) -> Iterator[Tuple[io.BytesIO, int]]:
    """
    Context manager that returns an in-memory WAV BytesIO and the sample rate.
    Useful for writing small snippets without touching disk.
    """
    buf = io.BytesIO()
    try:
        yield buf, sr
    finally:
        buf.close()


# ----------------------------
# Smoke test
# ----------------------------

def _smoke_test(in_path: str, out_dir: str) -> None:
    """
    Minimal CLI smoke test:
    - Load in_path, convert to mono 48k
    - Save WAV and FLAC
    - Print duration and SHA hashes
    """
    ensure_dir(out_dir)
    a = ensure_mono_48k(in_path)
    wav_out = os.path.join(out_dir, "mono48k.wav")
    flac_out = os.path.join(out_dir, "mono48k.flac")

    write_audio(wav_out, a, subtype="PCM_16")
    write_audio(flac_out, a)  # FLAC default

    print(f"[io_utils] Loaded: {in_path}")
    print(f"[io_utils] Duration: {a.duration:.3f} s  |  SR: {a.sr}  |  Ch: {a.n_channels}")
    print(f"[io_utils] Wrote: {wav_out} (sha256: {file_sha256(wav_out)[:16]}...)")
    print(f"[io_utils] Wrote: {flac_out} (sha256: {file_sha256(flac_out)[:16]}...)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="io_utils smoke test")
    parser.add_argument("--in", dest="in_path", required=True, help="Input audio file")
    parser.add_argument("--out", dest="out_dir", default="./runs/io_smoke", help="Output directory")
    args = parser.parse_args()

    _smoke_test(args.in_path, args.out_dir)
