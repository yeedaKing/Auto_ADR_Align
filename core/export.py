# core/export.py
"""
CSV export helpers for Auto-ADR Align.

Primary outputs (MVP):
- anchors.csv:  time map as (t_adr_sec, t_guide_sec) anchors
- path.csv:     DTW path as (i_adr, i_guide, t_adr_sec, t_guide_sec) for debugging
- segments.csv: segments as (idx, start_sec, end_sec, dur_sec) (optional)
- stats.csv:    key/value stats (optional)

All writers are "atomic": write to temp file then replace destination.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _atomic_write_text(path: str, text: str) -> None:
    _ensure_parent_dir(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _atomic_write_csv(path: str, rows: Iterable[Sequence[object]]) -> None:
    _ensure_parent_dir(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def write_anchors_csv(
    anchors: np.ndarray,
    out_path: str,
    precision: int = 6,
    include_header: bool = True,
) -> None:
    """
    anchors: (K,2) array with columns [t_adr_sec, t_guide_sec]
    """
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must have shape (K, 2)")

    fmt = f"{{:.{precision}f}}"
    rows: List[List[object]] = []
    if include_header:
        rows.append(["t_adr_sec", "t_guide_sec"])

    for t_a, t_g in anchors:
        rows.append([fmt.format(float(t_a)), fmt.format(float(t_g))])

    _atomic_write_csv(out_path, rows)


def write_dtw_path_csv(
    path_pairs: List[Tuple[int, int]],
    hop_adr_s: float,
    hop_guide_s: float,
    out_path: str,
    precision: int = 6,
    include_header: bool = True,
) -> None:
    """
    path_pairs: list[(i_adr, i_guide)]
    """
    fmt = f"{{:.{precision}f}}"
    rows: List[List[object]] = []
    if include_header:
        rows.append(["i_adr", "i_guide", "t_adr_sec", "t_guide_sec"])

    for i_a, i_g in path_pairs:
        t_a = i_a * hop_adr_s
        t_g = i_g * hop_guide_s
        rows.append([int(i_a), int(i_g), fmt.format(float(t_a)), fmt.format(float(t_g))])

    _atomic_write_csv(out_path, rows)


def write_segments_csv(
    segments_s: List[Tuple[float, float]],
    out_path: str,
    precision: int = 6,
    include_header: bool = True,
) -> None:
    fmt = f"{{:.{precision}f}}"
    rows: List[List[object]] = []
    if include_header:
        rows.append(["idx", "start_sec", "end_sec", "dur_sec"])

    for idx, (s, e) in enumerate(segments_s, 1):
        rows.append([idx, fmt.format(float(s)), fmt.format(float(e)), fmt.format(float(e - s))])

    _atomic_write_csv(out_path, rows)


def write_stats_csv(stats: Dict[str, float], out_path: str, precision: int = 6) -> None:
    fmt = f"{{:.{precision}f}}"
    rows: List[List[object]] = [["key", "value"]]
    for k in sorted(stats.keys()):
        v = stats[k]
        if isinstance(v, (int, np.integer)):
            rows.append([k, int(v)])
        else:
            rows.append([k, fmt.format(float(v))])
    _atomic_write_csv(out_path, rows)


# ----------------------------
# Convenience CLI
# ----------------------------

def _cli_compute_and_export(guide_path: str, adr_path: str, out_dir: str) -> None:
    """
    Convenience command:
    - loads audio
    - extracts MFCC
    - runs DTW
    - writes anchors.csv, path.csv, stats.csv
    """
    from core.io_utils import ensure_mono_48k
    from core.features import extract_mfcc, FeatureConfig
    from core.dtw_map import align_feature_batches, DTWConfig

    g = ensure_mono_48k(guide_path)
    a = ensure_mono_48k(adr_path)

    fb_g = extract_mfcc(g.samples, g.sr, FeatureConfig())
    fb_a = extract_mfcc(a.samples, a.sr, FeatureConfig())

    res = align_feature_batches(fb_a, fb_g, DTWConfig())

    os.makedirs(out_dir, exist_ok=True)
    anchors_path = os.path.join(out_dir, "anchors.csv")
    path_path = os.path.join(out_dir, "path.csv")
    stats_path = os.path.join(out_dir, "stats.csv")

    write_anchors_csv(res.anchors, anchors_path)
    write_dtw_path_csv(res.path, fb_a.hop_s, fb_g.hop_s, path_path)
    write_stats_csv(res.stats, stats_path)

    print(f"[export] wrote: {anchors_path}")
    print(f"[export] wrote: {path_path}")
    print(f"[export] wrote: {stats_path}")


# python3 -m core.export --guide playground/mono48k.wav --adr playground/mono48kadr.wav --out_dir outputs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSV export helpers (anchors/path/stats)")
    parser.add_argument("--guide", help="Guide wav/flac (for convenience DTW+export)")
    parser.add_argument("--adr", help="ADR wav/flac (for convenience DTW+export)")
    parser.add_argument("--out_dir", default="outputs", help="Output directory for CSV files")
    args = parser.parse_args()

    if args.guide and args.adr:
        _cli_compute_and_export(args.guide, args.adr, args.out_dir)
    else:
        print("[export] No --guide/--adr provided.  This module mainly provides writer functions.")
