# bin/adr_align.py
"""
End-to-end Auto-ADR Align CLI.

What this does (MVP):
- Loads a "guide" dialogue clip (production audio) and an "adr" clip (replacement recording).
- Extracts MFCC-like features (core.features, no librosa).
- Runs banded DTW (core.dtw_map) to align ADR time -> guide time.
- Exports:
    - anchors.csv (piecewise-linear time map: t_adr_sec -> t_guide_sec)
    - path.csv    (full DTW path, per-frame mapping)
    - stats.csv   (summary numbers)

Optional:
- If --segment_guide is set, it will also segment the guide into phrases and write:
    - guide_segments.csv
    - guide_segments_mapped_to_adr.csv  (estimated ADR time ranges using inverse map)

Typical usage:
  python3 -m bin.adr_align --guide playground/mono48k.wav --adr playground/mono48kadr.wav --out outputs/run1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from core.io_utils import ensure_mono_48k
from core.features import extract_mfcc, FeatureConfig
from core.dtw_map import align_feature_batches, DTWConfig
from core.export import write_anchors_csv, write_dtw_path_csv, write_stats_csv
from core.render import render_conformed_wav, RenderConfig


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _basename_noext(p: str) -> str:
    return Path(p).stem


def _dedup_monotone_y_for_interp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    np.interp requires xp to be increasing.  For inverse mapping we want to interpolate x(y).
    If y has flat runs / duplicates, keep the max x for each y (monotone nondecreasing).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return x, y

    # Ensure nondecreasing
    y = np.maximum.accumulate(y)
    x = np.maximum.accumulate(x)

    # Dedup exact duplicates in y by keeping max x
    out_y = [y[0]]
    out_x = [x[0]]
    for i in range(1, len(y)):
        if abs(y[i] - out_y[-1]) <= 1e-12:
            out_x[-1] = max(out_x[-1], x[i])
        else:
            out_y.append(y[i])
            out_x.append(x[i])
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def _interp_inverse_from_anchors(anchors: np.ndarray):
    """
    Returns two callables:
      fwd(t_adr)   -> t_guide  (interp on anchors[:,0] -> anchors[:,1])
      inv(t_guide) -> t_adr    (interp on anchors[:,1] -> anchors[:,0])
    """
    a = np.asarray(anchors, dtype=np.float64)
    tA = np.maximum.accumulate(a[:, 0])
    tG = np.maximum.accumulate(a[:, 1])

    # forward: guide as function of adr
    def fwd(t_adr: np.ndarray) -> np.ndarray:
        return np.interp(t_adr, tA, tG, left=tG[0], right=tG[-1])

    # inverse: adr as function of guide (need strictly nondecreasing xp)
    x_for_inv, y_for_inv = _dedup_monotone_y_for_interp(tA, tG)  # returns (x=tA, y=tG) but deduped on y
    # Here y_for_inv is increasing-ish (deduped), x_for_inv is corresponding tA.

    def inv(t_guide: np.ndarray) -> np.ndarray:
        return np.interp(t_guide, y_for_inv, x_for_inv, left=x_for_inv[0], right=x_for_inv[-1])

    return fwd, inv


def _write_segments_csv(path: Path, segments: List[Tuple[float, float]]) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "start_sec", "end_sec", "dur_sec"])
        for i, (s, e) in enumerate(segments, 1):
            w.writerow([i, f"{s:.6f}", f"{e:.6f}", f"{(e - s):.6f}"])


def _write_mapped_segments_csv(
    path: Path,
    guide_segments: List[Tuple[float, float]],
    inv_map,  # callable: t_guide -> t_adr
) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "guide_start_sec", "guide_end_sec", "adr_start_est_sec", "adr_end_est_sec"])
        for i, (gs, ge) in enumerate(guide_segments, 1):
            adr_s = float(inv_map(np.asarray([gs], dtype=np.float64))[0])
            adr_e = float(inv_map(np.asarray([ge], dtype=np.float64))[0])
            # enforce sane ordering
            if adr_e < adr_s:
                adr_s, adr_e = adr_e, adr_s
            w.writerow([i, f"{gs:.6f}", f"{ge:.6f}", f"{adr_s:.6f}", f"{adr_e:.6f}"])


def run_align(
    guide_path: str,
    adr_path: str,
    out_dir: Path,
    feat_cfg: FeatureConfig,
    dtw_cfg: DTWConfig,
    write_path: bool = True,
    segment_guide: bool = False,
    seg_min_silence: float = 0.35,
    seg_rel_db: float = 20.0,
    render: bool = False,
    render_out: str | None = None,
    fade_ms: float = 20.0,
) -> None:
    _safe_mkdir(out_dir)

    guide = ensure_mono_48k(guide_path)
    adr = ensure_mono_48k(adr_path)

    fb_g = extract_mfcc(guide.samples, guide.sr, feat_cfg)
    fb_a = extract_mfcc(adr.samples, adr.sr, feat_cfg)

    res = align_feature_batches(fb_a, fb_g, dtw_cfg)

    anchors_csv = out_dir / "anchors.csv"
    stats_csv = out_dir / "stats.csv"
    path_csv = out_dir / "path.csv"

    write_anchors_csv(res.anchors, str(anchors_csv))
    write_stats_csv(res.stats, str(stats_csv))

    if write_path:
        write_dtw_path_csv(res.path, fb_a.hop_s, fb_g.hop_s, str(path_csv))


    # Optional phrase segmentation on guide + mapping to ADR time via inverse map
    if segment_guide:
        from core.segment import segment_phrases, SegmenterConfig
        seg_cfg = SegmenterConfig(min_silence_s=seg_min_silence, rel_db_above_noise=seg_rel_db)
        gsegs = segment_phrases(guide.samples, guide.sr, seg_cfg)
        _write_segments_csv(out_dir / "guide_segments.csv", gsegs)

        _, inv_map = _interp_inverse_from_anchors(res.anchors)
        _write_mapped_segments_csv(out_dir / "guide_segments_mapped_to_adr.csv", gsegs, inv_map)

    if render:
        out_wav = Path(render_out) if render_out else (out_dir / "adr_conformed.wav")
        rcfg = RenderConfig(fade_ms=fade_ms)
        render_conformed_wav(
            guide_path=guide_path,
            adr_path=adr_path,
            anchors_csv=str(anchors_csv),
            out_wav=str(out_wav),
            cfg=rcfg,
        )

    # Console summary
    print("[adr_align] guide:", guide_path)
    print("[adr_align] adr  :", adr_path)
    print(f"[adr_align] out  : {out_dir}")
    print(f"[adr_align] guide_dur={guide.duration:.3f}s | adr_dur={adr.duration:.3f}s")
    print("[adr_align] stats:", {k: round(v, 4) for k, v in res.stats.items()})
    print("[adr_align] wrote:", "anchors.csv,", "stats.csv" + (", path.csv" if write_path else ""))
    if segment_guide:
        print("[adr_align] wrote: guide_segments.csv, guide_segments_mapped_to_adr.csv")

    if render:
        print(f"[adr_align] wrote: {out_wav}")


"""
python3 -m bin.adr_align \
  --guide playground/mono48k.wav \
  --adr playground/mono48kadr.wav \
  --out outputs/test_render \
  --segment_guide \
  --render
"""
def main() -> None:
    p = argparse.ArgumentParser(description="Auto-ADR Align (DTW-based time-map exporter)")
    p.add_argument("--guide", required=True, help="Guide wav/flac (production)")
    p.add_argument("--adr", required=True, help="ADR wav/flac (replacement)")
    p.add_argument("--out", default=None, help="Output directory (default: outputs/<guide>__<adr>)")
    p.add_argument("--no_path", action="store_true", help="Do not write path.csv (can be large)")

    # Feature params (must match core.features defaults unless you want to experiment)
    p.add_argument("--frame_ms", type=float, default=25.0, help="Feature frame size (ms)")
    p.add_argument("--hop_ms", type=float, default=10.0, help="Feature hop size (ms)")
    p.add_argument("--n_mfcc", type=int, default=20, help="MFCC count")
    p.add_argument("--n_mels", type=int, default=40, help="Mel bands")

    # DTW params
    p.add_argument("--band", type=float, default=0.15, help="Sakoe-Chiba band fraction (0.05-0.30 typical)")
    p.add_argument("--step_penalty", type=float, default=1e-3, help="Penalty for horiz/vert steps")
    p.add_argument("--anchor_every", type=int, default=5, help="Take an anchor every N ADR frames")
    p.add_argument("--simplify_eps", type=float, default=0.020, help="Anchor simplification epsilon (sec)")
    p.add_argument("--slope_min", type=float, default=0.0, help="Min slope dy/dx on anchors (guide/adr)")
    p.add_argument("--slope_max", type=float, default=3.0, help="Max slope dy/dx on anchors (guide/adr)")

    # Optional guide segmentation export
    p.add_argument("--segment_guide", action="store_true", help="Also export guide phrase segments + mapped ADR ranges")
    p.add_argument("--seg_min_silence", type=float, default=0.35, help="Guide segmentation: min silence to split (sec)")
    p.add_argument("--seg_rel_db", type=float, default=20.0, help="Guide segmentation: threshold above noise floor (dB)")

    # Optional rendering
    p.add_argument("--render", action="store_true", help="Render adr_conformed.wav using anchors.csv")
    p.add_argument("--render_out", default=None, help="Output WAV path (default: <out>/adr_conformed.wav)")
    p.add_argument("--fade_ms", type=float, default=20.0, help="Render: boundary fade length in ms")


    args = p.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"{_basename_noext(args.guide)}__{_basename_noext(args.adr)}"
    feat_cfg = FeatureConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )
    dtw_cfg = DTWConfig(
        band_frac=args.band,
        step_penalty=args.step_penalty,
        anchor_every=args.anchor_every,
        simplify_eps_s=args.simplify_eps,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
    )

    run_align(
        guide_path=args.guide,
        adr_path=args.adr,
        out_dir=out_dir,
        feat_cfg=feat_cfg,
        dtw_cfg=dtw_cfg,
        write_path=(not args.no_path),
        segment_guide=args.segment_guide,
        seg_min_silence=args.seg_min_silence,
        seg_rel_db=args.seg_rel_db,
        render=args.render,
        render_out=args.render_out,
        fade_ms=args.fade_ms,
    )


if __name__ == "__main__":
    main()
