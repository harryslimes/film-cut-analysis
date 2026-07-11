"""Tune the dissolve-peak height (gradual_height) against ground truth.

Sharp threshold is fixed; we sweep the all-frames peak height that promotes a dissolve
to a cut, and score the hybrid (sharp + dissolves) against real labels.

  python tune_gradual.py --video data/movies/charade_1963.mp4 --gt data/movies/charade_1963.gt.json --sharp 0.4
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import torch

from detectors.base import ffprobe_info
from detectors.transnet import _model, _gradual_peaks
from evaluate import match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sharp", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--min-gap-s", type=float, default=0.4)
    args = ap.parse_args()

    gt = json.load(open(args.gt))["cuts"]
    fps, n_frames, W, H = ffprobe_info(args.video)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", args.video,
           "-vf", "scale_cuda=48:27,hwdownload,format=nv12", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
    model = _model()
    fr = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
    with torch.no_grad():
        single, allf = model.predict_frames(fr, quiet=True)
    single = single.cpu().numpy().reshape(-1)
    allf = allf.cpu().numpy().reshape(-1)

    scenes = model.predictions_to_scenes(single, threshold=args.sharp)
    sharp = [int(s[0]) for s in scenes[1:]] if len(scenes) > 1 else []
    gap = max(1, int(args.min_gap_s * fps))

    print(f"gt={len(gt)}  sharp@{args.sharp}={len(sharp)} cuts  fps={fps:.3f}  tol=±{args.tol}s\n")
    print(f"{'g_ht':>5} {'+diss':>6} {'cuts':>5} {'P':>6} {'R':>6} {'F1':>6}")
    # baseline: sharp only
    m0 = match(sorted(round(f/fps,4) for f in sharp), gt, tol=args.tol)
    print(f"{'off':>5} {0:6d} {len(sharp):5d} {m0['precision']:6.3f} {m0['recall']:6.3f} {m0['f1']:6.3f}")
    for gh in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15):
        cuts_f = list(sharp)
        n_add = 0
        for pf in _gradual_peaks(allf, fps, gh, args.min_gap_s):
            if all(abs(pf - s) > gap for s in sharp):
                cuts_f.append(pf); n_add += 1
        cuts = sorted(round(f/fps,4) for f in cuts_f)
        m = match(cuts, gt, tol=args.tol)
        print(f"{gh:5.2f} {n_add:6d} {len(cuts):5d} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")


if __name__ == "__main__":
    main()
