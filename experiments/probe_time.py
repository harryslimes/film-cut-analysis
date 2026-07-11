"""Inspect TransNetV2's raw prediction curve around a target time, and bisect the
threshold needed to detect a cut there.

  python probe_time.py --video data/movies/his_girl_friday_clean.mp4 --at 64 --window 4
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import torch

from detectors.base import ffprobe_info
from detectors.transnet import _model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--window", type=float, default=4.0)
    args = ap.parse_args()

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

    lo = int((args.at - args.window) * fps)
    hi = int((args.at + args.window) * fps)
    lo, hi = max(0, lo), min(len(single), hi)

    print(f"fps={fps:.3f}  probing {args.at}s ± {args.window}s  (frames {lo}-{hi})\n")
    print(f"{'frame':>6} {'time':>7} {'single':>7} {'all':>7}")
    peak_s = (0.0, -1, 0.0)
    for i in range(lo, hi):
        s, a = float(single[i]), float(allf[i])
        mark = "  <== single peak" if s > peak_s[0] else ""
        if s > peak_s[0]:
            peak_s = (s, i, i / fps)
        bar = "#" * int(s * 40)
        print(f"{i:6d} {i/fps:7.2f} {s:7.3f} {a:7.3f} {bar}")

    ps, pf, pt = peak_s
    print(f"\nPEAK single-frame prediction near {args.at}s: {ps:.3f} at {pt:.2f}s (frame {pf})")
    print(f"peak all-frame prediction in window:      {allf[lo:hi].max():.3f}")

    # bisection: smallest 0.05-step threshold that still fires here
    fired = None
    for thr in [round(t, 2) for t in np.arange(0.5, 0.04, -0.05)]:
        scenes = model.predictions_to_scenes(single, threshold=thr)
        starts = [int(s[0]) / fps for s in scenes[1:]]
        if any(abs(t - args.at) <= args.window for t in starts):
            fired = thr
            hit = min((t for t in starts if abs(t - args.at) <= args.window),
                      key=lambda t: abs(t - args.at))
            print(f"\nBISECT: a cut first appears at threshold <= {thr:.2f} "
                  f"(detected cut at {hit:.2f}s)")
            break
    if fired is None:
        print(f"\nBISECT: no cut here even at threshold 0.05 -> peak {ps:.3f} is too weak; "
              f"this is a very slow dissolve TransNet barely responds to.")


if __name__ == "__main__":
    main()
