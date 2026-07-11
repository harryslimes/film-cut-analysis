"""Tune TransNetV2 for dissolve recall against real ground truth.

Decodes a film ONCE to 48x27, runs the net, then sweeps the detection threshold on
both prediction streams (single-frame = sharp cuts, all-frames = gradual transitions)
and scores each against a ground-truth cut list. Prints the precision/recall/F1 curve
so we can pick a threshold that recovers dissolves without wrecking precision.

  python tune_transnet.py --video data/movies/charade_1963.mp4 --gt data/movies/charade_1963.gt.json
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import torch

from detectors.base import ffprobe_info
from detectors.transnet import _model
from evaluate import match


def decode_predict(video_path):
    fps, n_frames, W, H = ffprobe_info(video_path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", video_path,
           "-vf", "scale_cuda=48:27,hwdownload,format=nv12", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
    model = _model()
    fr = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
    with torch.no_grad():
        single, allf = model.predict_frames(fr, quiet=True)
    return fps, single.cpu().numpy().reshape(-1), allf.cpu().numpy().reshape(-1)


def cuts_from(pred, fps, threshold):
    from detectors.transnet import _model
    scenes = _model().predictions_to_scenes(pred, threshold=threshold)
    return sorted(round(int(s[0]) / fps, 4) for s in scenes[1:]) if len(scenes) > 1 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    gt = json.load(open(args.gt))["cuts"]
    fps, single, allf = decode_predict(args.video)
    print(f"gt={len(gt)} cuts  frames={len(single)}  fps={fps:.3f}  tol=±{args.tol}s\n")

    for name, pred in [("single", single), ("all", allf)]:
        print(f"=== {name}-frame predictions ===")
        print(f"{'thr':>5} {'cuts':>5} {'P':>6} {'R':>6} {'F1':>6}")
        for thr in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
            cuts = cuts_from(pred, fps, thr)
            m = match(cuts, gt, tol=args.tol)
            print(f"{thr:5.2f} {len(cuts):5d} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
        print()


if __name__ == "__main__":
    main()
