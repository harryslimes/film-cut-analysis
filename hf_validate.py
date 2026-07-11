"""Validate the TransNetV2 thresholds across a diverse labelled SET (not one film).

Streams labelled clips from HuggingFace it-just-works/shot-boundary-detection
(Cut / Transition / Empty), runs TransNetV2 on each, and records the peak single-frame
and all-frames predictions. Then sweeps thresholds to show, across hundreds of clips:
  - sharp threshold: recall on Cut clips vs false-fire rate on Empty clips
  - gradual height : recall on Transition clips vs false-fire on Empty clips

  python hf_validate.py --per-class 120
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile

import numpy as np
import torch

from detectors.transnet import _model


def clip_scores(mp4_bytes, model):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4_bytes); f.flush()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", f.name,
               "-vf", "scale=48:27", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(raw) < 27 * 48 * 3:
        return None
    frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
    fr = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
    with torch.no_grad():
        single, allf = model.predict_frames(fr, quiet=True)
    return float(single.max()), float(allf.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    args = ap.parse_args()
    from datasets import load_dataset

    model = _model()
    ds = load_dataset("it-just-works/shot-boundary-detection", split="test", streaming=True)
    want = {"C": args.per_class, "T": args.per_class, "E": args.per_class}
    got = {k: [] for k in want}          # label -> list of (max_single, max_all)
    for ex in ds:
        lab = ex["json"]["label"]
        if lab in want and len(got[lab]) < want[lab]:
            s = clip_scores(ex["mp4"], model)
            if s:
                got[lab].append(s)
        if all(len(got[k]) >= want[k] for k in want):
            break
    for k in got:
        print(f"  {k}: {len(got[k])} clips")

    C = np.array(got["C"]); T = np.array(got["T"]); E = np.array(got["E"])
    print("\n=== SHARP threshold: Cut recall vs Empty false-fire (single-frame peak) ===")
    print(f"{'thr':>5} {'Cut_recall':>11} {'Empty_FP':>9}")
    for thr in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
        cr = (C[:, 0] > thr).mean()
        efp = (E[:, 0] > thr).mean()
        print(f"{thr:5.2f} {cr:11.3f} {efp:9.3f}")

    print("\n=== GRADUAL height: Transition recall vs Empty false-fire (all-frames peak) ===")
    print(f"{'g_ht':>5} {'Trans_recall':>13} {'Empty_FP':>9}")
    for gh in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2):
        tr = (T[:, 1] > gh).mean()
        efp = (E[:, 1] > gh).mean()
        print(f"{gh:5.2f} {tr:13.3f} {efp:9.3f}")


if __name__ == "__main__":
    main()
