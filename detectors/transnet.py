"""TransNetV2 -- small neural net, near state-of-the-art shot-boundary accuracy.

The network ingests tiny 48x27 thumbnails and looks at a sliding window of frames, so
it natively handles flashes, whip-pans, AND gradual transitions (dissolves/fades) that
threshold methods miss entirely. Inference is trivial for a 5090, so throughput is
bounded by frame decode -- which is why we offer an NVDEC decode path.

We decode ourselves (optionally on NVDEC) to 48x27 rgb, then call the model's
predict_frames (it handles the 100-frame sliding window internally).
"""
from __future__ import annotations

import subprocess

import numpy as np
import torch

from .base import DetectResult, Timer, ffprobe_info, wrap_times

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from transnetv2_pytorch import TransNetV2
        _MODEL = TransNetV2(device="cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.eval()
    return _MODEL


def _gradual_peaks(allf, fps, height, min_gap_s):
    """Local maxima of the all-frames stream = centres of dissolves/fades.
    The single-frames stream barely responds to a slow dissolve, but the all-frames
    stream makes a smooth hump; its peak marks the transition."""
    n = len(allf)
    gap = max(1, int(min_gap_s * fps))
    peaks = []
    i = 1
    while i < n - 1:
        if allf[i] >= height and allf[i] >= allf[i - 1] and allf[i] > allf[i + 1]:
            # walk to the true local max of this hump
            j = i
            while j + 1 < n and allf[j + 1] >= allf[j]:
                j += 1
            peaks.append(j)
            i = j + gap  # suppress neighbours within min_gap
        else:
            i += 1
    return peaks


def detect(video_path, method="cuda", threshold=0.4,
           gradual_height=0.35, min_gap_s=0.4, strobe_guard=True) -> DetectResult:
    # 0.4 default tuned on Charade vs real MovieNet labels: best F1 (0.961), recovers
    # ~1/3 of dissolves the stock 0.5 misses at negligible precision cost. Lower to 0.3
    # to lead on recall, raise to 0.5 for max precision.
    # gradual_height>0 adds dissolve/fade centres from the all-frames stream (the single
    # stream barely responds to slow dissolves); set to 0 to disable. Default 0.35 is the
    # consensus of three signals: Charade F1 vs MovieNet, transition recall on the diverse
    # HuggingFace set (~66%), and His Girl Friday ASL vs the published 13.7s. Drop to ~0.30
    # for dissolve-heavy films (higher transition recall), raise to trim faint false peaks.
    fps, n_frames, W, H = ffprobe_info(video_path)

    if method == "cuda":
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = "scale_cuda=48:27,hwdownload,format=nv12"
    else:
        pre = []
        vf = "scale=48:27"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *pre, "-i", video_path,
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    model = _model()
    with Timer() as t:
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
        frames_t = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
        with torch.no_grad():
            single, allf = model.predict_frames(frames_t, quiet=True)
        preds = single.cpu().numpy().reshape(-1)
        allp = allf.cpu().numpy().reshape(-1)
        scenes = model.predictions_to_scenes(preds, threshold=threshold)

    # sharp cuts: first frame of every scene after the first (high precision)
    sharp = [int(s[0]) for s in scenes[1:]] if len(scenes) > 1 else []
    n_sharp = len(sharp)
    # dissolve/fade centres from the all-frames stream, if not already a sharp cut
    n_gradual = 0
    if gradual_height and gradual_height > 0:
        gap = max(1, int(min_gap_s * fps))
        for pf in _gradual_peaks(allp, fps, gradual_height, min_gap_s):
            if all(abs(pf - s) > gap for s in sharp):
                sharp.append(pf)
                n_gradual += 1
    cuts = sorted(round(f / fps, 4) for f in sharp)
    # suppress strobe/flash bursts (e.g. Vertigo's Nightmare): thins only pathologically
    # dense regions, leaves normal cutting untouched. See postfilter.dampen_strobe.
    n_before = len(cuts)
    if strobe_guard:
        from postfilter import dampen_strobe
        # safe global setting: thins only pathologically dense regions, zero collateral
        # on normal fast cutting. For a known flash montage use a targeted region override.
        cuts = dampen_strobe(cuts, dens_window=8, dens_max=8, merge_gap=4, keep_gap=2.5)
    return DetectResult(
        name=f"transnetv2[{method}]", events=wrap_times(cuts), elapsed=t.elapsed,
        n_frames=len(frames), fps_source=fps, scores=preds.tolist(),
        extra={"threshold": threshold, "gradual_height": gradual_height,
               "n_sharp": n_sharp, "n_gradual": n_gradual,
               "n_strobe_removed": n_before - len(cuts)},
    )
