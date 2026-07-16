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

from .base import DetectResult, Timer, ffprobe_info
from detector_events import (build_transnet_events, gradual_span_frames, peak_prob,
                             transnet_decode_extra)

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
        # hwdownload can only emit the hw surface's OWN sw_format: nv12 for an 8-bit
        # source, p010le for a 10-bit one. A `format=nv12|p010le` alternation cannot fix
        # 10-bit -- filter negotiation picks nv12 out of the candidate list regardless of
        # the order written, hwdownload then rejects it ("Invalid output format nv12 for
        # hwframe download") and the decode yields zero frames. So convert to 8-bit nv12
        # ON the GPU, inside scale_cuda, BEFORE the download: the surface really is nv12
        # by the time hwdownload sees it, and both bit depths work. The 10->8-bit
        # reduction is immaterial at a 48x27 analysis thumbnail.
        vf = "scale_cuda=48:27:format=nv12,hwdownload,format=nv12"
    else:
        pre = []
        vf = "scale=48:27"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *pre, "-i", video_path,
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    # A2-2: transnet owns this decode subprocess, so a nonzero ffmpeg exit is reported
    # honestly (decode_ok False + stderr tail) rather than crashing the CLI (was
    # check=True). ffmpeg being absent still raises (FileNotFoundError) -- a genuinely
    # unrunnable situation, not a partial result. The model is loaded only once we have
    # frames to run it on.
    FRAME_BYTES = 27 * 48 * 3            # one 48x27 rgb24 frame
    decode_extra = {}
    frames = None
    with Timer() as t:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw = proc.stdout or b""
        n_full = len(raw) // FRAME_BYTES        # whole frames only (drop any torn tail)
        if proc.returncode != 0:
            decode_extra = transnet_decode_extra(
                proc.returncode, (proc.stderr or b"").decode("utf-8", "replace"))
        if n_full:
            frames = np.frombuffer(raw[:n_full * FRAME_BYTES], np.uint8).reshape([-1, 27, 48, 3])
            model = _model()
            frames_t = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
            with torch.no_grad():
                single, allf = model.predict_frames(frames_t, quiet=True)
            preds = single.cpu().numpy().reshape(-1)
            allp = allf.cpu().numpy().reshape(-1)
            scenes = model.predictions_to_scenes(preds, threshold=threshold)

    settings = {"method": method, "threshold": threshold, "gradual_height": gradual_height,
                "min_gap_s": min_gap_s, "strobe_guard": strobe_guard}
    if frames is None:
        # nothing usable decoded -> honest empty result; the exporter downgrades status to
        # 'failed' (no events) and surfaces decode_detail as a warning (A3.4).
        return DetectResult(name=f"transnetv2[{method}]", events=[], elapsed=t.elapsed,
                            n_frames=0, fps_source=fps, scores=[], settings=settings,
                            extra=decode_extra)

    # sharp cuts: first frame of every scene after the first (high precision)
    sharp_frames = [int(s[0]) for s in scenes[1:]] if len(scenes) > 1 else []
    n_sharp = len(sharp_frames)
    # dissolve/fade centres from the all-frames stream, if not already near an accepted
    # cut. `accepted` grows as graduals are taken, so graduals suppress each other too --
    # acceptance logic is UNCHANGED from before the amendment.
    gradual_frames = []
    if gradual_height and gradual_height > 0:
        gap = max(1, int(min_gap_s * fps))
        accepted = list(sharp_frames)
        for pf in _gradual_peaks(allp, fps, gradual_height, min_gap_s):
            if all(abs(pf - g) > gap for g in accepted):
                gradual_frames.append(pf)
                accepted.append(pf)
    n_gradual = len(gradual_frames)

    # Hand plain per-candidate data to the dependency-light builder (A3.5). Hard cuts are
    # emitted before graduals, so a rounded-time collision keeps the hard cut (A1 dedupe).
    # Strobe suppression happens inside the builder. frame = the scene-start / peak frame
    # (already the first frame of the new shot -- A2 conformant). A2-3: hard-cut confidence
    # is the PEAK single-frame prob near the boundary (peak_prob), not preds[f] at the
    # scene-start frame (post-spike, ~0); the gradual stream already reads its own peak.
    hard = [(f, peak_prob(preds, f)) for f in sharp_frames]
    gradual = [(f, float(allp[f]), *gradual_span_frames(allp, f, gradual_height))
               for f in gradual_frames]
    events, n_strobe_removed = build_transnet_events(
        hard, gradual, fps, strobe_guard=strobe_guard,
        strobe_params=dict(dens_window=8, dens_max=8, merge_gap=4, keep_gap=2.5))

    return DetectResult(
        name=f"transnetv2[{method}]", events=events, elapsed=t.elapsed,
        n_frames=len(frames), fps_source=fps, scores=preds.tolist(),
        settings=settings,
        # decode_extra is {} on a clean exit; on a nonzero-but-usable decode it carries
        # decode_ok=False so the exporter downgrades this run to 'partial' (A3.4).
        extra={"n_sharp": n_sharp, "n_gradual": n_gradual,
               "n_strobe_removed": n_strobe_removed, **decode_extra},
    )
