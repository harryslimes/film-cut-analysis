"""FFmpeg's built-in `scene` metric.

The `select='gt(scene,T)'` filter emits a normalised [0,1] whole-frame difference
score; we print the timestamp whenever it exceeds T. This is the naive SAD method:
fast, zero Python per-frame cost, but a flash/whip-pan can trip it. Two flavours:

  method="cpu"   : software decode
  method="cuda"  : NVDEC decode + on-GPU downscale (scale_cuda), then hwdownload.

Detection at low resolution is plenty accurate and much faster, so we downscale to
`analyse_h` rows either way.
"""
from __future__ import annotations

import re
import subprocess

from .base import DetectResult, Timer, ffprobe_info
from detector_events import build_minimal_events, ffmpeg_decode_extra

_SHOWINFO = re.compile(r"pts_time:([0-9.]+)")


def detect(video_path, threshold=0.4, analyse_h=180, method="cpu") -> DetectResult:
    fps, n_frames, w, h = ffprobe_info(video_path)
    aw = -2  # keep aspect, even width

    if method == "cuda":
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = (f"scale_cuda={aw}:{analyse_h},hwdownload,format=nv12,"
              f"select='gt(scene,{threshold})',showinfo")
    else:
        pre = []
        vf = f"scale={aw}:{analyse_h},select='gt(scene,{threshold})',showinfo"

    cmd = ["ffmpeg", "-hide_banner", "-nostats", *pre, "-i", video_path,
           "-vf", vf, "-an", "-f", "null", "-"]

    # ffmpeg binary absent raises FileNotFoundError from subprocess.run itself -- the only
    # "genuinely unrunnable" case we still let propagate (S3-F2 gap 2).
    with Timer() as t:
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    stderr = proc.stderr.decode(errors="replace")

    cuts = sorted(float(m) for m in _SHOWINFO.findall(stderr))
    # A3.4 / gap 2: this detector owns the decode subprocess, so report a nonzero exit
    # rather than raise -- the exporter emits status "partial" (some showinfo survived) or
    # "failed" (none, so events is empty), instead of the tool crashing on a bad decode.
    extra = ffmpeg_decode_extra(proc.returncode, stderr)
    return DetectResult(
        name=f"ffmpeg-scene[{method}]", events=build_minimal_events(cuts), elapsed=t.elapsed,
        n_frames=n_frames, fps_source=fps,
        settings={"method": method, "threshold": threshold, "analyse_h": analyse_h},
        extra=extra,
    )
