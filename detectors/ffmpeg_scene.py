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
from detector_events import build_minimal_events

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

    with Timer() as t:
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode != 0 and "showinfo" not in stderr:
        raise RuntimeError(f"ffmpeg failed:\n{stderr[-1500:]}")

    cuts = sorted(float(m) for m in _SHOWINFO.findall(stderr))
    return DetectResult(
        name=f"ffmpeg-scene[{method}]", events=build_minimal_events(cuts), elapsed=t.elapsed,
        n_frames=n_frames, fps_source=fps,
        settings={"method": method, "threshold": threshold, "analyse_h": analyse_h},
        extra={},
    )
