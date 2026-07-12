"""Shared types and helpers for cut detectors.

Every detector exposes:  detect(video_path, **opts) -> DetectResult
A "cut" is the timestamp (seconds) of the FIRST frame of a new shot.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict

from cut_events import CutEvent, Confidence, Span, project_cuts, wrap_times   # event-first cut record (design §2)
# CutEvent/Confidence/Span/wrap_times re-exported here so detectors build events off the usual `.base` surface.


@dataclass
class DetectResult:
    name: str
    events: list[CutEvent]      # the ONLY cut record; `cuts` is a derived projection
    elapsed: float              # wall-clock seconds spent detecting
    n_frames: int
    fps_source: float           # frame rate of the video
    scores: list[float] | None = None   # optional per-frame change metric
    extra: dict = field(default_factory=dict)

    @property
    def cuts(self) -> list[float]:
        """Read-only legacy projection: one float timestamp per event, derived at
        access time so it can never drift from the events (design §2). There is
        deliberately no assignable float list -- events are the source of truth."""
        return project_cuts(self.events)

    @property
    def analysed_fps(self) -> float:
        """How many frames/second the detector processed (throughput)."""
        return self.n_frames / self.elapsed if self.elapsed > 0 else float("nan")

    @property
    def realtime_x(self) -> float:
        """Speed relative to playback (e.g. 40x = 40 minutes of video / minute)."""
        return (self.n_frames / self.fps_source) / self.elapsed if self.elapsed > 0 else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("scores", None)          # too big to dump by default
        d.pop("events", None)          # heavy; the summary keeps the light projection
        d["cuts"] = self.cuts          # generated projection (legacy summary shape)
        d["analysed_fps"] = round(self.analysed_fps, 1)
        d["realtime_x"] = round(self.realtime_x, 1)
        return d


def ffprobe_info(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, n_frames, width, height). n_frames may be estimated from duration*fps."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,width,height,duration",
        "-of", "json", video_path,
    ]
    out = json.loads(subprocess.check_output(cmd).decode())
    s = out["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    width, height = int(s["width"]), int(s["height"])
    n = int(s.get("nb_frames") or 0)
    if n == 0:  # containers without a frame count -> estimate
        dur = float(s.get("duration") or 0)
        n = int(round(dur * fps))
    return fps, n, width, height


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
