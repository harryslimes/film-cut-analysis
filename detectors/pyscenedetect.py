"""PySceneDetect -- the de-facto standard library.

ContentDetector : HSV-weighted frame difference. Because a brightness change mostly
    moves the Value channel, weighting H/S higher makes it robust to a light switching
    on -- one of our target false positives.

AdaptiveDetector: compares each frame's content score to a rolling average of its
    neighbours. A whip-pan produces a sustained high score (no single spike), so it is
    NOT flagged; a real cut spikes against its neighbours. Best all-rounder for the
    two false-positive cases the user cares about.

Both support `downscale` for speed -- cut detection is fine at ~1/4 resolution.
"""
from __future__ import annotations

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

from .base import DetectResult, Timer, ffprobe_info
from detector_events import build_minimal_events


def _run(video_path, detector, name, downscale, settings):
    fps, n_frames, w, h = ffprobe_info(video_path)
    with Timer() as t:
        video = open_video(video_path)
        mgr = SceneManager()
        if downscale and downscale > 1:
            mgr.auto_downscale = False
            mgr.downscale = downscale
        mgr.add_detector(detector)
        mgr.detect_scenes(video, show_progress=False)
        scenes = mgr.get_scene_list()
    # scene list is (start, end) pairs; a cut is the start of every scene after the first
    cuts = sorted({s[0].get_seconds() for s in scenes[1:]}) if len(scenes) > 1 else []
    return DetectResult(name=name, events=build_minimal_events(cuts), elapsed=t.elapsed,
                        n_frames=n_frames, fps_source=fps, settings=settings, extra={})


def detect_content(video_path, threshold=27.0, downscale=2) -> DetectResult:
    return _run(video_path, ContentDetector(threshold=threshold), "psd-content", downscale,
                {"method": "cpu", "threshold": threshold, "downscale": downscale})


def detect_adaptive(video_path, adaptive_threshold=3.0, min_content_val=15.0, downscale=2) -> DetectResult:
    det = AdaptiveDetector(adaptive_threshold=adaptive_threshold,
                           min_content_val=min_content_val)
    return _run(video_path, det, "psd-adaptive", downscale,
                {"method": "cpu", "adaptive_threshold": adaptive_threshold,
                 "min_content_val": min_content_val, "downscale": downscale})
