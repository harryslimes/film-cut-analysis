"""Extract cut times from a video and write them as CSV / EDL / JSON.

  python cut_times.py movie.mkv                         # -> movie.cuts.csv
  python cut_times.py movie.mkv --detector transnet-cuda --format edl
  python cut_times.py movie.mkv --detector psd-adaptive --format all

Pick a detector by name (see `python cut_times.py --list`). Sensible default is a fast,
brightness/pan-robust one; use transnet-* for best accuracy on real footage.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys

from cut_export import build_document


def hhmmssff(seconds, fps):
    f = int(round(seconds * fps))
    ff = f % int(round(fps))
    s = f // int(round(fps))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}:{ff:02d}"


def write_csv(path, cuts, fps):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cut_index", "time_seconds", "timecode"])
        for i, c in enumerate(cuts, 1):
            wr.writerow([i, f"{c:.3f}", hhmmssff(c, fps)])


def srt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, cuts, duration, label="Shot"):
    """One subtitle per shot; the number increments at each cut (watch in VLC)."""
    bounds = [0.0] + list(cuts) + [duration]
    with open(path, "w") as fh:
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if end <= start:
                continue
            fh.write(f"{i+1}\n{srt_ts(start)} --> {srt_ts(end)}\n"
                     f"{label} {i+1}\n\n")


def write_edl(path, cuts, fps, duration):
    # minimal CMX3600-style EDL: one clip per shot
    bounds = [0.0] + list(cuts) + [duration]
    with open(path, "w") as fh:
        fh.write("TITLE: cut-time-extractor\nFCM: NON-DROP FRAME\n\n")
        for i in range(len(bounds) - 1):
            src_in = hhmmssff(bounds[i], fps)
            src_out = hhmmssff(bounds[i + 1], fps)
            fh.write(f"{i+1:03d}  AX       V     C        "
                     f"{src_in} {src_out} {src_in} {src_out}\n")


# registry key -> (detector_id, backend). Explicit and auditable rather than parsed from
# the display name (design §3.1: id/backend are stable identifiers; the legacy display
# string is never parsed). backend = the decode path the key selects; psd/motion-vectors
# offer no such choice, so backend is absent (None -> omitted). Unknown keys fall back to
# the raw key with no backend.
_DETECTOR_IDS = {
    "ffmpeg-cpu":     ("ffmpeg-scene", "cpu"),
    "ffmpeg-cuda":    ("ffmpeg-scene", "cuda"),
    "psd-content":    ("psd-content", None),
    "psd-adaptive":   ("psd-adaptive", None),
    "torch-cpu":      ("torch-gpu", "cpu"),
    "torch-cuda":     ("torch-gpu", "cuda"),
    "transnet-cpu":   ("transnetv2", "cpu"),
    "transnet-cuda":  ("transnetv2", "cuda"),
    "motion-vectors": ("motion-vectors", None),
}


def _iso_utc(epoch=None):
    """UTC timestamp as ...Z -- from `epoch` (an os.stat mtime) if given, else now."""
    dt = (datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
          if epoch is not None else datetime.datetime.now(datetime.timezone.utc))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_provenance(repo_dir):
    """(commit, dirty) read from git at runtime, or (None, None) -- never faked -- when
    git or the .git dir is unavailable (a warning then goes to stderr)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                               capture_output=True, text=True, check=True).stdout
        return commit, bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        print("warning: git provenance unavailable; omitting tool_commit/tool_dirty",
              file=sys.stderr)
        return None, None


def build_v2_document(video_path, detector_key, res, repo_dir=None):
    """Assemble the design-§3 v2 document for a completed detector run and return it
    (validated by the S3-E1 serializer). Kept separate from main() so it is importable
    and testable without a video. `res` is a DetectResult (reads .events / .fps_source /
    .n_frames / .name / .extra)."""
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    st = os.stat(video_path)

    duration = res.n_frames / res.fps_source if res.fps_source else 0.0
    have_duration = duration > 0

    source = {"path": video_path, "size_bytes": st.st_size,
              "mtime_utc": _iso_utc(st.st_mtime),
              "duration_seconds": round(duration, 3) if have_duration else None}

    detector_id, backend = _DETECTOR_IDS.get(detector_key, (detector_key, None))
    commit, dirty = _git_provenance(repo_dir)
    run = {"detector_id": detector_id, "backend": backend,
           "tool_commit": commit, "tool_dirty": dirty,
           "model": "transnetv2-pytorch (weights as installed)" if detector_id == "transnetv2" else None,
           "settings": dict(res.extra),   # the detector's self-reported resolved params (see report)
           "generated_by": "cut_times.py", "generated_utc": _iso_utc()}

    warnings = []
    if res.extra.get("warning"):           # e.g. motion-vectors' fixed-GOP warning
        warnings.append(res.extra["warning"])
    if not have_duration:
        warnings.append("source duration unavailable; coverage omitted "
                        "(time-in-coverage check relaxes to time >= 0)")

    analysis = {"status": "complete",
                "coverage": [{"start": 0.0, "end": round(duration, 3)}] if have_duration else [],
                "warnings": warnings}

    return build_document(res.events, source=source, run=run, analysis=analysis,
                          fps=res.fps_source, video=video_path, detector=res.name)


def main():
    from detectors import REGISTRY   # deferred: keeps cut_times importable without the
                                     # heavy detector deps (torch/av/scenedetect) for tests
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--detector", default="torch-cpu")
    ap.add_argument("--format", default="csv", choices=["csv", "edl", "srt", "json", "all"])
    ap.add_argument("--label", default="Shot", help="subtitle label prefix (SRT)")
    ap.add_argument("--list", action="store_true", help="list available detectors")
    args = ap.parse_args()

    if args.list or not args.video:
        print("available detectors:")
        for k in REGISTRY:
            print("  ", k)
        return

    if args.detector not in REGISTRY:
        raise SystemExit(f"unknown detector '{args.detector}'. try --list")

    res = REGISTRY[args.detector](args.video)
    stem = args.video.rsplit(".", 1)[0]
    duration = res.n_frames / res.fps_source

    print(f"{res.name}: {len(res.cuts)} cuts in {res.elapsed:.1f}s "
          f"({res.analysed_fps:.0f} fps, {res.realtime_x:.0f}x realtime)")

    wrote = []
    if args.format in ("csv", "all"):
        p = stem + ".cuts.csv"; write_csv(p, res.cuts, res.fps_source); wrote.append(p)
    if args.format in ("edl", "all"):
        p = stem + ".cuts.edl"; write_edl(p, res.cuts, res.fps_source, duration); wrote.append(p)
    if args.format in ("srt", "all"):
        p = stem + ".cuts.srt"; write_srt(p, res.cuts, duration, args.label); wrote.append(p)
    if args.format in ("json", "all"):
        p = stem + ".cuts.json"
        doc = build_v2_document(args.video, args.detector, res)
        json.dump(doc, open(p, "w"), indent=2); wrote.append(p)
    for p in wrote:
        print("wrote", p)


if __name__ == "__main__":
    main()
