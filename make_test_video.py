"""Generate a synthetic video with KNOWN cut times + deliberate false-positive traps.

Because real movies are copyrighted, we build our own ground-truth clip. Each shot is
a moving crop of a large textured "world" image (so there is real, continuous motion
and detail for content detectors to chew on). Between shots we hard-cut to a new world.

We also embed two traps *inside* single shots -- these must NOT be reported as cuts:
  * LIGHT-ON: a near-black shot where a "light" ramps the brightness up hard.
  * WHIP-PAN: a shot whose crop window races across the world (fast camera pan).

Outputs:  data/test.mp4  and  data/test.gt.json  (ground-truth cut times in seconds).

Usage:  python make_test_video.py --seconds 60 --fps 30 --w 1280 --h 720
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np


def make_world(rng, h, w):
    """A big textured image: smooth low-freq color gradients + sharp high-freq detail."""
    H, W = h * 2, w * 2  # world is larger than the frame so we can pan/crop within it
    # low frequency color field
    ys = np.linspace(0, rng.uniform(2, 6) * np.pi, H)[:, None]
    xs = np.linspace(0, rng.uniform(2, 6) * np.pi, W)[None, :]
    base = np.stack([
        0.5 + 0.5 * np.sin(ys * rng.uniform(0.3, 1.0) + xs * rng.uniform(0.3, 1.0) + rng.uniform(0, 6)),
        0.5 + 0.5 * np.sin(ys * rng.uniform(0.3, 1.0) - xs * rng.uniform(0.3, 1.0) + rng.uniform(0, 6)),
        0.5 + 0.5 * np.cos(ys * rng.uniform(0.3, 1.0) + xs * rng.uniform(0.3, 1.0) + rng.uniform(0, 6)),
    ], axis=-1)
    # high frequency detail (fine texture so frame-diff has something to bite on)
    detail = rng.normal(0, 0.12, (H, W, 1))
    img = np.clip(base + detail, 0, 1)
    # a few bright geometric shapes for structure
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        r = rng.integers(20, 120)
        color = rng.uniform(0, 1, 3)
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        img[mask] = color
    return (img * 255).astype(np.uint8)


def crop(world, y, x, h, w):
    H, W = world.shape[:2]
    y = int(np.clip(y, 0, H - h)); x = int(np.clip(x, 0, W - w))
    return world[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--out", default="data/test.mp4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    h, w, fps = args.h, args.w, args.fps
    total = int(args.seconds * fps)

    # Plan shots: random lengths 1.2 - 4.5 s. Tag a couple with traps.
    shots = []
    t = 0
    idx = 0
    while t < total:
        length = int(rng.uniform(1.2, 4.5) * fps)
        length = min(length, total - t)
        kind = "normal"
        # sprinkle traps: every ~4th shot alternate a trap type, if long enough
        if length > int(1.5 * fps):
            if idx % 4 == 2:
                kind = "lighton"
            elif idx % 4 == 3:
                kind = "whippan"
        shots.append((t, length, kind))
        t += length
        idx += 1

    cut_times = [round(s / fps, 4) for (s, _, _) in shots[1:]]  # first shot start isn't a cut

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE,
    )

    trap_log = []
    for (start, length, kind) in shots:
        world = make_world(rng, h, w)
        H, W = world.shape[:2]
        # random start crop + slow drift (handheld feel)
        y0, x0 = rng.uniform(0, H - h), rng.uniform(0, W - w)
        dy, dx = rng.uniform(-1, 1) * 1.5, rng.uniform(-1, 1) * 1.5
        if kind == "whippan":
            # race the window across the whole world -> huge frame-to-frame motion, but no cut
            dx = (W - w) / max(length - 1, 1) * rng.choice([-1, 1])
            dy = rng.uniform(-1, 1)
        # traps are stored as the shot's interior interval: any cut reported strictly
        # inside (away from the real boundaries) is a false positive attributable to it.
        if kind in ("whippan", "lighton"):
            trap_log.append({
                "type": "whip-pan" if kind == "whippan" else "light-on",
                "start": round(start / fps, 3),
                "end": round((start + length) / fps, 3),
                "time": round((start + length // 2) / fps, 3),
            })
        for f in range(length):
            frame = crop(world, y0 + dy * f, x0 + dx * f, h, w).astype(np.float32)
            if kind == "lighton":
                # first half near-black, then a hard brightness ramp (a lamp switched on)
                half = length // 2
                if f < half:
                    gain = 0.06
                else:
                    gain = min(1.0, 0.06 + (f - half) / max(fps * 0.25, 1) * 0.94)
                frame *= gain
            np.clip(frame, 0, 255, out=frame)
            ff.stdin.write(frame.astype(np.uint8).tobytes())

    ff.stdin.close()
    ff.wait()

    gt = {
        "video": args.out, "fps": fps, "width": w, "height": h,
        "n_frames": total, "duration_s": round(total / fps, 3),
        "cuts": cut_times,                 # <-- ground truth
        "traps": trap_log,                 # events that must NOT be flagged as cuts
    }
    gt_path = args.out.rsplit(".", 1)[0] + ".gt.json"
    with open(gt_path, "w") as fh:
        json.dump(gt, fh, indent=2)
    print(f"wrote {args.out}  ({total} frames, {total/fps:.1f}s)")
    print(f"wrote {gt_path}  ({len(cut_times)} cuts, {len(trap_log)} traps)")


if __name__ == "__main__":
    main()
