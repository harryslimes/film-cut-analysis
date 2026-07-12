"""Custom NVDEC + torch pipeline -- built for raw speed on the RTX 5090.

Design (the producer/consumer idea):
  * ffmpeg decodes on the NVDEC block, downscales on GPU (scale_cuda) to a tiny frame,
    and pipes raw RGB to stdout.
  * A reader THREAD pulls frames off the pipe into a queue (overlaps decode with compute).
  * The main thread batches frames onto the CUDA cores, converts RGB->HSV, and computes
    a brightness-robust frame-difference score. NVDEC and the CUDA cores are separate
    silicon, so decode and inference overlap essentially for free.

Robustness:
  * HSV metric with the Value channel down-weighted -> a light switching on (which only
    scales V) barely moves the score.  [handles false-positive #1]
  * Adaptive thresholding: a cut must SPIKE above the local neighbourhood average, so a
    sustained whip-pan (high but flat score) is not flagged.  [handles #2]
"""
from __future__ import annotations

import queue
import subprocess
import threading
import warnings

import numpy as np
import torch

# frames come from a read-only pipe buffer; we only read them, never write in place
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from .base import DetectResult, Timer, ffprobe_info, wrap_times


def _rgb_to_hsv(x):  # x: (B,3,H,W) float in [0,1] -> (B,3,H,W) H,S,V in [0,1]
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + 1e-6)
    dc = delta + 1e-6
    rc = (maxc - r) / dc
    gc = (maxc - g) / dc
    bc = (maxc - b) / dc
    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta < 1e-6, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=1)


def _reader(proc, frame_bytes, q):
    stream = proc.stdout
    while True:
        buf = stream.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        q.put(buf)
    q.put(None)  # sentinel


def detect(video_path, method="cuda", analyse_h=108, batch=256,
           wV=0.10, adaptive_ratio=3.0, min_score=0.045, window=15,
           device="cuda") -> DetectResult:
    fps, n_frames, W, H = ffprobe_info(video_path)
    aw = (int(round(W / H * analyse_h)) // 2) * 2  # even width, keep aspect
    ah = analyse_h
    frame_bytes = aw * ah * 3

    if method == "cuda":
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        # hwdownload can only fetch the GPU frame's own sw format (nv12); the final
        # -pix_fmt rgb24 then converts on the CPU via swscale.
        vf = f"scale_cuda={aw}:{ah},hwdownload,format=nv12"
    else:
        pre = []
        vf = f"scale={aw}:{ah}"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *pre, "-i", video_path,
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    wH = wS = (1.0 - wV) / 2.0
    weights = torch.tensor([wH, wS, wV], device=device).view(1, 3, 1, 1)

    scores = []
    with Timer() as t:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=frame_bytes * 8)
        q: queue.Queue = queue.Queue(maxsize=8)
        th = threading.Thread(target=_reader, args=(proc, frame_bytes, q), daemon=True)
        th.start()

        prev_last = None
        pending = []

        def flush(bufs):
            nonlocal prev_last
            arr = np.frombuffer(b"".join(bufs), dtype=np.uint8).reshape(-1, ah, aw, 3)
            x = torch.from_numpy(arr).to(device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            hsv = _rgb_to_hsv(x)
            seq = hsv if prev_last is None else torch.cat([prev_last[None], hsv], dim=0)
            d = (seq[1:] - seq[:-1]).abs()
            # circular hue distance
            d[:, 0] = torch.minimum(d[:, 0], 1.0 - d[:, 0])
            per_frame = (d * weights).mean(dim=(1, 2, 3))  # (n,)
            scores.extend(per_frame.detach().cpu().tolist())
            prev_last = hsv[-1]

        while True:
            item = q.get()
            if item is None:
                break
            pending.append(item)
            if len(pending) >= batch:
                flush(pending)
                pending = []
        if pending:
            flush(pending)
        proc.wait()

    # --- adaptive thresholding on the score curve ---
    s = np.asarray(scores, dtype=np.float64)
    cuts = []
    n = len(s)
    last_cut = -10 ** 9
    for i in range(1, n):
        a = max(0, i - window)
        b = min(n, i + window + 1)
        neigh = np.concatenate([s[a:i], s[i + 1:b]])
        local = np.median(neigh) if neigh.size else 0.0
        if s[i] >= min_score and s[i] >= adaptive_ratio * (local + 1e-6):
            if i - last_cut >= 4:  # non-max suppression
                cuts.append(round(i / fps, 4))
                last_cut = i

    return DetectResult(
        name=f"torch-gpu[{method}]", events=wrap_times(cuts), elapsed=t.elapsed,
        n_frames=n_frames, fps_source=fps, scores=scores,
        extra={"analyse_res": f"{aw}x{ah}", "batch": batch,
               "adaptive_ratio": adaptive_ratio, "min_score": min_score},
    )
