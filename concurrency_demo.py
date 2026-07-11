"""Where the GPU actually wins: decoding MANY videos at once.

Single-stream, NVDEC+hwdownload loses to 16-thread software decode (PCIe copy + CPU
colour-convert dominate). But NVDEC is a dedicated unit: run several streams
concurrently and it stays busy while each software decoder is fighting for CPU cores.

This runs K concurrent detect() jobs and reports AGGREGATE throughput for each method,
so you can see how they scale with concurrency.

Usage:  python concurrency_demo.py data/speed_1080p.mp4 --streams 1,2,4,8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from detectors import torch_gpu
from detectors.base import ffprobe_info


def run_k(video, method, k):
    _, n_frames, _, _ = ffprobe_info(video)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(torch_gpu.detect, video, method=method) for _ in range(k)]
        [f.result() for f in futs]
    dt = time.perf_counter() - t0
    return k * n_frames / dt  # aggregate frames/sec across all streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--streams", default="1,2,4,8")
    args = ap.parse_args()
    ks = [int(x) for x in args.streams.split(",")]

    fps_src, _, w, h = ffprobe_info(args.video)
    print(f"\nvideo: {args.video}  ({w}x{h} @ {fps_src:.0f}fps)")
    print(f"{'streams':>8} {'cpu-decode agg fps':>20} {'cuda-decode agg fps':>22}")
    print("-" * 52)
    for k in ks:
        cpu = run_k(args.video, "cpu", k)
        cuda = run_k(args.video, "cuda", k)
        print(f"{k:8d} {cpu:20.0f} {cuda:22.0f}", flush=True)
    print("\nagg fps = total frames across all concurrent streams / wall time")
    print("Watch how cuda-decode keeps climbing while cpu-decode plateaus (cores saturate).")


if __name__ == "__main__":
    main()
