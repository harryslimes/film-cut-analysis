import subprocess, time
F = "data/movies/Rear.Window.1954.DVDRip.x264-HANDJOB/Rear.Window.1954.DVDRip.x264-HANDJOB.mkv"

def run_k(K):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "300", "-t", "120",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", F,
            "-vf", "scale_cuda=48:27,hwdownload,format=nv12", "-an", "-f", "null", "-"]
    t0 = time.perf_counter()
    procs = [subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(K)]
    for p in procs:
        p.wait()
    return time.perf_counter() - t0

print("decode 120s of video -> 48x27 (with hwdownload), K parallel streams:")
print(f"{'K':>3} {'wall(s)':>8} {'streams/s':>10} {'vid-sec/s':>10} {'scaling':>8}")
base = None
for K in (1, 2, 4, 8, 12, 16):
    w = run_k(K)
    tp = K / w
    if base is None:
        base = tp
    print(f"{K:3d} {w:8.1f} {tp:10.2f} {K*120/w:10.0f} {tp/base:7.1f}x", flush=True)
