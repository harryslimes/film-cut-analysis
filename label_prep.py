"""Prepare a sequence for the labelling web tool.

Runs TransNetV2 over a timespan, extracts EVERY candidate cut (local maxima of the
sharp AND gradual prediction streams, down to a low floor) with its confidence, and
saves before/after thumbnails. Marks which candidates are currently 'kept' (match the
video's .srt). The web tool then lets you reject wrong keeps and promote missed cuts.

  python label_prep.py --video <mkv> --start 5030 --end 5115 --name vertigo_nightmare
"""
import argparse, json, os, re, subprocess
import numpy as np, torch, cv2
from detectors.base import ffprobe_info
from detectors.transnet import _model


def local_maxima(sig, floor, win=3, sep=3):
    peaks, n, last = [], len(sig), -10**9
    i = 1
    while i < n - 1:
        if sig[i] >= floor and sig[i] >= sig[i-1] and sig[i] > sig[i+1]:
            j = i
            while j+1 < n and sig[j+1] >= sig[j]:
                j += 1
            if j - last >= sep:
                peaks.append(j); last = j
            i = j + 1
        else:
            i += 1
    return peaks


def srt_cuts(path):
    if not os.path.exists(path):
        return []
    times = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->", open(path).read())
    starts = sorted({int(h)*3600+int(m)*60+int(s)+int(ms)/1000 for h, m, s, ms in times})
    return starts[1:] if starts else []   # each shot starts at a cut (drop the 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--delta", type=float, default=0.2); ap.add_argument("--tw", type=int, default=200)
    ap.add_argument("--floor", type=float, default=0.10)
    a = ap.parse_args()
    fps, N, W, H = ffprobe_info(a.video)
    outdir = f"results/label_{a.name}"; os.makedirs(outdir + "/thumbs", exist_ok=True)
    a0 = a.start - 1.0

    # tiny frames for TransNet + display thumbs (colour)
    tn = np.frombuffer(subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(a0),
        "-t",str(a.end-a0+1),"-i",a.video,"-vf","scale=48:27,format=rgb24","-f","rawvideo","-"],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout, np.uint8).reshape([-1,27,48,3])
    th = int(round(a.tw*H/W/2)*2)
    disp = np.frombuffer(subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(a0),
        "-t",str(a.end-a0+1),"-i",a.video,"-vf",f"scale={a.tw}:{th},format=bgr24","-f","rawvideo","-"],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout, np.uint8).reshape([-1,th,a.tw,3])

    model = _model()
    with torch.no_grad():
        single, allf = model.predict_frames(torch.from_numpy(np.ascontiguousarray(tn)).to(model.device), quiet=True)
    single = single.cpu().numpy().reshape(-1); allf = allf.cpu().numpy().reshape(-1)

    # candidate frames = local maxima of either stream
    frames = {}
    for f in local_maxima(single, a.floor): frames[f] = max(frames.get(f, 0), single[f])
    for f in local_maxima(allf, max(a.floor, 0.15)): frames[f] = max(frames.get(f, 0), allf[f])
    kept_times = srt_cuts(a.video.rsplit(".", 1)[0] + ".srt")
    d = int(round(a.delta * fps)); nlocal = len(tn)

    cands = []
    for cid, f in enumerate(sorted(frames)):
        t = round(a0 + f / fps, 3)
        ib, ia = max(0, f-d), min(nlocal-1, f+d)
        cv2.imwrite(f"{outdir}/thumbs/{cid}_b.jpg", disp[ib])
        cv2.imwrite(f"{outdir}/thumbs/{cid}_a.jpg", disp[ia])
        cands.append({"id": cid, "frame": int(f), "time": t,
                      "prob": round(float(frames[f]), 3),
                      "sharp": round(float(single[f]), 3), "gradual": round(float(allf[f]), 3),
                      "kept": False})
    # mark ONE nearest candidate per current kept cut (avoids near-duplicate keeps)
    for kt in kept_times:
        near = min(cands, key=lambda c: abs(c["time"] - kt))
        if abs(near["time"] - kt) <= 0.4:
            near["kept"] = True
    meta = {"video": a.video, "name": a.name, "start": a.start, "end": a.end,
            "fps": fps, "candidates": cands}
    json.dump(meta, open(f"{outdir}/candidates.json", "w"), indent=1)
    print(f"{len(cands)} candidates ({sum(c['kept'] for c in cands)} currently kept) -> {outdir}/")


if __name__ == "__main__":
    main()
