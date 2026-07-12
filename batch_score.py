"""Score every movie under data/movies against its MovieNet ground truth.

For each video it: maps the film to an IMDb id, converts the MovieNet shot list to
reference cuts (using the film's real fps), runs TransNetV2, syncs the reference onto
your copy's timeline (handles leader/fps offset), scores P/R/F1, and writes a prediction
SRT + a ground-truth SRT.

TransNet results are cached in <stem>.cuts.json so re-runs are instant.

  python batch_score.py                 # all movies
  python batch_score.py --tol 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from cut_times import write_srt
from movienet_to_gt import parse_shots
from evaluate import match
from sync import sync


def _detect_worker(video):
    """Runs in its own process -> independent NVDEC stream + CUDA context.
    Decodes the film, runs TransNetV2, caches cuts. Returns (video, n_cuts, err)."""
    import json as _json
    try:
        from detectors import transnet
        r = transnet.detect(video, method="cuda")
        stem = video.rsplit(".", 1)[0]
        # design §4: tag the private cache so a v2-export consumer never mistakes this
        # reduced shape for a full cut-events document. Reader below still accepts both.
        _json.dump({"format": "film-cut-analysis/score-cache", "cuts": r.cuts, "fps": r.fps_source},
                   open(stem + ".cuts.json", "w"))
        return (video, len(r.cuts), None)
    except Exception as e:  # noqa
        return (video, 0, str(e)[:200])

# folder/filename keyword -> IMDb id (all confirmed present in MovieNet)
KNOWN = {
    "wizard of oz": "tt0032138", "wizard.of.oz": "tt0032138",
    "wonderful life": "tt0038650", "wonderful.life": "tt0038650",
    "rear window": "tt0047396", "rear.window": "tt0047396",
    "vertigo": "tt0052357",
    "north by northwest": "tt0053125", "north.by.northwest": "tt0053125",
    "charade": "tt0056923",
    "his girl friday": "tt0032599", "his_girl_friday": "tt0032599",  # not in MovieNet
}

SHOT_DIR = "data/movienet/shot_detection/shot"
VIDEO_INFO = "data/movienet/movie1K.video_info.v1.json"


def find_imdb(path):
    low = path.lower()
    for kw, imdb in KNOWN.items():
        if kw in low:
            return imdb
    return None


def discover_videos():
    vids = []
    for ext in ("mp4", "mkv", "avi", "m4v", "webm"):
        vids += glob.glob(f"data/movies/**/*.{ext}", recursive=True)
    return sorted(set(vids))


def ref_cuts_for(imdb, video_info):
    shotfile = os.path.join(SHOT_DIR, f"{imdb}.txt")
    if not os.path.exists(shotfile):
        return None, None
    shots = parse_shots(shotfile)
    fps = float(video_info.get(imdb, {}).get("fps") or 0) or 24.0
    cuts = sorted(round(s[0] / fps, 4) for s in shots[1:])   # start of each shot after 1st
    return cuts, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--redetect", action="store_true", help="ignore cached cuts")
    ap.add_argument("--workers", type=int, default=4, help="films decoded in parallel")
    args = ap.parse_args()

    video_info = json.load(open(VIDEO_INFO)) if os.path.exists(VIDEO_INFO) else {}

    # figure out which films are scorable and which still need detection
    plan = []  # (video, stem, imdb, ref, tag)
    for video in discover_videos():
        stem = video.rsplit(".", 1)[0]
        imdb = find_imdb(video)
        tag = os.path.basename(video)[:42]
        if not imdb:
            print(f"SKIP  {tag:44} (no IMDb mapping)"); continue
        ref, ref_fps = ref_cuts_for(imdb, video_info)
        if ref is None:
            print(f"SKIP  {tag:44} ({imdb}: no MovieNet shot file)"); continue
        plan.append((video, stem, imdb, ref, tag))

    to_detect = [p[0] for p in plan
                 if args.redetect or not os.path.exists(p[1] + ".cuts.json")]
    if to_detect:
        print(f"\ndetecting {len(to_detect)} films, {args.workers} in parallel "
              f"(NVDEC concurrent streams)...", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for video, n, err in ex.map(_detect_worker, to_detect):
                tag = os.path.basename(video)[:42]
                print(f"  {'ERR ' if err else 'done'} {tag:44} "
                      f"{('cuts='+str(n)) if not err else err}", flush=True)
        print(f"detection wall time: {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for video, stem, imdb, ref, tag in plan:
        cache = stem + ".cuts.json"
        if not os.path.exists(cache):
            print(f"SKIP  {tag:44} (detection failed)"); continue
        d = json.load(open(cache)); my, my_fps = d["cuts"], d.get("fps", 24.0)

        # sync reference onto our timeline
        s = sync(ref, my, tol=args.tol)
        gt = sorted(round(s["scale"] * c + s["offset"], 4) for c in ref)
        gt = [c for c in gt if c >= 0]
        json.dump({"cuts": gt, "fps": my_fps, "sync": s}, open(stem + ".gt.json", "w"))

        # write SRTs
        dur = (max(my + gt) + 5) if (my or gt) else 5
        write_srt(stem + ".srt", my, dur, "Shot")
        write_srt(stem + ".movienet.srt", gt, dur, "Shot")

        m = match(my, gt, tol=args.tol)
        rows.append({"film": tag, "imdb": imdb, "ref": len(ref), "pred": len(my),
                     "P": m["precision"], "R": m["recall"], "F1": m["f1"],
                     "sync_rate": s["match_rate"], "scale": s["scale"], "offset": s["offset"]})
        print(f"OK    {tag:44} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  sync={s['match_rate']:.2f}", flush=True)

    # summary
    print("\n" + "=" * 78)
    print(f"{'film':44} {'ref':>5} {'pred':>5} {'P':>5} {'R':>5} {'F1':>5} {'sync':>5}")
    print("-" * 78)
    for r in rows:
        print(f"{r['film']:44} {r['ref']:5d} {r['pred']:5d} {r['P']:5.2f} {r['R']:5.2f} "
              f"{r['F1']:5.2f} {r['sync_rate']:5.2f}")
    if rows:
        import statistics
        print("-" * 78)
        print(f"{'MEAN':44} {'':5} {'':5} "
              f"{statistics.mean(r['P'] for r in rows):5.2f} "
              f"{statistics.mean(r['R'] for r in rows):5.2f} "
              f"{statistics.mean(r['F1'] for r in rows):5.2f}")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/batch_scores.json", "w"), indent=2)
    print(f"\nwrote results/batch_scores.json  ({len(rows)} films)")
    print("watch any film with its  *.movienet.srt (ground truth) or *.srt (our prediction)")


if __name__ == "__main__":
    main()
