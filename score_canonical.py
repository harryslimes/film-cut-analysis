"""Score a detector's cuts against a region-canonical ground truth (e.g. Vertigo nightmare).

Only cuts inside the canonical's [region] are scored, so you can test any detector on the
hard sequence. Feed cuts from a JSON ({"cuts":[...]}) or run a registered detector live.

  # score a precomputed cut list
  python score_canonical.py --gt groundtruth/vertigo_nightmare.canonical.json --cuts mycuts.json
  # or run a detector (e.g. the motion-vector one) on the movie and score it
  python score_canonical.py --gt groundtruth/vertigo_nightmare.canonical.json \
      --video "data/movies/.../Vertigo ... .mkv" --detector motion-vectors --tol 0.4
"""
import argparse, json
from evaluate import match


def in_region(cuts, a, b):
    return sorted(c for c in cuts if a <= c <= b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="region-canonical json with region+cuts")
    ap.add_argument("--cuts", help="json with a 'cuts' list to score")
    ap.add_argument("--video", help="run a detector on this video instead")
    ap.add_argument("--detector", default="transnet-cuda")
    ap.add_argument("--tol", type=float, default=0.4)
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    lo, hi = gt["region"]
    ref = in_region(gt["cuts"], lo, hi)

    if a.cuts:
        cuts = json.load(open(a.cuts))["cuts"]
    elif a.video:
        from detectors import REGISTRY
        cuts = REGISTRY[a.detector](a.video).cuts
    else:
        raise SystemExit("give --cuts or --video")
    pred = in_region(cuts, lo, hi)

    m = match(pred, ref, tol=a.tol)
    print(f"canonical: {gt.get('movie','?')} [{gt['sequence']}]  region {lo}-{hi}s  "
          f"{len(ref)} cuts  (tol ±{a.tol}s)")
    print(f"detector : {len(pred)} cuts in region")
    print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}"
          f"   (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
    print(f"\n  reference: {[round(x,1) for x in ref]}")
    print(f"  detected : {[round(x,1) for x in pred]}")
    print(f"\nnote: {gt.get('note','')}")


if __name__ == "__main__":
    main()
