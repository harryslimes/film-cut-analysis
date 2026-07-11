"""Score predicted cuts against ground truth with a temporal tolerance window."""
from __future__ import annotations


def match(pred, gt, tol=0.5):
    """Greedy 1-1 matching of predicted cut times to ground-truth cut times.

    Returns dict with tp, fp, fn, precision, recall, f1 and the matched pairs.
    A prediction within `tol` seconds of an unused GT cut counts as a true positive.
    """
    gt_used = [False] * len(gt)
    tp = 0
    fp = 0
    matched = []
    for p in sorted(pred):
        best, bestd = -1, tol + 1e-9
        for j, g in enumerate(gt):
            if gt_used[j]:
                continue
            d = abs(p - g)
            if d <= bestd:
                best, bestd = j, d
        if best >= 0:
            gt_used[best] = True
            tp += 1
            matched.append((p, gt[best]))
        else:
            fp += 1
    fn = gt_used.count(False)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "matched": matched}


def trap_hits(pred, traps, tol=0.5):
    """Predicted cuts landing INSIDE a trap shot (light-on / whip-pan) = false positives.

    We count a prediction that falls in the shot interior, i.e. more than `tol` from
    either real boundary, so a legitimately-detected cut at the shot edge doesn't count.
    """
    hits = []
    for p in pred:
        for tr in traps:
            s, e = tr.get("start"), tr.get("end")
            if s is None:                       # legacy point-trap
                if abs(p - tr["time"]) <= tol:
                    hits.append({"type": tr["type"], "time": round(p, 3)})
                    break
            elif s + tol < p < e - tol:
                hits.append({"type": tr["type"], "time": round(p, 3)})
                break
    return hits
