"""Per-detector event construction -- the dependency-light layer (CineScript S3-F1,
amendment A3.5). Stdlib only (imports cut_events and, for transnet, postfilter): no
torch / numpy / av / scenedetect. Each detector's heavy compute produces PLAIN data
(lists of floats / ints), then calls the matching builder here; the tests import and
execute these EXACT functions, so the event-construction logic is pinned, not copied.

Two amendment rulings live here:
- A1 dedupe: every builder does deterministic FIRST-WINS dedupe on the (rounded) time
  before emitting, so a collision (rounding / encoder stutter) keeps the first-emitted
  candidate's metadata. The serializer's duplicate-time rejection is now a backstop.
- A2 frame: `frame` is the FIRST FRAME OF THE NEW SHOT (the frame after the transition
  boundary), zero-based, in the detector's own numbering. torch records i + 1 here.
"""
from __future__ import annotations

from cut_events import Confidence, CutEvent, Span


def _first_wins(pairs):
    """Deterministic first-wins dedupe: given (key, value) pairs in emission order,
    keep the value of the FIRST occurrence of each key. Returns an insertion-ordered
    dict (so callers can sort the keys for output)."""
    out = {}
    for k, v in pairs:
        out.setdefault(k, v)
    return out


def build_minimal_events(times):
    """ffmpeg / psd: bare timestamps -> minimal events (transition_kind 'unknown',
    everything else omitted). First-wins dedupe on exact time, then sorted."""
    by_time = _first_wins((t, CutEvent(time=t)) for t in times)
    return [by_time[t] for t in sorted(by_time)]


def ffmpeg_decode_extra(returncode, stderr):
    """ffmpeg-scene decode diagnostics from its exit code + stderr (S3-F2 gap 2):
    {} on a clean exit; otherwise decode_ok False with a detail that carries the stderr
    tail when NO usable showinfo output survived (an unusable decode -> empty events ->
    the exporter reports status 'failed'; a nonzero exit that still emitted showinfo ->
    'partial'). Dependency-light so the branch is executed by tests, not just reviewed."""
    if returncode == 0:
        return {}
    detail = f"ffmpeg-scene exited {returncode}"
    if "showinfo" not in stderr:
        detail += f"; unusable decode:\n{stderr[-1500:]}"
    return {"decode_ok": False, "decode_detail": detail}


def build_torch_events(accepts, fps):
    """torch: accepts = [(i, spike), ...] in emission order (ascending i), where i is
    the score-array index of the accepted transition and spike is the native
    hsv_spike_ratio. Emits time = round(i/fps, 4) (canonical, unchanged) and
    frame = i + 1 (A2: the first frame of the new shot -- scores[i] is the change INTO
    frame i+1). First-wins dedupe on rounded time; sorted."""
    def _ev(i, spike):
        t = round(i / fps, 4)
        return t, CutEvent(time=t, frame=i + 1, transition_kind="unknown",
                           confidence=Confidence(float(spike), "hsv_spike_ratio",
                                                 higher_is_stronger=True))
    by_time = _first_wins(_ev(i, spike) for i, spike in accepts)
    return [by_time[t] for t in sorted(by_time)]


def build_mv_events(kept):
    """motion-vectors: kept = [(pts_seconds, iframe_index, score), ...] in emission order
    (ascending), for I-frames whose combined score cleared `keep`. time = round(pts, 3)
    (PTS, not frame/fps), frame = the I-frame index (already the first frame of the new
    shot), confidence = native mv_score. First-wins dedupe on rounded time; sorted."""
    def _ev(pts, idx, score):
        t = round(float(pts), 3)
        return t, CutEvent(time=t, frame=int(idx), transition_kind="unknown",
                           confidence=Confidence(float(score), "mv_score",
                                                 higher_is_stronger=True))
    by_time = _first_wins(_ev(pts, idx, score) for pts, idx, score in kept)
    return [by_time[t] for t in sorted(by_time)]


def gradual_span_frames(prob, pf, height):
    """Measured extent of a gradual transition: the maximal contiguous run of frames
    around peak `pf` whose all-frames probability stays >= `height` -- the same signal
    and threshold that located the peak, so this is honest extent, not a span invented
    from a point. Returns (lo_frame, hi_frame) with lo <= pf <= hi (pf is itself >=
    height). Callers omit the span when lo == hi (a single-frame run has no extent).
    Works on any indexable `prob` (list or numpy array) -- kept dependency-light."""
    lo = pf
    while lo - 1 >= 0 and prob[lo - 1] >= height:
        lo -= 1
    hi = pf
    while hi + 1 < len(prob) and prob[hi + 1] >= height:
        hi += 1
    return lo, hi


def build_transnet_events(hard, gradual, fps, *, strobe_guard=True, strobe_params=None):
    """transnet: hard = [(frame, single_prob), ...] and gradual =
    [(frame, allf_prob, span_lo_frame, span_hi_frame), ...]. Emits a "hard" event per
    sharp cut (metric transnet_prob) and a "gradual" event per dissolve peak (metric
    transnet_gradual_prob, with a span when the run spans >1 frame). Hard candidates are
    emitted before gradual ones, so on a rounded-time collision the hard cut wins
    (first-wins dedupe, A1). Strobe suppression (postfilter.dampen_strobe -- stdlib) then
    thins pathologically dense regions. Returns (events, n_strobe_removed)."""
    def _hard(f, prob):
        t = round(f / fps, 4)
        return t, CutEvent(time=t, frame=int(f), transition_kind="hard",
                           confidence=Confidence(float(prob), "transnet_prob",
                                                 higher_is_stronger=True))

    def _gradual(f, prob, lo_f, hi_f):
        t = round(f / fps, 4)
        span = Span(round(lo_f / fps, 4), round(hi_f / fps, 4)) if hi_f > lo_f else None
        return t, CutEvent(time=t, frame=int(f), transition_kind="gradual",
                           confidence=Confidence(float(prob), "transnet_gradual_prob",
                                                 higher_is_stronger=True), span=span)

    cands = [_hard(f, p) for f, p in hard] + [_gradual(f, p, lo, hi) for f, p, lo, hi in gradual]
    by_time = _first_wins(cands)
    cut_times = sorted(by_time)
    n_before = len(cut_times)
    if strobe_guard:
        from postfilter import dampen_strobe
        kept = dampen_strobe(cut_times, **(strobe_params or {}))
    else:
        kept = cut_times
    events = [by_time[t] for t in sorted(kept)]
    return events, n_before - len(kept)
