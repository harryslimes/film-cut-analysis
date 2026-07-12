"""S3-E4: the enriched event shape transnet.py now emits (design §3.2, §5 S3-E4).
Torch-free: transnet.py does `import torch` at module top, so we reproduce its
accept-point construction with synthetic values -- no torch, no video. The real
import/runtime path (incl. the real _gradual_span_frames) is source-reviewed only.

Per accepted cut frame f, transnet builds:
  hard   : CutEvent(time=round(f/fps,4), frame=f, transition_kind="hard",
                    confidence=Confidence(float(preds[f]), "transnet_prob", True))
  gradual: CutEvent(time=round(f/fps,4), frame=f, transition_kind="gradual",
                    confidence=Confidence(float(allp[f]), "transnet_gradual_prob", True),
                    span=Span(round(lo/fps,4), round(hi/fps,4)) if hi > lo else None)
No flags on any event (see the slice's flags decision).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import (  # noqa: E402
    Confidence, CutEvent, Span, event_to_dict, validate_events,
)


def _span_frames(allp, pf, height):
    """Mirrors transnet.py._gradual_span_frames (kept in sync by review)."""
    lo = pf
    while lo - 1 >= 0 and allp[lo - 1] >= height:
        lo -= 1
    hi = pf
    while hi + 1 < len(allp) and allp[hi + 1] >= height:
        hi += 1
    return lo, hi


def _hard_event(f, prob, fps):
    return CutEvent(time=round(f / fps, 4), frame=f, transition_kind="hard",
                    confidence=Confidence(float(prob), "transnet_prob", higher_is_stronger=True))


def _gradual_event(f, allp, height, fps):
    lo, hi = _span_frames(allp, f, height)
    return CutEvent(time=round(f / fps, 4), frame=f, transition_kind="gradual",
                    confidence=Confidence(float(allp[f]), "transnet_gradual_prob", higher_is_stronger=True),
                    span=Span(round(lo / fps, 4), round(hi / fps, 4)) if hi > lo else None)


class TestHardEvents(unittest.TestCase):
    def test_hard_shape(self):
        e = _hard_event(f=296, prob=0.98, fps=24.0)
        self.assertEqual(e.transition_kind, "hard")
        self.assertEqual(e.confidence.metric, "transnet_prob")
        self.assertEqual(e.confidence.value, 0.98)          # native passthrough
        self.assertTrue(e.confidence.higher_is_stronger)
        self.assertEqual(e.frame, 296)
        self.assertIsInstance(e.frame, int)
        self.assertEqual(e.time, round(296 / 24.0, 4))
        self.assertIsNone(e.span)                            # never a span on a hard cut
        d = event_to_dict(e)
        self.assertNotIn("span", d)
        self.assertNotIn("flags", d)                         # flags decision: none emitted


class TestGradualEvents(unittest.TestCase):
    # allp hump above height=0.35 across frames 8..12, peak at 10
    ALLP = [0.0, 0.0, 0.1, 0.2, 0.1, 0.05, 0.0, 0.2, 0.40, 0.55, 0.70, 0.50, 0.38, 0.20, 0.1]
    HEIGHT = 0.35
    FPS = 25.0

    def test_gradual_shape_and_metric(self):
        e = _gradual_event(f=10, allp=self.ALLP, height=self.HEIGHT, fps=self.FPS)
        self.assertEqual(e.transition_kind, "gradual")
        self.assertEqual(e.confidence.metric, "transnet_gradual_prob")
        self.assertEqual(e.confidence.value, 0.70)          # allp[10], native passthrough
        self.assertTrue(e.confidence.higher_is_stronger)

    def test_span_covers_above_threshold_run_and_sandwiches_time(self):
        e = _gradual_event(f=10, allp=self.ALLP, height=self.HEIGHT, fps=self.FPS)
        # run of allp >= 0.35 containing frame 10 is frames 8..12
        self.assertEqual(e.span.start, round(8 / self.FPS, 4))
        self.assertEqual(e.span.end, round(12 / self.FPS, 4))
        self.assertLessEqual(e.span.start, e.time)          # sandwich (§3.3 rule 4)
        self.assertLessEqual(e.time, e.span.end)
        validate_events([e])                                 # passes span-sandwich check

    def test_single_frame_run_omits_span(self):
        # a lone above-threshold frame (neighbours below height) -> no measurable extent
        allp = [0.0, 0.0, 0.9, 0.0, 0.0]
        e = _gradual_event(f=2, allp=allp, height=0.35, fps=self.FPS)
        self.assertIsNone(e.span)
        self.assertNotIn("span", event_to_dict(e))

    def test_peak_at_right_edge_gives_end_equal_time(self):
        # rising hump whose peak is the last above-threshold frame -> span extends left
        allp = [0.0, 0.40, 0.55, 0.70, 0.10]   # >=0.35 run is 1..3, peak at 3
        e = _gradual_event(f=3, allp=allp, height=0.35, fps=self.FPS)
        self.assertEqual(e.span.end, e.time)                 # end == time is valid (<=)
        self.assertLess(e.span.start, e.time)
        validate_events([e])


class TestMixedStreamKindMapping(unittest.TestCase):
    def test_kinds_and_validity_across_streams(self):
        fps = 24.0
        preds = {24: 0.9, 120: 0.6}
        allp = [0.0] * 60 + [0.4, 0.5, 0.6, 0.5, 0.4] + [0.0] * 35  # hump ~ frames 60..64
        evs = sorted(
            [_hard_event(24, preds[24], fps), _hard_event(120, preds[120], fps),
             _gradual_event(62, allp, 0.35, fps)],
            key=lambda e: e.time)
        validate_events(evs)                                 # ascending, finite, spans valid
        kinds = [e.transition_kind for e in evs]
        self.assertEqual(kinds, ["hard", "gradual", "hard"])
        self.assertTrue(all("flags" not in event_to_dict(e) for e in evs))


if __name__ == "__main__":
    unittest.main()
