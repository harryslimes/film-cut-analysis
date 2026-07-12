"""S3-E3: the enriched event shape torch_gpu now emits at its accept point
(design §3.2, §5 S3-E3). Torch-free: we build the CutEvent exactly the way
torch_gpu's loop does, with synthetic score values -- no torch, no video. The
real import/runtime path is source-reviewed only (torch/ffmpeg required).

torch_gpu's accept point does, per cut frame i:
    spike = float(s[i] / (local + 1e-6))
    CutEvent(time=round(i/fps, 4), frame=i,
             confidence=Confidence(spike, "hsv_spike_ratio", higher_is_stronger=True))
No transition_kind (defaults "unknown"), no span, no flags.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import (  # noqa: E402
    Confidence, CutEvent, event_to_dict, validate_events,
)


def _torch_event(i, s_i, local, fps):
    """Reproduce torch_gpu.py's accept-point construction verbatim."""
    spike = float(s_i / (local + 1e-6))
    return CutEvent(time=round(i / fps, 4), frame=i,
                    confidence=Confidence(spike, "hsv_spike_ratio", higher_is_stronger=True))


class TestTorchEventShape(unittest.TestCase):
    def test_metric_name_and_kind(self):
        e = _torch_event(i=300, s_i=0.20, local=0.05, fps=25.0)
        self.assertEqual(e.confidence.metric, "hsv_spike_ratio")
        self.assertTrue(e.confidence.higher_is_stronger)
        self.assertEqual(e.transition_kind, "unknown")   # never a guessed "hard" (§3.2)
        self.assertIsNone(e.span)
        self.assertIsNone(e.flags)

    def test_native_value_passthrough_not_normalised(self):
        # spike ratio ~ 0.20 / (0.05 + 1e-6) = 3.999...; stored raw, NOT clamped to [0,1]
        e = _torch_event(i=300, s_i=0.20, local=0.05, fps=25.0)
        self.assertAlmostEqual(e.confidence.value, 0.20 / (0.05 + 1e-6))
        self.assertGreater(e.confidence.value, 1.0)       # proves no [0,1] normalisation

    def test_frame_is_int_and_time_matches_detector_formula(self):
        e = _torch_event(i=296, s_i=0.3, local=0.02, fps=24.0)
        self.assertIsInstance(e.frame, int)
        self.assertEqual(e.frame, 296)
        self.assertEqual(e.time, round(296 / 24.0, 4))

    def test_local_zero_stays_finite_and_valid(self):
        # neighbourhood median can be 0.0; (local + 1e-6) keeps the ratio finite
        e = _torch_event(i=10, s_i=0.05, local=0.0, fps=25.0)
        self.assertEqual(e.confidence.value, 0.05 / 1e-6)   # 50000.0, finite
        validate_events([e])                                # §3.3 accepts it

    def test_events_validate_and_serialize_minimally(self):
        evs = [_torch_event(i, s_i, local, 25.0)
               for i, s_i, local in [(10, 0.06, 0.01), (40, 0.5, 0.03), (200, 0.12, 0.0)]]
        validate_events(evs)                                # ascending, finite, valid
        d = event_to_dict(evs[0])
        self.assertEqual(set(d), {"time", "frame", "transition_kind", "confidence"})
        self.assertNotIn("span", d)
        self.assertNotIn("flags", d)


if __name__ == "__main__":
    unittest.main()
