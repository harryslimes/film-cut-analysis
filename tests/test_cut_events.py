"""Unit tests for the event core (cut_events.py) -- design §3.2 / §3.3.
Stdlib only; importable and runnable without torch/opencv/scenedetect."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import (  # noqa: E402
    Confidence, CutEvent, ExportValidationError, Span, event_from_dict,
    event_to_dict, project_cuts, validate_events, wrap_times,
)

COVERAGE = [{"start": 0.0, "end": 1000.0}]


class TestProjection(unittest.TestCase):
    def test_project_cuts_is_event_order(self):
        evs = [CutEvent(time=1.0), CutEvent(time=2.5), CutEvent(time=9.0)]
        self.assertEqual(project_cuts(evs), [1.0, 2.5, 9.0])

    def test_wrap_times_makes_minimal_unknown_events(self):
        evs = wrap_times([3.0, 4.0])
        self.assertEqual([e.transition_kind for e in evs], ["unknown", "unknown"])
        self.assertTrue(all(e.frame is None and e.confidence is None and e.span is None for e in evs))
        self.assertEqual(project_cuts(evs), [3.0, 4.0])


class TestEventToDictOmission(unittest.TestCase):
    def test_minimal_event_omits_absent_fields_never_null(self):
        d = event_to_dict(CutEvent(time=12.0))
        self.assertEqual(d, {"time": 12.0, "transition_kind": "unknown"})
        for absent in ("frame", "span", "confidence", "flags"):
            self.assertNotIn(absent, d)          # omission, not null (§3.2)

    def test_rich_event_serializes_all_present_fields_in_order(self):
        e = CutEvent(time=245.1, frame=5882, transition_kind="gradual",
                     span=Span(244.6, 245.71),
                     confidence=Confidence(0.41, "transnet_gradual_prob"),
                     flags=["strobe_suppressed"])
        d = event_to_dict(e)
        self.assertEqual(list(d.keys()),
                         ["time", "frame", "transition_kind", "span", "confidence", "flags"])
        self.assertEqual(d["span"], {"start": 244.6, "end": 245.71})
        self.assertEqual(d["confidence"],
                         {"value": 0.41, "metric": "transnet_gradual_prob", "higher_is_stronger": True})
        self.assertEqual(d["flags"], ["strobe_suppressed"])

    def test_empty_flags_are_omitted(self):
        self.assertNotIn("flags", event_to_dict(CutEvent(time=1.0, flags=[])))

    def test_roundtrip_from_dict(self):
        e = CutEvent(time=1.0, frame=24, transition_kind="hard",
                     confidence=Confidence(3.2, "hsv_spike_ratio"))
        self.assertEqual(event_to_dict(event_from_dict(event_to_dict(e))), event_to_dict(e))


class TestValidateEvents(unittest.TestCase):
    def test_valid_sequence_passes(self):
        evs = [CutEvent(time=1.0, transition_kind="hard"),
               CutEvent(time=2.0, transition_kind="gradual", span=Span(1.9, 2.2))]
        validate_events(evs, coverage=COVERAGE)   # should not raise

    def test_negative_time_rejected(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=-0.1)])

    def test_non_finite_time_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ExportValidationError):
                validate_events([CutEvent(time=bad)])

    def test_bool_is_not_a_valid_time(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=True)])

    def test_duplicate_time_rejected(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=5.0), CutEvent(time=5.0)])

    def test_out_of_order_rejected(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=5.0), CutEvent(time=4.0)])

    def test_bad_transition_kind_rejected(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=1.0, transition_kind="dissolve")])

    def test_span_sandwich_enforced(self):
        # time outside [start, end]
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=3.0, transition_kind="gradual", span=Span(1.0, 2.0))])
        # boundary-inclusive is fine
        validate_events([CutEvent(time=2.0, transition_kind="gradual", span=Span(2.0, 2.0))])

    def test_confidence_requires_metric(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=1.0, confidence=Confidence(0.9, ""))])

    def test_confidence_value_must_be_finite(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=1.0, confidence=Confidence(float("nan"), "m"))])

    def test_frame_must_be_nonnegative_int(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=1.0, frame=-1)])
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=1.0, frame=1.5)])

    def test_time_outside_coverage_rejected(self):
        with self.assertRaises(ExportValidationError):
            validate_events([CutEvent(time=2000.0)], coverage=COVERAGE)

    def test_no_coverage_skips_bounds_check(self):
        validate_events([CutEvent(time=2000.0)])   # no coverage supplied -> only >= 0 enforced


if __name__ == "__main__":
    unittest.main()
