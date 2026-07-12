"""S3-F1 A3.2: negative tests for every validator hole the audit demonstrated.
One validation path (validate_document) guards both write and read. Stdlib only.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import Confidence, CutEvent, ExportValidationError, Span  # noqa: E402
from cut_export import build_document, serialize, validate_document  # noqa: E402


def _base_doc():
    """A valid v2 document via the real write path, to mutate per test."""
    events = [
        CutEvent(time=1.0, frame=25, transition_kind="hard",
                 confidence=Confidence(0.9, "transnet_prob")),
        CutEvent(time=2.0, frame=50, transition_kind="gradual",
                 span=Span(1.9, 2.1), confidence=Confidence(0.4, "transnet_gradual_prob")),
    ]
    return build_document(
        events,
        source={"path": "m.mkv", "size_bytes": 10},
        run={"detector_id": "transnetv2", "backend": "cuda"},
        analysis={"status": "complete", "coverage": [{"start": 0.0, "end": 100.0}], "warnings": []},
        fps=25.0, video="m.mkv", detector="transnetv2[cuda]")


class TestBaselineValid(unittest.TestCase):
    def test_base_doc_is_valid(self):
        validate_document(_base_doc())            # no raise


class TestNoNaN(unittest.TestCase):
    def test_nan_event_time(self):
        d = _base_doc(); d["cut_events"][0]["time"] = float("nan"); d["cuts"][0] = float("nan")
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_inf_in_source(self):
        d = _base_doc(); d["source"]["size_bytes"] = float("inf")
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_nan_in_confidence(self):
        d = _base_doc(); d["cut_events"][0]["confidence"]["value"] = float("nan")
        self.assertRaises(ExportValidationError, validate_document, d)


class TestNoNullsAtAnyDepth(unittest.TestCase):
    def test_null_in_source(self):
        d = _base_doc(); d["source"]["duration_seconds"] = None
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_null_event_frame(self):
        d = _base_doc(); d["cut_events"][0]["frame"] = None
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_null_in_nested_run_settings(self):
        d = _base_doc(); d["run"]["settings"] = {"threshold": None}
        self.assertRaises(ExportValidationError, validate_document, d)


class TestNoSilentRepairOnRead(unittest.TestCase):
    def test_missing_time(self):
        d = _base_doc(); del d["cut_events"][0]["time"]
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_missing_transition_kind(self):
        d = _base_doc(); del d["cut_events"][0]["transition_kind"]
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_missing_confidence_metric(self):
        d = _base_doc(); del d["cut_events"][0]["confidence"]["metric"]
        self.assertRaises(ExportValidationError, validate_document, d)


class TestSpanOnlyOnGradual(unittest.TestCase):
    def test_span_on_hard_rejected(self):
        d = _base_doc(); d["cut_events"][0]["span"] = {"start": 0.9, "end": 1.1}  # event 0 is hard
        self.assertRaises(ExportValidationError, validate_document, d)


class TestFlagsWhitelist(unittest.TestCase):
    def test_documented_flag_accepted(self):
        d = _base_doc(); d["cut_events"][0]["flags"] = ["strobe_suppressed"]
        validate_document(d)                       # no raise

    def test_unknown_flag_rejected(self):
        d = _base_doc(); d["cut_events"][0]["flags"] = ["mystery_flag"]
        self.assertRaises(ExportValidationError, validate_document, d)


class TestCoverageOrder(unittest.TestCase):
    def test_unordered_coverage_rejected(self):
        d = _base_doc(); d["analysis"]["coverage"] = [{"start": 100.0, "end": 0.0}]
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_zero_width_coverage_rejected(self):
        d = _base_doc(); d["analysis"]["coverage"] = [{"start": 5.0, "end": 5.0}]
        self.assertRaises(ExportValidationError, validate_document, d)


class TestBooleansTypeChecked(unittest.TestCase):
    def test_non_bool_higher_is_stronger(self):
        d = _base_doc(); d["cut_events"][0]["confidence"]["higher_is_stronger"] = 1
        self.assertRaises(ExportValidationError, validate_document, d)

    def test_bool_frame_rejected(self):
        d = _base_doc(); d["cut_events"][0]["frame"] = True
        self.assertRaises(ExportValidationError, validate_document, d)


class TestDuplicateTimeOnRead(unittest.TestCase):
    def test_duplicate_time_rejected(self):
        d = _base_doc()
        d["cut_events"][1]["time"] = 1.0
        d["cuts"] = [1.0, 1.0]
        self.assertRaises(ExportValidationError, validate_document, d)


class TestOnePathWriteEqualsRead(unittest.TestCase):
    def test_build_document_rejects_at_write(self):
        # the same rule (span-only-on-gradual) that read enforces must fire on WRITE
        with self.assertRaises(ExportValidationError):
            build_document(
                [CutEvent(time=1.0, transition_kind="hard", span=Span(0.9, 1.1))],
                source={"path": "m"}, run={},
                analysis={"status": "complete", "coverage": [{"start": 0.0, "end": 10.0}]},
                fps=25.0, video="m", detector="x")


class TestSerializeAllowNan(unittest.TestCase):
    def test_serialized_json_has_no_nan_tokens(self):
        s = serialize(
            [CutEvent(time=1.0, transition_kind="hard", confidence=Confidence(0.9, "m"))],
            source={"path": "m"}, run={},
            analysis={"status": "complete", "coverage": [{"start": 0.0, "end": 10.0}]},
            fps=25.0, video="m", detector="x")
        self.assertNotIn("NaN", s)
        self.assertNotIn("Infinity", s)


if __name__ == "__main__":
    unittest.main()
