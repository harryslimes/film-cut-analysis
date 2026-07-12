"""Unit tests for the v2 serializer (cut_export.py) -- design §3 / §3.3,
including the committed fixture pair (§3.3 rule 6). Stdlib only."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import Confidence, CutEvent, ExportValidationError, Span  # noqa: E402
from cut_export import (  # noqa: E402
    FORMAT, SCHEMA_VERSION, build_document, serialize, validate_document,
)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _meta(**over):
    """Minimal source/run/analysis kwargs for build_document; override as needed."""
    kw = dict(
        source={"path": "D:\\Films\\movie.mkv", "size_bytes": 10, "mtime_utc": "2026-07-12T09:14:22Z"},
        run={"detector_id": "torch", "backend": "cpu", "generated_by": "cut_times.py"},
        analysis={"status": "complete", "coverage": [{"start": 0.0, "end": 1000.0}], "warnings": []},
        fps=24.0, video="movie.mkv", detector="torch-gpu[cpu]",
    )
    kw.update(over)
    return kw


class TestBuildDocument(unittest.TestCase):
    def test_shape_and_discriminators(self):
        evs = [CutEvent(time=12.345, frame=296, transition_kind="hard",
                        confidence=Confidence(0.98, "transnet_prob"))]
        doc = build_document(evs, **_meta())
        self.assertEqual(doc["format"], FORMAT)
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        # trailing legacy keys retained verbatim (§3)
        self.assertEqual((doc["fps"], doc["video"], doc["detector"]), (24.0, "movie.mkv", "torch-gpu[cpu]"))
        self.assertEqual(doc["analysis"]["event_count"], 1)          # derived, not trusted
        self.assertEqual(doc["analysis"]["status"], "complete")

    def test_cuts_is_generated_projection(self):
        evs = [CutEvent(time=1.0), CutEvent(time=2.0), CutEvent(time=3.5)]
        doc = build_document(evs, **_meta())
        self.assertEqual(doc["cuts"], [1.0, 2.0, 3.5])
        self.assertEqual(doc["cuts"], [ce["time"] for ce in doc["cut_events"]])

    def test_source_omits_absent_duration(self):
        # duration_seconds not supplied -> key must be absent, never null (§3.2)
        doc = build_document([CutEvent(time=1.0)], **_meta())
        self.assertNotIn("duration_seconds", doc["source"])
        self.assertNotIn(None, doc["source"].values())

    def test_source_keeps_duration_when_known(self):
        m = _meta()
        m["source"] = dict(m["source"], duration_seconds=1000.0)
        doc = build_document([CutEvent(time=1.0)], **m)
        self.assertEqual(doc["source"]["duration_seconds"], 1000.0)

    def test_bad_status_rejected(self):
        with self.assertRaises(ExportValidationError):
            build_document([CutEvent(time=1.0)], **_meta(analysis={"status": "done", "coverage": []}))

    def test_build_enforces_event_rules(self):
        # out-of-coverage time bubbles up through build_document
        with self.assertRaises(ExportValidationError):
            build_document([CutEvent(time=5000.0)], **_meta())

    def test_serialize_returns_valid_json_roundtrips_through_validator(self):
        s = serialize([CutEvent(time=1.0, transition_kind="hard")], **_meta())
        validate_document(json.loads(s))          # should not raise


class TestValidateDocument(unittest.TestCase):
    def test_wrong_format_rejected(self):
        doc = build_document([CutEvent(time=1.0)], **_meta())
        doc["format"] = "something/else"
        with self.assertRaises(ExportValidationError):
            validate_document(doc)

    def test_unsupported_version_rejected(self):
        doc = build_document([CutEvent(time=1.0)], **_meta())
        doc["schema_version"] = 99
        with self.assertRaises(ExportValidationError):
            validate_document(doc)

    def test_tampered_projection_rejected(self):
        doc = build_document([CutEvent(time=1.0), CutEvent(time=2.0)], **_meta())
        doc["cuts"] = [1.0, 2.5]                   # hand-edited away from the events
        with self.assertRaises(ExportValidationError):
            validate_document(doc)

    def test_event_count_mismatch_rejected(self):
        doc = build_document([CutEvent(time=1.0)], **_meta())
        doc["analysis"]["event_count"] = 7
        with self.assertRaises(ExportValidationError):
            validate_document(doc)


class TestFixturePair(unittest.TestCase):
    """§3.3 rule 6: the committed fixtures are executable examples."""
    def test_valid_fixture_passes(self):
        with open(os.path.join(FIX, "valid_cut_export.json")) as fh:
            validate_document(json.load(fh))      # should not raise

    def test_invalid_fixture_rejected(self):
        with open(os.path.join(FIX, "invalid_cut_export.json")) as fh:
            doc = json.load(fh)
        with self.assertRaises(ExportValidationError):
            validate_document(doc)


if __name__ == "__main__":
    unittest.main()
