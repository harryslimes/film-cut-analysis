"""S3-F1 A3.3 + S3-F2 gap 3: batch_score's cache reader accepts ONLY its own tagged
score-cache or a narrow untagged legacy {cuts, fps}, and rejects v2 cut-events
documents / unknown formats. BOTH forms require a valid fps -- a missing fps is
rejected, never defaulted to 24.0. Executes the real load_score_cache (stdlib).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from score_cache import SCORE_CACHE_FORMAT, load_score_cache  # noqa: E402


class TestAccepts(unittest.TestCase):
    def test_tagged_score_cache(self):
        d = {"format": SCORE_CACHE_FORMAT, "cuts": [1.0, 2.0], "fps": 25.0}
        self.assertEqual(load_score_cache(d), ([1.0, 2.0], 25.0))

    def test_narrow_legacy_cuts_and_fps(self):
        self.assertEqual(load_score_cache({"cuts": [1.0], "fps": 30.0}), ([1.0], 30.0))


class TestRejectsFormat(unittest.TestCase):
    def test_v2_cut_events_document_rejected(self):
        # a v2 export also has cuts/fps at top level -- must NOT be read as a cache
        d = {"format": "film-cut-analysis/cut-events", "cuts": [1.0, 2.0], "fps": 25.0}
        self.assertRaises(ValueError, load_score_cache, d)

    def test_unknown_format_rejected(self):
        self.assertRaises(ValueError, load_score_cache,
                          {"format": "something/else", "cuts": [1.0], "fps": 25.0})

    def test_untagged_with_extra_keys_rejected(self):
        # legacy must be NARROW: a sync-remapped gt.json has extra keys -> reject
        self.assertRaises(ValueError, load_score_cache,
                          {"cuts": [1.0], "fps": 25.0, "sync": {}})

    def test_missing_cuts_rejected(self):
        self.assertRaises(ValueError, load_score_cache, {"format": SCORE_CACHE_FORMAT, "fps": 25.0})

    def test_non_dict_rejected(self):
        self.assertRaises(ValueError, load_score_cache, [1.0, 2.0])


class TestRejectsMissingOrInvalidFps(unittest.TestCase):
    """gap 3: fps is never invented -- missing or invalid fps is a rejection."""
    def test_legacy_missing_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache, {"cuts": [1.0]})

    def test_tagged_missing_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache,
                          {"format": SCORE_CACHE_FORMAT, "cuts": [1.0]})

    def test_bool_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache,
                          {"format": SCORE_CACHE_FORMAT, "cuts": [1.0], "fps": True})

    def test_zero_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache, {"cuts": [1.0], "fps": 0})

    def test_negative_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache, {"cuts": [1.0], "fps": -25.0})

    def test_nonnumeric_fps_rejected(self):
        self.assertRaises(ValueError, load_score_cache, {"cuts": [1.0], "fps": "25"})


class TestWriterIncludesFps(unittest.TestCase):
    """gap 3: batch_score's own cache writer must always include fps, so no self-written
    cache becomes unreadable under the stricter reader."""
    def test_batch_cache_writer_writes_fps(self):
        with open(os.path.join(ROOT, "batch_score.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("SCORE_CACHE_FORMAT", src)
        self.assertIn('"fps": r.fps_source', src)


if __name__ == "__main__":
    unittest.main()
