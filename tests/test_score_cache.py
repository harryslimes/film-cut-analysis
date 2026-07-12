"""S3-F1 A3.3: batch_score's cache reader accepts ONLY its own tagged score-cache
or a narrow untagged legacy {cuts, fps}; it REJECTS v2 cut-events documents and any
other/unknown format. Executes the real load_score_cache (stdlib). Replaces the old
test that wrongly asserted v2 documents were acceptable as cache.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_cache import SCORE_CACHE_FORMAT, load_score_cache  # noqa: E402


class TestAccepts(unittest.TestCase):
    def test_tagged_score_cache(self):
        d = {"format": SCORE_CACHE_FORMAT, "cuts": [1.0, 2.0], "fps": 25.0}
        self.assertEqual(load_score_cache(d), ([1.0, 2.0], 25.0))

    def test_narrow_legacy_cuts_and_fps(self):
        self.assertEqual(load_score_cache({"cuts": [1.0], "fps": 30.0}), ([1.0], 30.0))

    def test_narrow_legacy_cuts_only_defaults_fps(self):
        self.assertEqual(load_score_cache({"cuts": [1.0]}), ([1.0], 24.0))


class TestRejects(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
