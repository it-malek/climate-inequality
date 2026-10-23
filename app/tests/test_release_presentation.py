"""Publication labels must follow verified release context, not execution-stage flags."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import loaders
from src.snapshot import write_manifest


class ReleasePresentationTest(unittest.TestCase):
    def test_released_caption_preserves_execution_metadata_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {"explain": {"correction_set_id": "recorded-correction",
                                   "computation_status": "prospective-development"}}
            asset = root / "stats.json"
            asset.write_text(json.dumps(summary))
            before = asset.read_bytes()
            write_manifest(root, version="2.1.0")
            with patch.object(loaders, "APP_DATA_DIR", root):
                self.assertEqual(loaders.release_caption("stats.json", section="explain"),
                                 "Released analysis 2.1.0.")
                self.assertEqual(asset.read_bytes(), before)
                with self.assertRaisesRegex(ValueError, "not covered"):
                    loaders.release_caption("outside.json")
                asset.write_text(json.dumps({"explain": {}}))
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    loaders.release_caption("stats.json", section="explain")

    def test_historical_summary_is_not_reclassified_as_corrected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stats.json").write_text('{"explain": {"historical": true}}')
            write_manifest(root, version="2.1.0")
            with patch.object(loaders, "APP_DATA_DIR", root):
                self.assertEqual(loaders.release_caption("stats.json", section="explain"),
                                 "Historical saved summary included in release 2.1.0.")


if __name__ == "__main__":
    unittest.main()
