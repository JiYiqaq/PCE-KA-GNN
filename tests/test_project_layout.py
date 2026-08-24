import hashlib
import unittest
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_DATA_SHA256 = (
    "218E2034F895682505815C60AD42E14B0E24F1F1E11740834403660BB3F0487F"
)


class ProjectLayoutTests(unittest.TestCase):
    def test_default_config_uses_the_bundled_source_dataset(self):
        config_path = PROJECT_DIR / "config" / "pce.yaml"
        self.assertTrue(config_path.is_file(), "default PCE config must be bundled")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["data_path"], "data/raw/Active_Database.csv")
        data_path = PROJECT_DIR / config["data_path"]
        self.assertTrue(data_path.is_file(), "source dataset must be bundled")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_DATA_SHA256)

    def test_tracked_configs_do_not_reference_the_original_computer(self):
        config_dir = PROJECT_DIR / "config"
        self.assertTrue(config_dir.is_dir(), "config directory must exist")
        for config_path in config_dir.glob("*.yaml"):
            text = config_path.read_text(encoding="utf-8")
            self.assertNotIn("G:/", text)
            self.assertNotIn("D:/PychramProject/KA-GNN-main", text)


if __name__ == "__main__":
    unittest.main()
