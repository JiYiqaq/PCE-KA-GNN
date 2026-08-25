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

    def test_all_run_configs_require_cuda_and_share_the_pair_cache(self):
        config_dir = PROJECT_DIR / "config"
        expected_cache_paths = {
            "pce.yaml": "data/processed/canonical_pairs.csv",
            "pce_quick.yaml": "data/processed/canonical_pairs.csv",
            "pce_smoke.yaml": "data/processed/pce_smoke_pairs.csv",
        }
        for config_path in config_dir.glob("*.yaml"):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["device"], "cuda", config_path.name)
            self.assertEqual(
                config["prepared_pairs_cache_path"],
                expected_cache_paths[config_path.name],
                config_path.name,
            )

    def test_gpu_requirements_pin_the_verified_production_stack(self):
        gpu_requirements_path = PROJECT_DIR / "requirements-gpu.txt"
        self.assertTrue(
            gpu_requirements_path.is_file(),
            "the verified CUDA environment must have a reproducible requirements file",
        )
        gpu_requirements = gpu_requirements_path.read_text(encoding="utf-8")
        common_requirements = (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("torch==2.1.2+cu118", gpu_requirements)
        self.assertIn("dgl==2.2.1+cu118", gpu_requirements)
        self.assertIn("-r requirements.txt", gpu_requirements)
        self.assertIn("torchdata==0.7.1", common_requirements)
        self.assertIn("pydantic==1.10.26", common_requirements)
        self.assertIn("psutil==7.0.0", common_requirements)
        self.assertNotIn("dgl==2.0.0", common_requirements)

    def test_readme_documents_gpu_as_the_only_formal_runtime(self):
        readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")

        self.assertIn("pce_kagnn_gpu", readme)
        self.assertIn("requirements-gpu.txt", readme)
        self.assertIn("2.1.2+cu118", readme)
        self.assertIn("2.2.1+cu118", readme)
        self.assertNotIn("CPU 版 PyTorch", readme)


if __name__ == "__main__":
    unittest.main()
