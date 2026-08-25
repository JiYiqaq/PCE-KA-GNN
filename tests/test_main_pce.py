import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


class MainPCETests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("main_pce"),
            "main_pce.py must provide the independent regression entry point",
        )
        return importlib.import_module("main_pce")

    def test_load_config_honors_the_supplied_path(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text("data_path: custom.csv\nepochs: 3\n", encoding="utf-8")

            config = main_pce.load_config(path)

        self.assertEqual(config["data_path"], "custom.csv")
        self.assertEqual(config["epochs"], 3)

    def test_load_pair_data_reuses_a_prepared_pair_table(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prepared_path = directory / "prepared.csv"
            pd.DataFrame(
                {
                    "donor_smiles": ["CCO"],
                    "acceptor_smiles": ["CCN"],
                    "pce": [7.5],
                }
            ).to_csv(prepared_path, index=False)

            pairs, audit = main_pce.load_pair_data(
                {"prepared_pairs_path": str(prepared_path)},
                directory / "output",
            )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(float(pairs.iloc[0]["pce"]), 7.5)
        self.assertEqual(audit["source"], "prepared_pairs")

    def test_load_pair_data_reuses_a_fingerprinted_automatic_cache(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw_path = directory / "raw.csv"
            cache_path = directory / "processed" / "canonical_pairs.csv"
            pd.DataFrame(
                {
                    "donor_smiles": ["CCO"],
                    "acceptor_smiles": ["CCN"],
                    "pce": [7.5],
                }
            ).to_csv(raw_path, index=False)
            config = {
                "data_path": str(raw_path),
                "prepared_pairs_cache_path": str(cache_path),
            }

            first_pairs, first_audit = main_pce.load_pair_data(config, directory / "out")
            second_pairs, second_audit = main_pce.load_pair_data(config, directory / "out")

            self.assertEqual(first_audit["source"], "raw_csv")
            self.assertEqual(second_audit["source"], "prepared_pairs_cache")
            self.assertTrue(cache_path.is_file())
            self.assertTrue(cache_path.with_suffix(".meta.json").is_file())
            pd.testing.assert_frame_equal(first_pairs, second_pairs)

    def test_load_pair_data_invalidates_cache_when_the_source_changes(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw_path = directory / "raw.csv"
            cache_path = directory / "processed" / "canonical_pairs.csv"
            config = {
                "data_path": str(raw_path),
                "prepared_pairs_cache_path": str(cache_path),
            }
            pd.DataFrame(
                {
                    "donor_smiles": ["CCO"],
                    "acceptor_smiles": ["CCN"],
                    "pce": [7.5],
                }
            ).to_csv(raw_path, index=False)
            main_pce.load_pair_data(config, directory / "out")

            pd.DataFrame(
                {
                    "donor_smiles": ["CCO", "CCC"],
                    "acceptor_smiles": ["CCN", "CCO"],
                    "pce": [7.5, 4.0],
                }
            ).to_csv(raw_path, index=False)
            pairs, audit = main_pce.load_pair_data(config, directory / "out")

        self.assertEqual(audit["source"], "raw_csv")
        self.assertEqual(len(pairs), 2)

    def test_automatic_pair_cache_uses_repository_line_endings(self):
        main_pce = self.load_module()
        pairs = pd.DataFrame(
            {
                "donor_smiles": ["CCO"],
                "acceptor_smiles": ["CCN"],
                "pce": [7.5],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "canonical_pairs.csv"
            main_pce.write_automatic_pair_cache(
                pairs,
                cache_path,
                {"version": 1},
                {"source": "raw_csv", "unique_pairs": 1},
            )
            cache_bytes = cache_path.read_bytes()

        self.assertIn(b"\n", cache_bytes)
        self.assertNotIn(b"\r\n", cache_bytes)

    def test_pair_data_status_identifies_the_validated_cache(self):
        main_pce = self.load_module()
        pairs = pd.DataFrame(
            {
                "donor_smiles": ["CCO"],
                "acceptor_smiles": ["CCN"],
                "pce": [7.5],
            }
        )

        message = main_pce.pair_data_status_message(
            pairs,
            {"source": "prepared_pairs_cache"},
        )

        self.assertIn("Loaded 1", message)
        self.assertIn("validated pair cache", message)

    def test_result_paths_inside_the_project_are_portable(self):
        main_pce = self.load_module()
        project_file = main_pce.PROJECT_DIR / "data" / "raw" / "Active_Database.csv"

        self.assertEqual(
            main_pce.portable_result_path(project_file),
            "data/raw/Active_Database.csv",
        )

    def test_audit_path_fields_are_made_portable(self):
        main_pce = self.load_module()
        cache_path = main_pce.PROJECT_DIR / "data" / "processed" / "canonical_pairs.csv"

        portable = main_pce.portable_audit_paths(
            {
                "source": "prepared_pairs_cache",
                "prepared_pairs_cache_path": str(cache_path),
                "unique_pairs": 5877,
            }
        )

        self.assertEqual(
            portable["prepared_pairs_cache_path"],
            "data/processed/canonical_pairs.csv",
        )
        self.assertEqual(portable["unique_pairs"], 5877)

    def test_resolve_device_refuses_non_cuda_configuration(self):
        main_pce = self.load_module()
        self.assertTrue(
            hasattr(main_pce, "resolve_device"),
            "the production entry point must expose an explicit CUDA resolver",
        )

        with self.assertRaisesRegex(ValueError, "device: cuda"):
            main_pce.resolve_device("cpu")

    def test_resolve_device_fails_instead_of_falling_back_without_cuda(self):
        main_pce = self.load_module()
        self.assertTrue(hasattr(main_pce, "resolve_device"))

        with mock.patch.object(main_pce.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                main_pce.resolve_device("cuda")

    def test_runtime_metadata_rejects_a_cpu_device(self):
        main_pce = self.load_module()
        self.assertTrue(
            hasattr(main_pce, "runtime_metadata"),
            "every result summary must include verified GPU runtime metadata",
        )

        with self.assertRaisesRegex(ValueError, "CUDA"):
            main_pce.runtime_metadata(main_pce.torch.device("cpu"))

    def test_set_seed_configures_deterministic_cuda_execution(self):
        main_pce = self.load_module()

        with mock.patch.object(main_pce.dgl, "seed") as dgl_seed:
            main_pce.set_seed(42)

        dgl_seed.assert_called_once_with(42)
        self.assertTrue(main_pce.torch.are_deterministic_algorithms_enabled())
        self.assertTrue(main_pce.torch.backends.cudnn.deterministic)
        self.assertFalse(main_pce.torch.backends.cudnn.benchmark)
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")


if __name__ == "__main__":
    unittest.main()
