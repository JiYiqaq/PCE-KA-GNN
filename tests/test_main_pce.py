import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dgl
import pandas as pd
import torch


def raw_device_frame(rows=1):
    return pd.DataFrame(
        {
            "id": list(range(1, rows + 1)),
            "doi": ["10/example"] * rows,
            "donor_smiles": ["CCO"] * rows,
            "acceptor_smiles": ["CCN"] * rows,
            "pce": [7.5] * rows,
            "homo_d": [-5.2] * rows,
            "device_type": ["conventional"] * rows,
        }
    )


class MainPCETests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(importlib.util.find_spec("main_pce"))
        return importlib.import_module("main_pce")

    def test_load_config_honors_the_supplied_path(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text("data_path: custom.csv\nepochs: 3\n", encoding="utf-8")
            config = main_pce.load_config(path)
        self.assertEqual(config["data_path"], "custom.csv")
        self.assertEqual(config["epochs"], 3)

    def test_device_cache_is_fingerprinted_reused_and_preserves_repeated_rows(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw_path = directory / "raw.csv"
            cache_path = directory / "processed" / "device_records.csv"
            frame = raw_device_frame(2)
            frame.loc[1, "pce"] = 9.0
            frame.to_csv(raw_path, index=False)
            config = {
                "data_path": str(raw_path),
                "prepared_devices_cache_path": str(cache_path),
            }
            first, first_audit = main_pce.load_device_data(config)
            second, second_audit = main_pce.load_device_data(config)

            self.assertEqual(len(first), 2)
            self.assertEqual(first_audit["source"], "raw_csv")
            self.assertEqual(second_audit["source"], "prepared_devices_cache")
            self.assertEqual(set(second["pce"]), {7.5, 9.0})
            self.assertTrue(cache_path.is_file())
            self.assertTrue(cache_path.with_suffix(".meta.json").is_file())

    def test_device_cache_invalidates_when_source_changes(self):
        main_pce = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw_path = directory / "raw.csv"
            cache_path = directory / "device_records.csv"
            config = {
                "data_path": str(raw_path),
                "prepared_devices_cache_path": str(cache_path),
            }
            raw_device_frame(1).to_csv(raw_path, index=False)
            main_pce.load_device_data(config)
            raw_device_frame(2).to_csv(raw_path, index=False)
            records, audit = main_pce.load_device_data(config)

        self.assertEqual(len(records), 2)
        self.assertEqual(audit["source"], "raw_csv")

    def test_device_cache_uses_repository_line_endings(self):
        main_pce = self.load_module()
        records, audit = main_pce.prepare_device_table(raw_device_frame(1))
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "device_records.csv"
            main_pce.write_automatic_device_cache(
                records, cache_path, {"version": 2}, audit
            )
            cache_bytes = cache_path.read_bytes()
        self.assertIn(b"\n", cache_bytes)
        self.assertNotIn(b"\r\n", cache_bytes)

    def test_result_paths_and_audit_paths_are_portable(self):
        main_pce = self.load_module()
        project_file = main_pce.PROJECT_DIR / "data" / "raw" / "Active_Database.csv"
        self.assertEqual(main_pce.portable_result_path(project_file), "data/raw/Active_Database.csv")
        portable = main_pce.portable_audit_paths(
            {"prepared_devices_cache_path": str(main_pce.PROJECT_DIR / "data/processed/device_records.csv")}
        )
        self.assertEqual(portable["prepared_devices_cache_path"], "data/processed/device_records.csv")

    def test_resolve_device_refuses_cpu_and_fails_without_cuda(self):
        main_pce = self.load_module()
        with self.assertRaisesRegex(ValueError, "device: cuda"):
            main_pce.resolve_device("cpu")
        with mock.patch.object(main_pce.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                main_pce.resolve_device("cuda")

    def test_runtime_metadata_rejects_cpu(self):
        main_pce = self.load_module()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            main_pce.runtime_metadata(torch.device("cpu"))

    def test_set_seed_configures_deterministic_cuda_execution(self):
        main_pce = self.load_module()
        with mock.patch.object(main_pce.dgl, "seed") as dgl_seed:
            main_pce.set_seed(42)
        dgl_seed.assert_called_once_with(42)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")

    @unittest.skipUnless(torch.cuda.is_available(), "production graph preload requires CUDA")
    def test_graph_preload_moves_every_graph_to_cuda_and_reports_memory(self):
        main_pce = self.load_module()
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graph.ndata["feat"] = torch.ones(2, 113)
        graph.edata["feat"] = torch.ones(2, 21)

        preloaded, audit = main_pce.preload_graphs_to_cuda(
            {"CC": graph}, torch.device("cuda"), max_free_fraction=0.9, safety_factor=2.0
        )

        self.assertEqual(preloaded["CC"].device.type, "cuda")
        self.assertGreater(audit["estimated_raw_mb"], 0.0)
        self.assertGreaterEqual(audit["actual_allocated_mb"], 0.0)
        self.assertTrue(audit["all_graphs_preloaded"])

    @unittest.skipUnless(torch.cuda.is_available(), "production graph preload requires CUDA")
    def test_graph_preload_fails_instead_of_using_cpu_when_budget_is_insufficient(self):
        main_pce = self.load_module()
        graph = dgl.graph(([0], [0]), num_nodes=1)
        graph.ndata["feat"] = torch.ones(1, 113)
        with mock.patch.object(main_pce.torch.cuda, "mem_get_info", return_value=(1, 2)):
            with self.assertRaisesRegex(RuntimeError, "GPU graph preload memory check failed"):
                main_pce.preload_graphs_to_cuda(
                    {"C": graph}, torch.device("cuda"), max_free_fraction=0.5, safety_factor=2.0
                )


if __name__ == "__main__":
    unittest.main()
