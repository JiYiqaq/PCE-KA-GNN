import importlib
import importlib.util
import unittest

import dgl
import pandas as pd
import torch


class PCEDataTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(importlib.util.find_spec("pce.data"))
        return importlib.import_module("pce.data")

    def test_prepare_device_table_preserves_conditions_and_repeated_pairs(self):
        data = self.load_module()
        frame = pd.DataFrame(
            {
                "id": [11, 12, 13, 14],
                "doi": ["a", "a", "b", "c"],
                "donor_smiles": ["OCC", "CCO", "CCC", "CCO"],
                "acceptor_smiles": ["CCN", "CCN", "CCO", None],
                "pce": [5.0, 7.0, 3.0, 9.0],
                "solvent_canonical": ["chloroform", "toluene", None, "water"],
                "annealing_temp": [100.0, 140.0, None, 80.0],
                "voc": [0.7, 0.8, 0.6, 0.9],
                "jsc": [10.0, 12.0, 8.0, 13.0],
                "ff": [60.0, 65.0, 55.0, 70.0],
                "pce_recomputed": [4.2, 6.24, 2.64, 8.19],
            }
        )

        prepared, audit = data.prepare_device_table(frame)

        self.assertEqual(len(prepared), 3)
        repeated = prepared[
            (prepared["donor_smiles"] == "CCO") & (prepared["acceptor_smiles"] == "CCN")
        ]
        self.assertEqual(len(repeated), 2)
        self.assertEqual(set(repeated["pce"]), {5.0, 7.0})
        self.assertEqual(set(repeated["solvent_canonical"]), {"chloroform", "toluene"})
        self.assertEqual(set(repeated["annealing_temp"]), {100.0, 140.0})
        for forbidden in ("voc", "jsc", "ff", "pce_recomputed", "pce_best", "pce_avg"):
            self.assertNotIn(forbidden, prepared.columns)
        self.assertEqual(audit["input_rows"], 4)
        self.assertEqual(audit["usable_device_rows"], 3)
        self.assertEqual(audit["unique_pairs"], 2)

    def test_device_split_keeps_every_pair_in_exactly_one_partition(self):
        data = self.load_module()
        rows = []
        for pair_index in range(10):
            for condition in range(2):
                rows.append(
                    {
                        "donor_smiles": f"D{pair_index}",
                        "acceptor_smiles": f"A{pair_index}",
                        "pce": float(pair_index + condition),
                    }
                )
        frame = pd.DataFrame(rows)

        splits = data.split_device_table_by_pair(frame, 0.8, 0.1, 0.1, seed=42)

        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [16, 2, 2])
        keys = {
            name: set(zip(part["donor_smiles"], part["acceptor_smiles"]))
            for name, part in splits.items()
        }
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])
        self.assertEqual(sum(len(part) for part in splits.values()), len(frame))

    def test_device_split_balances_rows_when_pair_replicate_counts_differ(self):
        data = self.load_module()
        rows = []
        pair_sizes = [40, 12, 8, 5, 4, 3, 3, 2, 2, 1]
        for pair_index, size in enumerate(pair_sizes):
            rows.extend(
                {
                    "donor_smiles": f"D{pair_index}",
                    "acceptor_smiles": f"A{pair_index}",
                    "pce": float(pair_index),
                }
                for _ in range(size)
            )
        frame = pd.DataFrame(rows)

        splits = data.split_device_table_by_pair(frame, 0.8, 0.1, 0.1, seed=42)

        row_counts = [len(splits[name]) for name in ("train", "validation", "test")]
        self.assertLessEqual(abs(row_counts[0] - 64), 8)
        self.assertLessEqual(abs(row_counts[1] - 8), 4)
        self.assertLessEqual(abs(row_counts[2] - 8), 4)

    def test_device_dataset_collator_preserves_context_and_provenance(self):
        data = self.load_module()
        graph_a = dgl.graph(([0], [0]), num_nodes=1)
        graph_b = dgl.graph(([0], [0]), num_nodes=1)
        for graph in (graph_a, graph_b):
            graph.ndata["feat"] = torch.ones(1, 113)
        devices = pd.DataFrame(
            {
                "record_id": [101, 102],
                "doi": ["doi-a", "doi-b"],
                "donor_smiles": ["A", "B"],
                "acceptor_smiles": ["B", "A"],
                "pce": [4.5, 8.5],
            }
        )
        numeric = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
        categorical = torch.tensor([[2, 0], [3, 4]], dtype=torch.long)
        dataset = data.DeviceGraphDataset(
            devices, {"A": graph_a, "B": graph_b}, numeric, categorical
        )

        batch = data.collate_device_graphs([dataset[0], dataset[1]])

        self.assertEqual(batch["donor_graph"].batch_size, 2)
        self.assertEqual(batch["acceptor_graph"].batch_size, 2)
        self.assertTrue(torch.equal(batch["numeric_context"], numeric))
        self.assertTrue(torch.equal(batch["categorical_context"], categorical))
        self.assertTrue(torch.equal(batch["target"], torch.tensor([4.5, 8.5])))
        self.assertEqual(batch["record_id"], [101, 102])
        self.assertEqual(batch["doi"], ["doi-a", "doi-b"])


if __name__ == "__main__":
    unittest.main()
