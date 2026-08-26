import importlib
import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import dgl
import pandas as pd
import torch


class PCEDataTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pce"),
            "pce package must isolate the regression workflow",
        )
        self.assertIsNotNone(
            importlib.util.find_spec("pce.data"),
            "pce.data must provide the PCE pair-data pipeline",
        )
        return importlib.import_module("pce.data")

    def test_prepare_pair_table_canonicalizes_and_aggregates_median(self):
        data = self.load_module()
        frame = pd.DataFrame(
            {
                "donor_smiles": ["CCO", "C(C)O", "CCN", "bad smiles", None],
                "acceptor_smiles": [
                    "c1ccccc1",
                    "c1ccccc1",
                    "CCO",
                    "CCO",
                    "CCO",
                ],
                "pce": [10.0, 14.0, 7.0, 3.0, 5.0],
            }
        )

        pairs, audit = data.prepare_pair_table(frame)

        self.assertEqual(len(pairs), 2)
        ethanol_pair = pairs[
            (pairs["donor_smiles"] == "CCO")
            & (pairs["acceptor_smiles"] == "c1ccccc1")
        ].iloc[0]
        self.assertEqual(ethanol_pair["pce"], 12.0)
        self.assertEqual(ethanol_pair["replicate_count"], 2)
        self.assertEqual(audit["input_rows"], 5)
        self.assertEqual(audit["valid_rows"], 3)
        self.assertEqual(audit["unique_pairs"], 2)

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
            (prepared["donor_smiles"] == "CCO")
            & (prepared["acceptor_smiles"] == "CCN")
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

    def test_split_pair_table_is_complete_and_pair_disjoint(self):
        data = self.load_module()
        frame = pd.DataFrame(
            {
                "donor_smiles": [f"donor-{index}" for index in range(20)],
                "acceptor_smiles": [f"acceptor-{index}" for index in range(20)],
                "pce": [float(index) for index in range(20)],
            }
        )

        splits = data.split_pair_table(frame, 0.8, 0.1, 0.1, seed=42)

        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [16, 2, 2])
        keys = {
            name: set(zip(part["donor_smiles"], part["acceptor_smiles"]))
            for name, part in splits.items()
        }
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])
        self.assertEqual(len(set().union(*keys.values())), len(frame))

    def test_edge_features_are_aggregated_into_113_dimensions_only_once(self):
        data = self.load_module()
        self.assertTrue(
            hasattr(data, "add_edge_aggregates"),
            "pce.data must prepare the author's 113-dimensional node input",
        )
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graph.ndata["feat"] = torch.ones(2, 92)
        graph.edata["feat"] = torch.ones(2, 21)

        data.add_edge_aggregates(graph)
        data.add_edge_aggregates(graph)

        self.assertEqual(tuple(graph.ndata["feat"].shape), (2, 113))

    def test_graph_cache_builds_each_unique_molecule_once_and_reloads(self):
        data = self.load_module()
        self.assertTrue(
            hasattr(data, "build_graph_cache"),
            "pce.data must cache unique molecular graphs",
        )
        pairs = pd.DataFrame(
            {
                "donor_smiles": ["A", "A"],
                "acceptor_smiles": ["B", "C"],
                "pce": [1.0, 2.0],
            }
        )
        calls = []

        def graph_builder(smiles, encoder_atom, encoder_bond):
            calls.append(smiles)
            graph = dgl.graph(([0], [0]), num_nodes=1)
            graph.ndata["feat"] = torch.ones(1, 92)
            graph.edata["feat"] = torch.ones(1, 21)
            return graph

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "graphs.pt"
            graphs, usable, first_audit = data.build_graph_cache(
                pairs,
                cache_path,
                "cgcnn",
                "dim_14",
                graph_builder=graph_builder,
            )
            self.assertEqual(sorted(calls), ["A", "B", "C"])
            self.assertEqual(first_audit["built_molecules"], 3)
            self.assertEqual(len(usable), 2)

            calls.clear()
            with mock.patch.object(data.torch, "save") as save_graph_cache:
                loaded_graphs, loaded_usable, second_audit = data.build_graph_cache(
                    pairs,
                    cache_path,
                    "cgcnn",
                    "dim_14",
                    graph_builder=graph_builder,
                )
            save_graph_cache.assert_not_called()

        self.assertEqual(calls, [])
        self.assertEqual(second_audit["loaded_cached_molecules"], 3)
        self.assertEqual(set(graphs), set(loaded_graphs))
        self.assertEqual(len(loaded_usable), 2)

    def test_graph_cache_rejects_nonfinite_edge_or_node_features(self):
        data = self.load_module()
        pairs = pd.DataFrame(
            {
                "donor_smiles": ["A"],
                "acceptor_smiles": ["B"],
                "pce": [1.0],
            }
        )

        def graph_builder(smiles, encoder_atom, encoder_bond):
            graph = dgl.graph(([0], [0]), num_nodes=1)
            graph.ndata["feat"] = torch.ones(1, 92)
            graph.edata["feat"] = torch.ones(1, 21)
            if smiles == "B":
                graph.edata["feat"][0, 0] = float("nan")
            return graph

        with tempfile.TemporaryDirectory() as directory:
            graphs, usable, audit = data.build_graph_cache(
                pairs,
                Path(directory) / "graphs.pt",
                "cgcnn",
                "dim_14",
                graph_builder=graph_builder,
            )

        self.assertEqual(set(graphs), {"A"})
        self.assertEqual(len(usable), 0)
        self.assertEqual(audit["failed_molecules"], 1)

    def test_pair_dataset_collator_batches_both_roles(self):
        data = self.load_module()
        self.assertTrue(
            hasattr(data, "PairGraphDataset") and hasattr(data, "collate_pair_graphs"),
            "pce.data must provide paired graph batching",
        )
        graph_a = dgl.graph(([0], [0]), num_nodes=1)
        graph_b = dgl.graph(([0], [0]), num_nodes=1)
        for graph in (graph_a, graph_b):
            graph.ndata["feat"] = torch.ones(1, 113)
        pairs = pd.DataFrame(
            {
                "donor_smiles": ["A", "B"],
                "acceptor_smiles": ["B", "A"],
                "pce": [1.5, 2.5],
            }
        )
        dataset = data.PairGraphDataset(pairs, {"A": graph_a, "B": graph_b})

        batch = data.collate_pair_graphs([dataset[0], dataset[1]])

        self.assertEqual(batch["donor_graph"].batch_size, 2)
        self.assertEqual(batch["acceptor_graph"].batch_size, 2)
        self.assertTrue(torch.equal(batch["target"], torch.tensor([1.5, 2.5])))
        self.assertEqual(batch["donor_smiles"], ["A", "B"])

    def test_author_graph_builder_uses_the_current_dgl_constructor(self):
        from utils.graph_path import atom_to_graph

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = atom_to_graph("CCO", "cgcnn", "dim_14")

        self.assertIsNot(graph, False)
        deprecated = [
            warning
            for warning in caught
            if "Recommend creating graphs by `dgl.graph(data)`" in str(warning.message)
        ]
        self.assertEqual(deprecated, [])


if __name__ == "__main__":
    unittest.main()
