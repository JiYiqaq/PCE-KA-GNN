import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dgl
import torch


class PCEGraphTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pce.graphs"),
            "pce.graphs must provide the production topology graph builder",
        )
        return importlib.import_module("pce.graphs")

    def test_valid_smiles_builds_finite_bidirectional_topology_graph(self):
        graphs = self.load_module()

        graph = graphs.build_topology_graph("CCO", "cgcnn", "dim_14")

        self.assertIsInstance(graph, dgl.DGLGraph)
        self.assertEqual(graph.num_nodes(), 9)
        self.assertEqual(graph.num_edges(), 16)
        self.assertEqual(tuple(graph.ndata["feat"].shape), (9, 113))
        self.assertEqual(tuple(graph.edata["feat"].shape), (16, 21))
        self.assertTrue(bool(torch.isfinite(graph.ndata["feat"]).all()))
        self.assertTrue(bool(torch.isfinite(graph.edata["feat"]).all()))
        edges = set(zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()))
        self.assertTrue(all((dst, src) in edges for src, dst in edges))
        self.assertNotIn("coor", graph.ndata)

    def test_isolated_valid_ion_has_well_shaped_empty_edge_features(self):
        graphs = self.load_module()

        graph = graphs.build_topology_graph("[Na+]", "cgcnn", "dim_14")

        self.assertGreaterEqual(graph.num_nodes(), 1)
        self.assertEqual(graph.num_edges(), 0)
        self.assertEqual(tuple(graph.ndata["feat"].shape[1:]), (113,))
        self.assertEqual(tuple(graph.edata["feat"].shape), (0, 21))

    def test_dummy_atom_is_rejected_instead_of_using_a_fake_element_feature(self):
        graphs = self.load_module()

        with self.assertRaisesRegex(ValueError, "dummy atom"):
            graphs.build_topology_graph("*CC", "cgcnn", "dim_14")

    def test_molecule_over_production_complexity_limit_is_rejected_before_sanitization(self):
        graphs = self.load_module()

        with self.assertRaisesRegex(ValueError, "500-heavy-atom"):
            graphs.build_topology_graph("C" * 501, "cgcnn", "dim_14")

    def test_double_bond_uses_the_declared_topological_length(self):
        graphs = self.load_module()

        graph = graphs.build_topology_graph("C=C", "cgcnn", "dim_14")

        double_bond_rows = graph.edata["feat"][graph.edata["feat"][:, 8] == 1]
        self.assertGreater(len(double_bond_rows), 0)
        self.assertTrue(
            torch.allclose(
                double_bond_rows[:, 11],
                torch.full_like(double_bond_rows[:, 11], 1.4),
            )
        )

    def test_cache_records_failures_and_does_not_rewrite_an_unchanged_cache(self):
        graphs = self.load_module()
        calls = []

        def builder(smiles, encoder_atom, encoder_bond):
            calls.append(smiles)
            if smiles == "invalid":
                raise ValueError("invalid test molecule")
            return graphs.build_topology_graph(smiles, encoder_atom, encoder_bond)

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "topology.pt"
            built, first_audit = graphs.build_topology_graph_cache(
                ["CCO", "CCN", "invalid"],
                cache_path,
                "cgcnn",
                "dim_14",
                graph_builder=builder,
            )
            self.assertEqual(set(built), {"CCO", "CCN"})
            self.assertEqual(first_audit["built_molecules"], 2)
            self.assertEqual(first_audit["failed_molecules"], 1)
            self.assertIn("invalid test molecule", first_audit["failure_reasons"]["invalid"])

            calls.clear()
            with mock.patch.object(graphs.torch, "save") as save_cache:
                loaded, second_audit = graphs.build_topology_graph_cache(
                    ["CCO", "CCN", "invalid"],
                    cache_path,
                    "cgcnn",
                    "dim_14",
                    graph_builder=builder,
                )

        self.assertEqual(calls, [])
        save_cache.assert_not_called()
        self.assertEqual(set(loaded), {"CCO", "CCN"})
        self.assertEqual(second_audit["loaded_cached_molecules"], 2)
        self.assertEqual(second_audit["failed_molecules"], 1)

    def test_parallel_cache_build_is_deterministic_and_reports_worker_count(self):
        graphs = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            cache, audit = graphs.build_topology_graph_cache(
                ["CCO", "CCN", "CCC"],
                Path(directory) / "parallel.pt",
                "cgcnn",
                "dim_14",
                num_workers=2,
            )

        self.assertEqual(list(cache), sorted(cache))
        self.assertEqual(audit["num_workers"], 2)
        self.assertEqual(audit["usable_molecules"], 3)


if __name__ == "__main__":
    unittest.main()
