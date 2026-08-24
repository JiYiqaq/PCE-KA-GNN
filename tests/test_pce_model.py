import importlib
import importlib.util
import unittest

import dgl
import torch


def make_graph(value):
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata["feat"] = torch.full((2, 113), float(value))
    return graph


class PCEModelTests(unittest.TestCase):
    def load_model_class(self):
        self.assertIsNotNone(
            importlib.util.find_spec("model.pce_ka_gnn"),
            "model.pce_ka_gnn must define the dual-graph regression model",
        )
        module = importlib.import_module("model.pce_ka_gnn")
        return module.DualKAGNNRegressor

    def test_forward_returns_one_finite_value_per_pair_and_backpropagates(self):
        model_class = self.load_model_class()
        model = model_class(
            in_feat=113,
            hidden_feat=8,
            fusion_hidden=6,
            grid_feat=1,
            num_layers=2,
            pooling="avg",
        )
        donor_graphs = dgl.batch([make_graph(0.1), make_graph(0.2)])
        acceptor_graphs = dgl.batch([make_graph(0.3), make_graph(0.4)])

        prediction = model(donor_graphs, acceptor_graphs)

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_output_is_not_limited_by_a_sigmoid(self):
        model_class = self.load_model_class()
        model = model_class(
            in_feat=113,
            hidden_feat=8,
            fusion_hidden=6,
            grid_feat=1,
            num_layers=1,
            pooling="avg",
        )
        with torch.no_grad():
            model.output_layer.fouriercoeffs.zero_()
            model.output_layer.bias.fill_(5.0)

        prediction = model(dgl.batch([make_graph(0.1)]), dgl.batch([make_graph(0.2)]))

        self.assertAlmostEqual(prediction.item(), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
