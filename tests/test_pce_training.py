import importlib
import importlib.util
import math
import unittest

import dgl
import torch
import torch.nn as nn


class ConstantPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, donor_graph, acceptor_graph):
        return self.value.expand(donor_graph.batch_size)


class NonFinitePairModel(ConstantPairModel):
    def forward(self, donor_graph, acceptor_graph):
        return self.value.expand(donor_graph.batch_size) * float("nan")


def make_batch(targets):
    graphs = [dgl.graph(([0], [0]), num_nodes=1) for _ in targets]
    return {
        "donor_graph": dgl.batch(graphs),
        "acceptor_graph": dgl.batch(graphs),
        "target": torch.tensor(targets, dtype=torch.float32),
    }


class PCETrainingTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pce.training"),
            "pce.training must provide regression training utilities",
        )
        return importlib.import_module("pce.training")

    def test_target_scaler_round_trip_and_unit_population_std(self):
        training = self.load_module()
        values = torch.tensor([10.0, 12.0, 14.0])

        scaler = training.TargetScaler.fit(values)
        standardized = scaler.transform(values)

        self.assertAlmostEqual(float(standardized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(standardized.std(unbiased=False)), 1.0, places=6)
        self.assertTrue(torch.allclose(scaler.inverse(standardized), values))

    def test_regression_metrics_are_reported_in_original_units(self):
        training = self.load_module()

        metrics = training.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["r2"], 0.5)

    def test_train_one_epoch_updates_model_parameter(self):
        training = self.load_module()
        model = ConstantPairModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = training.TargetScaler(mean=0.0, std=1.0)

        loss = training.train_one_epoch(
            model,
            [make_batch([1.0, 2.0])],
            optimizer,
            scaler,
            torch.device("cpu"),
        )

        self.assertGreater(loss, 0.0)
        self.assertGreater(model.value.item(), 0.0)

    def test_evaluate_returns_original_unit_predictions_and_metrics(self):
        training = self.load_module()
        self.assertTrue(
            hasattr(training, "evaluate"),
            "pce.training needs evaluation in original PCE units",
        )
        model = ConstantPairModel()
        scaler = training.TargetScaler(mean=10.0, std=2.0)

        result = training.evaluate(
            model,
            [make_batch([10.0, 12.0])],
            scaler,
            torch.device("cpu"),
        )

        self.assertEqual(result["targets"], [10.0, 12.0])
        self.assertEqual(result["predictions"], [10.0, 10.0])
        self.assertAlmostEqual(result["loss"], 0.5)
        self.assertAlmostEqual(result["metrics"]["mae"], 1.0)
        self.assertAlmostEqual(result["metrics"]["rmse"], math.sqrt(2.0))
        self.assertAlmostEqual(result["metrics"]["r2"], -1.0)

    def test_train_one_epoch_rejects_nonfinite_predictions(self):
        training = self.load_module()
        model = NonFinitePairModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            training.train_one_epoch(
                model,
                [make_batch([1.0, 2.0])],
                optimizer,
                training.TargetScaler(mean=0.0, std=1.0),
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
