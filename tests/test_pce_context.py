import importlib
import importlib.util
import math
import unittest

import pandas as pd
import torch


class PCEContextTests(unittest.TestCase):
    def load_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pce.context"),
            "pce.context must provide leakage-free context preprocessing",
        )
        return importlib.import_module("pce.context")

    def test_ratio_parsers_accept_declared_units_and_reject_ambiguous_values(self):
        context = self.load_module()

        self.assertAlmostEqual(context.parse_donor_acceptor_log_ratio("2:1"), math.log(2.0))
        self.assertAlmostEqual(context.parse_donor_acceptor_log_ratio("1:2 w/w"), math.log(0.5))
        self.assertTrue(math.isnan(context.parse_donor_acceptor_log_ratio("unknown")))
        self.assertAlmostEqual(context.parse_additive_percentage("0.5 vol%"), 0.5)
        self.assertAlmostEqual(context.parse_additive_percentage("3% v/v"), 3.0)
        self.assertTrue(math.isnan(context.parse_additive_percentage("1:2")))
        self.assertTrue(math.isnan(context.parse_additive_percentage(None)))

    def test_preprocessor_uses_train_only_statistics_and_explicit_missing_masks(self):
        context = self.load_module()
        train = pd.DataFrame(
            {
                "homo_d": [-5.0, -6.0, None],
                "lumo_d": [-3.0, -4.0, -3.5],
                "homo_a": [-5.8, -5.9, -6.0],
                "lumo_a": [-3.8, -3.9, -4.0],
                "active_layer_thickness": [100.0, 120.0, None],
                "annealing_temp": [90.0, 110.0, None],
                "d_a_ratio": ["1:1", "2:1", None],
                "additive_ratio": ["0.5%", None, "3 vol%"],
                "device_type": ["conventional", "inverted", None],
                "etl_canonical": ["ZnO", "PFN-Br", "ZnO"],
                "htl_canonical": ["PEDOT:PSS", "MoO3", "PEDOT:PSS"],
                "solvent_canonical": ["chloroform", "toluene", None],
                "additive_canonical": ["DIO", None, "CN"],
            }
        )
        validation = train.iloc[[0]].copy()
        validation.loc[:, "homo_d"] = -100.0
        validation.loc[:, "device_type"] = "future-device"

        preprocessor = context.ContextPreprocessor(min_category_frequency=1).fit(train)
        numeric, categorical = preprocessor.transform(validation)

        self.assertEqual(preprocessor.numeric_statistics["homo_d"]["median"], -5.5)
        self.assertEqual(tuple(numeric.shape), (1, len(context.NUMERIC_FEATURES) * 2))
        self.assertEqual(tuple(categorical.shape), (1, len(context.CATEGORICAL_FEATURES)))
        self.assertTrue(bool(torch.isfinite(numeric).all()))
        device_index = context.CATEGORICAL_FEATURES.index("device_type")
        self.assertEqual(int(categorical[0, device_index]), context.UNKNOWN_CATEGORY_INDEX)

        missing_numeric, missing_categorical = preprocessor.transform(train.iloc[[2]])
        homo_index = context.NUMERIC_FEATURES.index("homo_d")
        mask_index = len(context.NUMERIC_FEATURES) + homo_index
        self.assertEqual(float(missing_numeric[0, mask_index]), 0.0)
        self.assertEqual(int(missing_categorical[0, device_index]), context.MISSING_CATEGORY_INDEX)

    def test_preprocessor_serialization_round_trip_is_exact(self):
        context = self.load_module()
        frame = pd.DataFrame(
            {
                name: [None, None]
                for name in (*context.NUMERIC_SOURCE_COLUMNS, *context.CATEGORICAL_FEATURES)
            }
        )
        frame["homo_d"] = [-5.2, -5.4]
        frame["device_type"] = ["conventional", "inverted"]
        original = context.ContextPreprocessor(min_category_frequency=1).fit(frame)
        restored = context.ContextPreprocessor.from_dict(original.to_dict())

        original_numeric, original_categorical = original.transform(frame)
        restored_numeric, restored_categorical = restored.transform(frame)

        self.assertTrue(torch.equal(original_numeric, restored_numeric))
        self.assertTrue(torch.equal(original_categorical, restored_categorical))
        self.assertEqual(original.category_sizes, restored.category_sizes)

    def test_context_feature_contract_excludes_target_derived_columns(self):
        context = self.load_module()
        forbidden = {"pce", "voc", "jsc", "ff", "pce_recomputed", "pce_best", "pce_avg"}

        self.assertFalse(forbidden & set(context.NUMERIC_SOURCE_COLUMNS))
        self.assertFalse(forbidden & set(context.CATEGORICAL_FEATURES))


if __name__ == "__main__":
    unittest.main()
