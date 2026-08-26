import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.compare_experiments import compare_predictions


class ExperimentAnalysisTests(unittest.TestCase):
    def test_paired_comparison_reports_positive_improvement_for_better_multimodal_predictions(self):
        rows = []
        for index in range(6):
            rows.append(
                {
                    "record_id": index,
                    "donor_smiles": f"D{index}",
                    "acceptor_smiles": f"A{index}",
                    "pce": float(index + 1),
                }
            )
        multimodal = pd.DataFrame(rows)
        material = pd.DataFrame(rows)
        multimodal["predicted_pce"] = multimodal["pce"] + 0.1
        material["predicted_pce"] = material["pce"] + [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            multimodal_path = directory / "multimodal.csv"
            material_path = directory / "material.csv"
            multimodal.to_csv(multimodal_path, index=False)
            material.to_csv(material_path, index=False)

            result = compare_predictions(multimodal_path, material_path)

        self.assertEqual(result["test_records"], 6)
        self.assertEqual(result["test_pairs"], 6)
        self.assertGreater(result["multimodal_improvement"]["mae_reduction"], 0)
        self.assertGreater(result["multimodal_improvement"]["r2_increase"], 0)

    def test_paired_comparison_rejects_misaligned_test_records(self):
        first = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "donor_smiles": ["D1", "D2", "D3"],
                "acceptor_smiles": ["A1", "A2", "A3"],
                "pce": [1.0, 2.0, 3.0],
                "predicted_pce": [1.0, 2.0, 3.0],
            }
        )
        second = first.copy()
        second.loc[2, "record_id"] = 99
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first_path = directory / "first.csv"
            second_path = directory / "second.csv"
            first.to_csv(first_path, index=False)
            second.to_csv(second_path, index=False)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                compare_predictions(first_path, second_path)


if __name__ == "__main__":
    unittest.main()
