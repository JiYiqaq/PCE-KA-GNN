import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
