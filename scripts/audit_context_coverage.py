"""Report numeric availability and missing/unknown category coverage by split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from pce.context import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ContextPreprocessor  # noqa: E402


def audit(split_path: str | Path, checkpoint_path: str | Path) -> dict[str, object]:
    records = pd.read_csv(split_path, low_memory=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    preprocessor = ContextPreprocessor.from_dict(checkpoint["context_preprocessor"])
    result: dict[str, object] = {
        "preprocessor_fitted_on": "train_only",
        "category_sizes": dict(zip(CATEGORICAL_FEATURES, preprocessor.category_sizes)),
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        split = records.loc[records["split"] == split_name].reset_index(drop=True)
        numeric, categorical = preprocessor.transform(split)
        masks = numeric[:, len(NUMERIC_FEATURES) :]
        split_audit = {
            "rows": len(split),
            "numeric_available_fraction": {
                feature: float(masks[:, index].mean())
                for index, feature in enumerate(NUMERIC_FEATURES)
            },
            "categorical_missing_fraction": {
                feature: float((categorical[:, index] == 0).float().mean())
                for index, feature in enumerate(CATEGORICAL_FEATURES)
            },
            "categorical_unknown_fraction": {
                feature: float((categorical[:, index] == 1).float().mean())
                for index, feature in enumerate(CATEGORICAL_FEATURES)
            },
        }
        result["splits"][split_name] = split_audit
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-table", default="outputs/pce_multimodal/device_records_with_split.csv"
    )
    parser.add_argument("--checkpoint", default="outputs/pce_multimodal/best_model.pt")
    parser.add_argument("--output", default="outputs/context_coverage_audit.json")
    arguments = parser.parse_args()
    result = audit(arguments.split_table, arguments.checkpoint)
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
