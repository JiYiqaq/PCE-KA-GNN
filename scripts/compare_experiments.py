"""Paired, pair-cluster-aware comparison of two PCE experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, shapiro, wilcoxon

PAIR_COLUMNS = ["donor_smiles", "acceptor_smiles"]
ALIGNMENT_COLUMNS = ["record_id", *PAIR_COLUMNS, "pce"]


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    total = np.square(target - target.mean()).sum()
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / total),
    }


def cluster_bootstrap_mean_difference(
    pair_sums: np.ndarray,
    pair_counts: np.ndarray,
    seed: int = 42,
    resamples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    pair_count = len(pair_sums)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        sampled = rng.integers(0, pair_count, size=(stop - start, pair_count))
        estimates[start:stop] = pair_sums[sampled].sum(axis=1) / pair_counts[sampled].sum(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def compare_predictions(
    multimodal_path: str | Path,
    material_only_path: str | Path,
) -> dict[str, object]:
    multimodal = pd.read_csv(multimodal_path, low_memory=False)
    material_only = pd.read_csv(material_only_path, low_memory=False)
    for name, frame in (("multimodal", multimodal), ("material_only", material_only)):
        missing = sorted(set([*ALIGNMENT_COLUMNS, "predicted_pce"]).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {', '.join(missing)}")
    if len(multimodal) != len(material_only):
        raise ValueError("prediction tables contain different numbers of test records")
    if not multimodal[ALIGNMENT_COLUMNS].equals(material_only[ALIGNMENT_COLUMNS]):
        raise ValueError("prediction tables are not aligned to identical test records")

    target = multimodal["pce"].to_numpy(dtype=np.float64)
    multimodal_prediction = multimodal["predicted_pce"].to_numpy(dtype=np.float64)
    material_prediction = material_only["predicted_pce"].to_numpy(dtype=np.float64)
    multimodal_error = np.abs(target - multimodal_prediction)
    material_error = np.abs(target - material_prediction)
    difference = material_error - multimodal_error

    pair_frame = multimodal[PAIR_COLUMNS].copy()
    pair_frame["difference"] = difference
    pair_summary = pair_frame.groupby(PAIR_COLUMNS, sort=True)["difference"].agg(["sum", "size", "mean"])
    pair_differences = pair_summary["mean"].to_numpy(dtype=np.float64)
    shapiro_result = shapiro(pair_differences)
    nonzero = pair_differences[pair_differences != 0]
    wilcoxon_result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    ranks = rankdata(np.abs(nonzero))
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    rank_biserial = (positive_rank_sum - negative_rank_sum) / (
        positive_rank_sum + negative_rank_sum
    )
    ci_lower, ci_upper = cluster_bootstrap_mean_difference(
        pair_summary["sum"].to_numpy(dtype=np.float64),
        pair_summary["size"].to_numpy(dtype=np.int64),
    )

    multimodal_metrics = regression_metrics(target, multimodal_prediction)
    material_metrics = regression_metrics(target, material_prediction)
    mae_difference = material_metrics["mae"] - multimodal_metrics["mae"]
    return {
        "design": "paired predictions with donor-acceptor pair as the independent cluster",
        "test_records": int(len(multimodal)),
        "test_pairs": int(len(pair_summary)),
        "error_difference_definition": "material_only_absolute_error - multimodal_absolute_error",
        "multimodal_metrics": multimodal_metrics,
        "material_only_metrics": material_metrics,
        "multimodal_improvement": {
            "mae_reduction": float(mae_difference),
            "rmse_reduction": float(material_metrics["rmse"] - multimodal_metrics["rmse"]),
            "r2_increase": float(multimodal_metrics["r2"] - material_metrics["r2"]),
            "relative_mae_reduction_percent": float(
                100.0 * mae_difference / material_metrics["mae"]
            ),
            "pair_cluster_bootstrap_95_ci_for_mae_reduction": [ci_lower, ci_upper],
        },
        "record_level": {
            "multimodal_lower_absolute_error_fraction": float(np.mean(difference > 0)),
            "material_only_lower_absolute_error_fraction": float(np.mean(difference < 0)),
            "ties_fraction": float(np.mean(difference == 0)),
        },
        "pair_level_diagnostics": {
            "mean_difference": float(np.mean(pair_differences)),
            "median_difference": float(np.median(pair_differences)),
            "standard_deviation": float(np.std(pair_differences, ddof=1)),
            "shapiro_w": float(shapiro_result.statistic),
            "shapiro_p": float(shapiro_result.pvalue),
            "selected_test": "two-sided Wilcoxon signed-rank due to non-normal paired differences",
            "wilcoxon_statistic": float(wilcoxon_result.statistic),
            "wilcoxon_p": float(wilcoxon_result.pvalue),
            "rank_biserial_correlation": float(rank_biserial),
        },
        "limitation": (
            "This paired test quantifies test-pair variation for one deterministic training seed; "
            "it does not quantify model-training variability across seeds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal", default="outputs/pce_multimodal/test_predictions.csv")
    parser.add_argument("--material-only", default="outputs/pce_material_only/test_predictions.csv")
    parser.add_argument("--output", default="outputs/experiment_comparison.json")
    arguments = parser.parse_args()
    result = compare_predictions(arguments.multimodal, arguments.material_only)
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
