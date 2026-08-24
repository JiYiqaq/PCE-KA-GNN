from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from model.pce_ka_gnn import DualKAGNNRegressor
from pce.data import (
    PairGraphDataset,
    build_graph_cache,
    collate_pair_graphs,
    prepare_pair_table,
    split_pair_table,
)
from pce.training import TargetScaler, evaluate, train_one_epoch


PROJECT_DIR = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    pair_table: pd.DataFrame,
    graphs: dict,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PairGraphDataset(pair_table, graphs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_graphs,
        generator=generator,
        drop_last=False,
    )


def load_pair_data(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared_value = config.get("prepared_pairs_path")
    if prepared_value:
        prepared_path = resolve_project_path(prepared_value)
        if not prepared_path.exists():
            raise FileNotFoundError(f"Prepared PCE pair table not found: {prepared_path}")
        pair_table = pd.read_csv(prepared_path, low_memory=False)
        required = {"donor_smiles", "acceptor_smiles", "pce"}
        missing = sorted(required.difference(pair_table.columns))
        if missing:
            raise ValueError(
                "Prepared pair table is missing columns: " + ", ".join(missing)
            )
        numeric_pce = pd.to_numeric(pair_table["pce"], errors="coerce")
        valid = (
            pair_table["donor_smiles"].notna()
            & pair_table["acceptor_smiles"].notna()
            & numeric_pce.notna()
            & np.isfinite(numeric_pce)
        )
        if not bool(valid.all()):
            raise ValueError("Prepared pair table contains missing or non-finite values")
        if bool(pair_table.duplicated(["donor_smiles", "acceptor_smiles"]).any()):
            raise ValueError("Prepared pair table contains duplicate donor-acceptor pairs")
        pair_table = pair_table.copy()
        pair_table["pce"] = numeric_pce.astype(float)
        return pair_table, {
            "source": "prepared_pairs",
            "prepared_pairs_path": str(prepared_path),
            "unique_pairs": int(len(pair_table)),
        }

    data_path = resolve_project_path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"PCE CSV not found: {data_path}. Restore the file or update data_path in the YAML config."
        )
    raw_frame = pd.read_csv(data_path, low_memory=False)
    pair_table, data_audit = prepare_pair_table(
        raw_frame,
        donor_col=config.get("donor_column", "donor_smiles"),
        acceptor_col=config.get("acceptor_column", "acceptor_smiles"),
        target_col=config.get("target_column", "pce"),
    )
    data_audit["source"] = "raw_csv"
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(output_dir / "canonical_pairs.csv", index=False)
    return pair_table, data_audit


def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if "torch_num_threads" in config:
        torch.set_num_threads(int(config["torch_num_threads"]))

    data_path = resolve_project_path(config["data_path"])
    output_dir = resolve_project_path(config.get("output_dir", "outputs/pce"))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_path = resolve_project_path(
        config.get("graph_cache_path", "data/processed/pce_graphs.pt")
    )

    pair_table, data_audit = load_pair_data(config, output_dir)
    if data_audit["source"] == "prepared_pairs":
        print(f"Loaded {len(pair_table)} previously prepared donor-acceptor pairs.")
    else:
        print(
            f"Prepared {len(pair_table)} unique donor-acceptor pairs "
            f"from {data_audit['input_rows']} rows."
        )

    graphs, pair_table, graph_audit = build_graph_cache(
        pair_table,
        graph_cache_path,
        encoder_atom=config.get("encoder_atom", "cgcnn"),
        encoder_bond=config.get("encoder_bond", "dim_14"),
    )
    print(
        "Graph cache: "
        f"{len(graphs)} usable molecules, {graph_audit['failed_molecules']} failed molecules, "
        f"{len(pair_table)} usable pairs."
    )
    if len(pair_table) < 10:
        raise ValueError("fewer than 10 usable unique pairs remain after data and graph validation")

    splits = split_pair_table(
        pair_table,
        train_ratio=float(config.get("train_ratio", 0.8)),
        validation_ratio=float(config.get("validation_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
        seed=seed,
    )
    split_table = pd.concat(
        [part.assign(split=name) for name, part in splits.items()],
        ignore_index=True,
    )
    split_table.to_csv(output_dir / "prepared_pairs_with_split.csv", index=False)
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(part)}" for name, part in splits.items())
    )

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    loaders = {
        name: make_loader(
            part,
            graphs,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            seed=seed,
        )
        for name, part in splits.items()
    }
    scaler = TargetScaler.fit(torch.tensor(splits["train"]["pce"].to_numpy(), dtype=torch.float32))

    requested_device = str(config.get("device", "auto")).lower()
    use_cuda = requested_device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_config = {
        "in_feat": int(config.get("in_feat", 113)),
        "hidden_feat": int(config.get("hidden_feat", 64)),
        "fusion_hidden": int(config.get("fusion_hidden", 32)),
        "grid_feat": int(config.get("grid_feat", 1)),
        "num_layers": int(config.get("num_layers", 4)),
        "pooling": config.get("pooling", "avg"),
        "use_bias": bool(config.get("use_bias", True)),
    }
    model = DualKAGNNRegressor(**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    print(
        f"Device: {device}; trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters())}"
    )

    epochs = int(config.get("epochs", 100))
    patience_limit = int(config.get("patience", epochs))
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scaler, device)
        validation = evaluate(model, loaders["validation"], scaler, device)
        validation_metrics = validation["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": float(validation["loss"]),
            "validation_mae": float(validation_metrics["mae"]),
            "validation_rmse": float(validation_metrics["rmse"]),
            "validation_r2": float(validation_metrics["r2"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train MSE(z)={train_loss:.5f} | "
            f"val MAE={row['validation_mae']:.5f} | "
            f"val RMSE={row['validation_rmse']:.5f} | val R2={row['validation_r2']:.5f}"
        )

        if row["validation_mae"] < best_validation_mae:
            best_validation_mae = row["validation_mae"]
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "target_scaler": scaler.to_dict(),
                    "best_epoch": best_epoch,
                    "configuration": config,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping after {epoch} epochs.")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate(model, loaders["test"], scaler, device)
    prediction_table = splits["test"][["donor_smiles", "acceptor_smiles", "pce"]].copy()
    prediction_table["predicted_pce"] = test_result["predictions"]
    prediction_table["absolute_error"] = np.abs(
        prediction_table["pce"] - prediction_table["predicted_pce"]
    )
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "data_path": str(data_path),
        "device": str(device),
        "data_audit": data_audit,
        "graph_audit": graph_audit,
        "split_sizes": {name: int(len(part)) for name, part in splits.items()},
        "target_scaler": scaler.to_dict(),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": int(best_epoch),
        "test_metrics": test_result["metrics"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    metrics = summary["test_metrics"]
    print(
        f"Test metrics | MAE={metrics['mae']:.5f} | RMSE={metrics['rmse']:.5f} | "
        f"R2={metrics['r2']:.5f}"
    )
    print(f"Results saved to: {output_dir}")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Donor-acceptor KA-GNN PCE regression")
    parser.add_argument("--config", default="config/pce.yaml", help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
