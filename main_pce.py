from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import dgl
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
PAIR_CACHE_VERSION = 1
PAIR_COLUMNS = ("donor_smiles", "acceptor_smiles", "pce")


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


def portable_result_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def portable_audit_paths(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: portable_result_path(value)
        if key.endswith("_path") and isinstance(value, (str, Path))
        else value
        for key, value in audit.items()
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_prepared_pairs(path: str | Path) -> pd.DataFrame:
    pair_table = pd.read_csv(path, low_memory=False)
    missing = sorted(set(PAIR_COLUMNS).difference(pair_table.columns))
    if missing:
        raise ValueError("Prepared pair table is missing columns: " + ", ".join(missing))
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
    return pair_table


def pair_cache_metadata_path(cache_path: str | Path) -> Path:
    return Path(cache_path).with_suffix(".meta.json")


def pair_cache_signature(data_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PAIR_CACHE_VERSION,
        "source_sha256": sha256_file(data_path),
        "donor_column": config.get("donor_column", "donor_smiles"),
        "acceptor_column": config.get("acceptor_column", "acceptor_smiles"),
        "target_column": config.get("target_column", "pce"),
    }


def load_automatic_pair_cache(
    cache_path: Path,
    signature: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    metadata_path = pair_cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return None
        if metadata.get("cache_sha256") != sha256_file(cache_path):
            return None
        pair_table = read_prepared_pairs(cache_path)
        data_audit = dict(metadata["data_audit"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    data_audit["source"] = "prepared_pairs_cache"
    data_audit["prepared_pairs_cache_path"] = str(cache_path)
    return pair_table, data_audit


def write_automatic_pair_cache(
    pair_table: pd.DataFrame,
    cache_path: Path,
    signature: dict[str, Any],
    data_audit: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = pair_cache_metadata_path(cache_path)
    temporary_cache = cache_path.with_name(cache_path.name + ".tmp")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    pair_table.to_csv(temporary_cache, index=False)
    metadata = {
        "signature": signature,
        "cache_sha256": sha256_file(temporary_cache),
        "data_audit": {key: value for key, value in data_audit.items() if key != "source"},
    }
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_cache.replace(cache_path)
    temporary_metadata.replace(metadata_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: object) -> torch.device:
    requested = str(requested_device).strip().lower()
    if requested != "cuda":
        raise ValueError(
            "Production runs require configuration 'device: cuda'; "
            f"received {requested_device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this production run, but PyTorch cannot access it. "
            "Install the verified CUDA environment instead of falling back to CPU."
        )
    device = torch.device("cuda:0")
    try:
        probe = torch.ones(1, device=device)
        if float(probe.sum()) != 1.0:
            raise RuntimeError("CUDA verification returned an invalid value")
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(f"CUDA device verification failed: {error}") from error
    return device


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("CUDA runtime metadata requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "torch_version": torch.__version__,
        "dgl_version": dgl.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_memory_gb": round(properties.total_memory / 1024**3, 3),
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


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
        pair_table = read_prepared_pairs(prepared_path)
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
    cache_path = resolve_project_path(
        config.get("prepared_pairs_cache_path", "data/processed/canonical_pairs.csv")
    )
    signature = pair_cache_signature(data_path, config)
    cached = load_automatic_pair_cache(cache_path, signature)
    if cached is not None:
        return cached
    raw_frame = pd.read_csv(data_path, low_memory=False)
    pair_table, data_audit = prepare_pair_table(
        raw_frame,
        donor_col=config.get("donor_column", "donor_smiles"),
        acceptor_col=config.get("acceptor_column", "acceptor_smiles"),
        target_col=config.get("target_column", "pce"),
    )
    data_audit["source"] = "raw_csv"
    write_automatic_pair_cache(pair_table, cache_path, signature, data_audit)
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(output_dir / "canonical_pairs.csv", index=False)
    return pair_table, data_audit


def pair_data_status_message(
    pair_table: pd.DataFrame,
    data_audit: dict[str, Any],
) -> str:
    source = data_audit.get("source")
    if source == "prepared_pairs_cache":
        return f"Loaded {len(pair_table)} donor-acceptor pairs from the validated pair cache."
    if source == "prepared_pairs":
        return f"Loaded {len(pair_table)} donor-acceptor pairs from the prepared pair table."
    if source == "raw_csv":
        return (
            f"Prepared {len(pair_table)} unique donor-acceptor pairs "
            f"from {data_audit['input_rows']} rows."
        )
    raise ValueError(f"unknown pair-data source: {source!r}")


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
    print(pair_data_status_message(pair_table, data_audit))

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

    device = resolve_device(config.get("device"))
    runtime = runtime_metadata(device)
    torch.cuda.reset_peak_memory_stats(device)
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
        f"Runtime: PyTorch {runtime['torch_version']}; DGL {runtime['dgl_version']}; "
        f"CUDA {runtime['cuda_runtime']}; GPU {runtime['gpu_name']} "
        f"({runtime['gpu_memory_gb']:.1f} GB)"
    )
    print(
        f"Device: {runtime['device']}; trainable parameters: "
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
        "data_path": portable_result_path(data_path),
        "device": runtime["device"],
        "runtime": {
            **runtime,
            "peak_gpu_memory_mb": round(
                torch.cuda.max_memory_allocated(device) / 1024**2,
                3,
            ),
        },
        "data_audit": portable_audit_paths(data_audit),
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
