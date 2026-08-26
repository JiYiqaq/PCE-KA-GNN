from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

# Must be defined before CUDA is initialized for deterministic cuBLAS kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from model.pce_ka_gnn import DualKAGNNRegressor
from pce.context import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_SOURCE_COLUMNS,
    ContextPreprocessor,
)
from pce.data import (
    DEVICE_CATEGORICAL_COLUMNS,
    DEVICE_NUMERIC_COLUMNS,
    DeviceGraphDataset,
    collate_device_graphs,
    prepare_device_table,
    split_device_table_by_pair,
)
from pce.graphs import build_topology_graph_cache
from pce.training import TargetScaler, evaluate, train_one_epoch


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE_CACHE_VERSION = 2
DEVICE_COLUMNS = (
    "record_id",
    "doi",
    "donor_smiles",
    "acceptor_smiles",
    "pce",
    *DEVICE_NUMERIC_COLUMNS,
    *DEVICE_CATEGORICAL_COLUMNS,
)


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


def device_cache_metadata_path(cache_path: str | Path) -> Path:
    return Path(cache_path).with_suffix(".meta.json")


def device_cache_signature(data_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": DEVICE_CACHE_VERSION,
        "source_sha256": sha256_file(data_path),
        "donor_column": config.get("donor_column", "donor_smiles"),
        "acceptor_column": config.get("acceptor_column", "acceptor_smiles"),
        "target_column": config.get("target_column", "pce"),
        "retained_context_columns": [*DEVICE_NUMERIC_COLUMNS, *DEVICE_CATEGORICAL_COLUMNS],
    }


def read_prepared_devices(path: str | Path) -> pd.DataFrame:
    records = pd.read_csv(path, low_memory=False)
    missing = sorted(set(DEVICE_COLUMNS).difference(records.columns))
    if missing:
        raise ValueError("Prepared device table is missing columns: " + ", ".join(missing))
    records = records.loc[:, DEVICE_COLUMNS].copy()
    records["pce"] = pd.to_numeric(records["pce"], errors="coerce")
    valid = (
        records["donor_smiles"].notna()
        & records["acceptor_smiles"].notna()
        & records["pce"].notna()
        & np.isfinite(records["pce"])
        & records["donor_smiles"].astype(str).str.strip().ne("")
        & records["acceptor_smiles"].astype(str).str.strip().ne("")
    )
    if not bool(valid.all()):
        raise ValueError("Prepared device table contains invalid required values")
    for column in DEVICE_NUMERIC_COLUMNS:
        if column not in {"d_a_ratio", "additive_ratio"}:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    return records.reset_index(drop=True)


def load_automatic_device_cache(
    cache_path: Path,
    signature: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    metadata_path = device_cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return None
        if metadata.get("cache_sha256") != sha256_file(cache_path):
            return None
        records = read_prepared_devices(cache_path)
        audit = dict(metadata["data_audit"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    audit["source"] = "prepared_devices_cache"
    audit["prepared_devices_cache_path"] = str(cache_path)
    return records, audit


def write_automatic_device_cache(
    records: pd.DataFrame,
    cache_path: Path,
    signature: dict[str, Any],
    data_audit: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = device_cache_metadata_path(cache_path)
    temporary_cache = cache_path.with_name(cache_path.name + ".tmp")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    records.loc[:, DEVICE_COLUMNS].to_csv(temporary_cache, index=False, lineterminator="\n")
    metadata = {
        "signature": signature,
        "cache_sha256": sha256_file(temporary_cache),
        "data_audit": {key: value for key, value in data_audit.items() if key != "source"},
    }
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_cache.replace(cache_path)
    temporary_metadata.replace(metadata_path)


def load_device_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path = resolve_project_path(config["data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"PCE CSV not found: {data_path}")
    cache_path = resolve_project_path(
        config.get("prepared_devices_cache_path", "data/processed/device_records.csv")
    )
    signature = device_cache_signature(data_path, config)
    cached = load_automatic_device_cache(cache_path, signature)
    if cached is not None:
        return cached

    raw_frame = pd.read_csv(data_path, low_memory=False)
    records, audit = prepare_device_table(
        raw_frame,
        donor_col=config.get("donor_column", "donor_smiles"),
        acceptor_col=config.get("acceptor_column", "acceptor_smiles"),
        target_col=config.get("target_column", "pce"),
    )
    audit["source"] = "raw_csv"
    write_automatic_device_cache(records, cache_path, signature, audit)
    audit["prepared_devices_cache_path"] = str(cache_path)
    return records, audit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
            "CUDA is required, but PyTorch cannot access it; no CPU fallback is permitted."
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
        "precision": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def estimate_graph_memory_bytes(graphs: dict[str, dgl.DGLGraph]) -> int:
    total = 0
    for graph in graphs.values():
        tensors = [graph.ndata[name] for name in graph.ndata.keys()]
        tensors.extend(graph.edata[name] for name in graph.edata.keys())
        total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        total += graph.num_edges() * 2 * torch.tensor([], dtype=graph.idtype).element_size()
        total += (graph.num_nodes() + 1) * torch.tensor([], dtype=graph.idtype).element_size()
    return int(total)


def preload_graphs_to_cuda(
    graphs: dict[str, dgl.DGLGraph],
    device: torch.device,
    max_free_fraction: float,
    safety_factor: float,
) -> tuple[dict[str, dgl.DGLGraph], dict[str, Any]]:
    if device.type != "cuda":
        raise ValueError("graph preloading requires a CUDA device")
    if not 0 < max_free_fraction <= 1 or safety_factor < 1:
        raise ValueError("invalid graph preload memory policy")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    raw_bytes = estimate_graph_memory_bytes(graphs)
    required_bytes = math.ceil(raw_bytes * safety_factor)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    budget_bytes = int(free_bytes * max_free_fraction)
    if required_bytes > budget_bytes:
        raise RuntimeError(
            "GPU graph preload memory check failed: "
            f"estimated {required_bytes / 1024**2:.1f} MB with safety factor, "
            f"budget {budget_bytes / 1024**2:.1f} MB from {free_bytes / 1024**2:.1f} MB free."
        )

    before = torch.cuda.memory_allocated(device_index)
    try:
        preloaded = {smiles: graph.to(device) for smiles, graph in graphs.items()}
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA graph preloading failed after passing the conservative memory check; "
            "the run was stopped without a CPU fallback."
        ) from error
    actual = max(0, torch.cuda.memory_allocated(device_index) - before)
    if any(graph.device.type != "cuda" for graph in preloaded.values()):
        raise RuntimeError("not every molecular graph was preloaded to CUDA")
    return preloaded, {
        "all_graphs_preloaded": True,
        "graph_count": len(preloaded),
        "estimated_raw_mb": round(raw_bytes / 1024**2, 3),
        "safety_factor": float(safety_factor),
        "required_with_safety_mb": round(required_bytes / 1024**2, 3),
        "free_before_mb": round(free_bytes / 1024**2, 3),
        "total_gpu_memory_mb": round(total_bytes / 1024**2, 3),
        "memory_budget_mb": round(budget_bytes / 1024**2, 3),
        "actual_allocated_mb": round(actual / 1024**2, 3),
    }


def make_loader(
    records: pd.DataFrame,
    graphs: dict[str, dgl.DGLGraph],
    numeric_context: torch.Tensor,
    categorical_context: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DeviceGraphDataset(records, graphs, numeric_context, categorical_context),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_device_graphs,
        generator=generator,
        drop_last=False,
    )


def _split_audit(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        "rows": {name: int(len(part)) for name, part in splits.items()},
        "unique_pairs": {
            name: int(part[["donor_smiles", "acceptor_smiles"]].drop_duplicates().shape[0])
            for name, part in splits.items()
        },
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if "torch_num_threads" in config:
        torch.set_num_threads(int(config["torch_num_threads"]))

    device = resolve_device(config.get("device"))
    runtime = runtime_metadata(device)
    if not bool(config.get("preload_graphs_to_gpu", True)):
        raise ValueError("Production configurations must set preload_graphs_to_gpu: true")
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError("CUDA-preloaded DGL graphs require num_workers: 0")

    output_dir = resolve_project_path(config.get("output_dir", "outputs/pce_multimodal"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_project_path(config["data_path"])
    graph_cache_path = resolve_project_path(
        config.get("graph_cache_path", "data/processed/pce_topology_graphs.pt")
    )
    print(
        "Run contract | "
        f"device={runtime['device']} ({runtime['gpu_name']}); precision=float32; "
        f"deterministic CUDA=true; graph preload=required; batch={int(config.get('batch_size', 64))}; "
        f"epochs<={int(config.get('epochs', 100))}; output={portable_result_path(output_dir)}"
    )
    print(
        "RDKit topology construction uses CPU because RDKit has no CUDA graph builder; "
        "its versioned result is cached. All neural-network work remains on CUDA."
    )

    records, data_audit = load_device_data(config)
    print(
        f"Device data | rows={len(records)}; pairs={data_audit['unique_pairs']}; "
        f"molecules={data_audit['unique_molecules']}; source={data_audit['source']}"
    )
    required_smiles = pd.unique(
        pd.concat([records["donor_smiles"], records["acceptor_smiles"]], ignore_index=True)
    )
    graph_started = time.perf_counter()
    cpu_graphs, graph_audit = build_topology_graph_cache(
        required_smiles,
        graph_cache_path,
        encoder_atom=config.get("encoder_atom", "cgcnn"),
        encoder_bond=config.get("encoder_bond", "dim_14"),
    )
    graph_audit["cache_path"] = portable_result_path(graph_cache_path)
    graph_audit["elapsed_seconds"] = round(time.perf_counter() - graph_started, 3)
    usable_mask = records["donor_smiles"].isin(cpu_graphs) & records["acceptor_smiles"].isin(cpu_graphs)
    usable_records = records.loc[usable_mask].reset_index(drop=True).copy()
    graph_audit["usable_device_rows"] = int(len(usable_records))
    graph_audit["dropped_device_rows"] = int(len(records) - len(usable_records))
    graph_audit["usable_pairs"] = int(
        usable_records[["donor_smiles", "acceptor_smiles"]].drop_duplicates().shape[0]
    )
    success_rate = 100.0 * int(graph_audit["usable_molecules"]) / max(1, len(required_smiles))
    print(
        f"Topology graphs | usable={graph_audit['usable_molecules']}/{len(required_smiles)} "
        f"({success_rate:.2f}%); device rows={len(usable_records)}/{len(records)}; "
        f"CPU cache stage={graph_audit['elapsed_seconds']:.1f}s"
    )
    if len(usable_records) < 10:
        raise ValueError("fewer than 10 device records remain after graph validation")

    splits = split_device_table_by_pair(
        usable_records,
        train_ratio=float(config.get("train_ratio", 0.8)),
        validation_ratio=float(config.get("validation_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
        seed=seed,
    )
    split_audit = _split_audit(splits)
    split_table = pd.concat(
        [part.assign(split=name) for name, part in splits.items()], ignore_index=True
    )
    split_table.to_csv(output_dir / "device_records_with_split.csv", index=False, lineterminator="\n")
    print(
        "Pair-grouped split | "
        + ", ".join(
            f"{name}={split_audit['rows'][name]} rows/{split_audit['unique_pairs'][name]} pairs"
            for name in ("train", "validation", "test")
        )
    )

    preprocessor = ContextPreprocessor(
        min_category_frequency=int(config.get("min_category_frequency", 2))
    ).fit(splits["train"])
    context_tensors = {}
    for name, part in splits.items():
        numeric, categorical = preprocessor.transform(part)
        context_tensors[name] = (
            numeric.to(device=device, dtype=torch.float32),
            categorical.to(device=device, dtype=torch.long),
        )

    graphs, preload_audit = preload_graphs_to_cuda(
        cpu_graphs,
        device,
        max_free_fraction=float(config.get("graph_preload_max_free_fraction", 0.6)),
        safety_factor=float(config.get("graph_preload_safety_factor", 2.0)),
    )
    del cpu_graphs
    print(
        f"CUDA preload | {preload_audit['graph_count']} graphs; actual="
        f"{preload_audit['actual_allocated_mb']:.1f} MB; budget={preload_audit['memory_budget_mb']:.1f} MB"
    )

    batch_size = int(config.get("batch_size", 64))
    loaders = {
        name: make_loader(
            part,
            graphs,
            context_tensors[name][0],
            context_tensors[name][1],
            batch_size=batch_size,
            shuffle=name == "train",
            seed=seed,
        )
        for name, part in splits.items()
    }
    scaler = TargetScaler.fit(splits["train"]["pce"].to_numpy(dtype=np.float32))
    use_context = bool(config.get("use_context", True))
    model_config = {
        "in_feat": int(config.get("in_feat", 113)),
        "hidden_feat": int(config.get("hidden_feat", 64)),
        "fusion_hidden": int(config.get("fusion_hidden", 32)),
        "grid_feat": int(config.get("grid_feat", 1)),
        "num_layers": int(config.get("num_layers", 4)),
        "pooling": config.get("pooling", "avg"),
        "use_bias": bool(config.get("use_bias", True)),
        "numeric_context_dim": preprocessor.numeric_output_dim,
        "category_sizes": preprocessor.category_sizes,
        "context_hidden": int(config.get("context_hidden", 32)),
        "use_context": use_context,
    }
    torch.cuda.reset_peak_memory_stats(device)
    model = DualKAGNNRegressor(**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Model | mode={'multimodal' if use_context else 'material-only'}; "
        f"parameters={parameter_count}; PyTorch={runtime['torch_version']}; DGL={runtime['dgl_version']}; "
        f"CUDA runtime={runtime['cuda_runtime']}"
    )

    epochs = int(config.get("epochs", 100))
    patience_limit = int(config.get("patience", epochs))
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, epochs + 1):
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scaler, device)
        validation = evaluate(model, loaders["validation"], scaler, device)
        torch.cuda.synchronize(device)
        metrics = validation["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation["loss"]),
            "validation_mae": float(metrics["mae"]),
            "validation_rmse": float(metrics["rmse"]),
            "validation_r2": float(metrics["r2"]),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | {row['epoch_seconds']:.2f}s | train MSE(z)={train_loss:.5f} | "
            f"val MAE={row['validation_mae']:.5f} | val RMSE={row['validation_rmse']:.5f} | "
            f"val R2={row['validation_r2']:.5f}"
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
                    "context_preprocessor": preprocessor.to_dict(),
                    "feature_contract": {
                        "numeric_sources": list(NUMERIC_SOURCE_COLUMNS),
                        "numeric_features": list(NUMERIC_FEATURES),
                        "categorical_features": list(CATEGORICAL_FEATURES),
                    },
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

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, lineterminator="\n")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate(model, loaders["test"], scaler, device)
    prediction_table = splits["test"].copy()
    prediction_table["predicted_pce"] = test_result["predictions"]
    prediction_table["absolute_error"] = np.abs(
        prediction_table["pce"] - prediction_table["predicted_pce"]
    )
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False, lineterminator="\n")

    runtime["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 3)
    summary = {
        "task": "per-device multimodal PCE regression" if use_context else "per-device material-only PCE regression",
        "data_path": portable_result_path(data_path),
        "output_dir": portable_result_path(output_dir),
        "device": runtime["device"],
        "runtime": runtime,
        "data_audit": portable_audit_paths(data_audit),
        "graph_audit": graph_audit,
        "gpu_preload_audit": preload_audit,
        "split_audit": split_audit,
        "feature_contract": {
            "numeric_sources": list(NUMERIC_SOURCE_COLUMNS),
            "numeric_features_after_parsing": list(NUMERIC_FEATURES),
            "numeric_missing_masks": True,
            "categorical_features": list(CATEGORICAL_FEATURES),
            "preprocessor_fitted_on": "train_only",
            "target_derived_features_excluded": ["voc", "jsc", "ff", "pce_recomputed", "pce_avg", "pce_best"],
        },
        "target_scaler": scaler.to_dict(),
        "model_config": {**model_config, "category_sizes": list(model_config["category_sizes"])},
        "model_parameters": int(parameter_count),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "test_metrics": test_result["metrics"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    metrics = summary["test_metrics"]
    print(
        f"Test | MAE={metrics['mae']:.5f} | RMSE={metrics['rmse']:.5f} | R2={metrics['r2']:.5f} | "
        f"elapsed={summary['elapsed_seconds']:.1f}s | peak GPU={runtime['peak_gpu_memory_mb']:.1f} MB"
    )
    print(f"Results saved to: {output_dir}")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal KA-GNN PCE regression")
    parser.add_argument("--config", default="config/pce.yaml", help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run(load_config(arguments.config))


if __name__ == "__main__":
    main()
