from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset

from utils.graph_path import path_complex_mol


PAIR_COLUMNS = ("donor_smiles", "acceptor_smiles")
DEVICE_NUMERIC_COLUMNS = (
    "homo_d",
    "lumo_d",
    "homo_a",
    "lumo_a",
    "active_layer_thickness",
    "annealing_temp",
    "d_a_ratio",
    "additive_ratio",
)
DEVICE_CATEGORICAL_COLUMNS = (
    "device_type",
    "etl_canonical",
    "htl_canonical",
    "solvent_canonical",
    "additive_canonical",
)


def canonicalize_smiles(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def prepare_pair_table(
    frame: pd.DataFrame,
    donor_col: str = "donor_smiles",
    acceptor_col: str = "acceptor_smiles",
    target_col: str = "pce",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {donor_col, acceptor_col, target_col}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")

    selected = frame[[donor_col, acceptor_col, target_col]].copy()
    selected.columns = ["donor_smiles", "acceptor_smiles", "pce"]
    input_rows = len(selected)
    selected["pce"] = pd.to_numeric(selected["pce"], errors="coerce")

    present = (
        selected["donor_smiles"].notna()
        & selected["acceptor_smiles"].notna()
        & selected["pce"].notna()
        & np.isfinite(selected["pce"])
        & selected["donor_smiles"].astype(str).str.strip().ne("")
        & selected["acceptor_smiles"].astype(str).str.strip().ne("")
    )
    selected = selected.loc[present].copy()
    missing_or_non_numeric_rows = input_rows - len(selected)

    smiles_values = pd.unique(
        pd.concat(
            [selected["donor_smiles"], selected["acceptor_smiles"]],
            ignore_index=True,
        )
    )
    canonical = {value: canonicalize_smiles(value) for value in smiles_values}
    selected["donor_smiles"] = selected["donor_smiles"].map(canonical)
    selected["acceptor_smiles"] = selected["acceptor_smiles"].map(canonical)

    valid_smiles = selected["donor_smiles"].notna() & selected["acceptor_smiles"].notna()
    invalid_smiles_rows = int((~valid_smiles).sum())
    selected = selected.loc[valid_smiles].copy()

    grouped = (
        selected.groupby(list(PAIR_COLUMNS), as_index=False, sort=True)
        .agg(
            pce=("pce", "median"),
            replicate_count=("pce", "size"),
            pce_std=("pce", "std"),
            pce_min=("pce", "min"),
            pce_max=("pce", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["pce_std"] = grouped["pce_std"].fillna(0.0)

    audit = {
        "input_rows": int(input_rows),
        "missing_or_non_numeric_rows": int(missing_or_non_numeric_rows),
        "invalid_smiles_rows": invalid_smiles_rows,
        "valid_rows": int(len(selected)),
        "unique_pairs": int(len(grouped)),
        "collapsed_replicate_rows": int(len(selected) - len(grouped)),
    }
    return grouped, audit


def prepare_device_table(
    frame: pd.DataFrame,
    donor_col: str = "donor_smiles",
    acceptor_col: str = "acceptor_smiles",
    target_col: str = "pce",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {donor_col, acceptor_col, target_col}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")

    input_rows = len(frame)
    selected = pd.DataFrame(index=frame.index)
    selected["record_id"] = (
        frame["id"] if "id" in frame.columns else np.arange(input_rows, dtype=int)
    )
    selected["doi"] = frame["doi"] if "doi" in frame.columns else pd.NA
    selected["donor_smiles"] = frame[donor_col]
    selected["acceptor_smiles"] = frame[acceptor_col]
    selected["pce"] = pd.to_numeric(frame[target_col], errors="coerce")
    for column in DEVICE_NUMERIC_COLUMNS:
        if column in {"d_a_ratio", "additive_ratio"}:
            selected[column] = frame[column] if column in frame.columns else pd.NA
        else:
            selected[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                if column in frame.columns
                else np.nan
            )
    for column in DEVICE_CATEGORICAL_COLUMNS:
        selected[column] = frame[column] if column in frame.columns else pd.NA

    present = (
        selected["donor_smiles"].notna()
        & selected["acceptor_smiles"].notna()
        & selected["pce"].notna()
        & np.isfinite(selected["pce"])
        & selected["donor_smiles"].astype(str).str.strip().ne("")
        & selected["acceptor_smiles"].astype(str).str.strip().ne("")
    )
    selected = selected.loc[present].copy()
    missing_or_non_numeric_rows = input_rows - len(selected)

    unique_smiles = pd.unique(
        pd.concat(
            [selected["donor_smiles"], selected["acceptor_smiles"]],
            ignore_index=True,
        )
    )
    canonical = {value: canonicalize_smiles(value) for value in unique_smiles}
    selected["donor_smiles"] = selected["donor_smiles"].map(canonical)
    selected["acceptor_smiles"] = selected["acceptor_smiles"].map(canonical)
    valid_smiles = selected["donor_smiles"].notna() & selected["acceptor_smiles"].notna()
    invalid_smiles_rows = int((~valid_smiles).sum())
    selected = selected.loc[valid_smiles].reset_index(drop=True)

    audit = {
        "input_rows": int(input_rows),
        "missing_or_non_numeric_rows": int(missing_or_non_numeric_rows),
        "invalid_smiles_rows": invalid_smiles_rows,
        "usable_device_rows": int(len(selected)),
        "unique_pairs": int(selected[list(PAIR_COLUMNS)].drop_duplicates().shape[0]),
        "unique_molecules": int(
            len(set(selected["donor_smiles"]).union(selected["acceptor_smiles"]))
        ),
    }
    return selected, audit


def split_device_table_by_pair(
    device_table: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")

    pair_table = device_table[list(PAIR_COLUMNS)].drop_duplicates().reset_index(drop=True)
    active_splits = int(np.count_nonzero(ratios))
    if len(pair_table) < active_splits:
        raise ValueError("not enough unique molecular pairs to populate every requested split")

    counts = np.floor(len(pair_table) * ratios).astype(int)
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            counts[index] = 1
    counts[2] += len(pair_table) - int(counts.sum())
    if counts[2] < 0:
        counts[0] += counts[2]
        counts[2] = 0
    if np.any((ratios > 0) & (counts == 0)):
        raise ValueError("split ratios produce an empty requested split")

    shuffled = pair_table.iloc[
        np.random.default_rng(seed).permutation(len(pair_table))
    ].reset_index(drop=True)
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    pair_partitions = {
        "train": shuffled.iloc[:train_end],
        "validation": shuffled.iloc[train_end:validation_end],
        "test": shuffled.iloc[validation_end:],
    }
    row_keys = pd.MultiIndex.from_frame(device_table[list(PAIR_COLUMNS)])
    splits: Dict[str, pd.DataFrame] = {}
    for name, pairs in pair_partitions.items():
        allowed = pd.MultiIndex.from_frame(pairs[list(PAIR_COLUMNS)])
        splits[name] = device_table.loc[row_keys.isin(allowed)].reset_index(drop=True).copy()
    return splits


def split_pair_table(
    pair_table: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")

    active_splits = int(np.count_nonzero(ratios))
    if len(pair_table) < active_splits:
        raise ValueError("not enough unique pairs to populate every requested split")

    counts = np.floor(len(pair_table) * ratios).astype(int)
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            counts[index] = 1
    counts[2] += len(pair_table) - int(counts.sum())
    if counts[2] < 0:
        counts[0] += counts[2]
        counts[2] = 0
    if np.any((ratios > 0) & (counts == 0)):
        raise ValueError("split ratios produce an empty requested split")

    indices = np.random.default_rng(seed).permutation(len(pair_table))
    train_end = counts[0]
    validation_end = train_end + counts[1]
    partitions = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }
    return {
        name: pair_table.iloc[part].reset_index(drop=True).copy()
        for name, part in partitions.items()
    }


def add_edge_aggregates(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    if "agg_feats" in graph.ndata:
        return graph
    if "feat" not in graph.ndata or "feat" not in graph.edata:
        raise ValueError("graph must contain node and edge features named 'feat'")

    edge_width = int(graph.edata["feat"].shape[1])
    if graph.num_edges() == 0:
        graph.ndata["agg_feats"] = torch.zeros(
            graph.num_nodes(),
            edge_width,
            dtype=graph.ndata["feat"].dtype,
            device=graph.ndata["feat"].device,
        )
    else:
        graph.update_all(fn.copy_e("feat", "m"), fn.mean("m", "agg_feats"))
    graph.ndata["feat"] = torch.cat(
        [graph.ndata["feat"], graph.ndata["agg_feats"]],
        dim=1,
    )
    return graph


def graph_has_finite_features(graph: dgl.DGLGraph) -> bool:
    """Return whether every stored node/edge tensor contains finite values."""
    feature_tensors = [graph.ndata[name] for name in graph.ndata.keys()]
    feature_tensors.extend(graph.edata[name] for name in graph.edata.keys())
    return bool(feature_tensors) and all(
        bool(torch.isfinite(tensor).all()) for tensor in feature_tensors
    )


def build_graph_cache(
    pair_table: pd.DataFrame,
    cache_path: str | Path,
    encoder_atom: str,
    encoder_bond: str,
    graph_builder: Callable[[str, str, str], dgl.DGLGraph | bool] | None = None,
) -> tuple[dict[str, dgl.DGLGraph], pd.DataFrame, dict[str, int]]:
    cache_path = Path(cache_path)
    metadata = {
        "version": 1,
        "encoder_atom": encoder_atom,
        "encoder_bond": encoder_bond,
    }
    graphs: dict[str, dgl.DGLGraph] = {}
    failed_smiles: set[str] = set()
    loaded_cached_molecules = 0
    invalid_cached_molecules = 0
    cache_is_current = False
    cache_dirty = False

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") == metadata:
            cache_is_current = True
            graphs = payload.get("graphs", {})
            failed_smiles = set(payload.get("failed_smiles", []))
            invalid_cached = {
                smiles
                for smiles, graph in graphs.items()
                if not graph_has_finite_features(graph)
            }
            for smiles in invalid_cached:
                del graphs[smiles]
            failed_smiles.update(invalid_cached)
            invalid_cached_molecules = len(invalid_cached)
            loaded_cached_molecules = len(graphs)
            cache_dirty = bool(invalid_cached)

    if not cache_is_current:
        cache_dirty = True

    required_smiles = sorted(
        set(pair_table["donor_smiles"]).union(pair_table["acceptor_smiles"])
    )
    builder = graph_builder or path_complex_mol
    built_molecules = 0
    for smiles in required_smiles:
        if smiles in graphs or smiles in failed_smiles:
            continue
        try:
            graph = builder(smiles, encoder_atom, encoder_bond)
            if graph is False or graph is None:
                failed_smiles.add(smiles)
                cache_dirty = True
                continue
            graph = add_edge_aggregates(graph)
            if not graph_has_finite_features(graph):
                failed_smiles.add(smiles)
                cache_dirty = True
                continue
            graphs[smiles] = graph
            built_molecules += 1
            cache_dirty = True
        except Exception:
            failed_smiles.add(smiles)
            cache_dirty = True

    usable_mask = pair_table["donor_smiles"].isin(graphs) & pair_table["acceptor_smiles"].isin(graphs)
    usable_pairs = pair_table.loc[usable_mask].reset_index(drop=True).copy()

    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata,
                "graphs": graphs,
                "failed_smiles": sorted(failed_smiles),
            },
            cache_path,
        )
    audit = {
        "requested_unique_molecules": int(len(required_smiles)),
        "loaded_cached_molecules": int(loaded_cached_molecules),
        "invalid_cached_molecules": int(invalid_cached_molecules),
        "built_molecules": int(built_molecules),
        "failed_molecules": int(len(set(required_smiles) & failed_smiles)),
        "usable_pairs": int(len(usable_pairs)),
        "dropped_pairs_for_graph_failure": int(len(pair_table) - len(usable_pairs)),
    }
    return graphs, usable_pairs, audit


class PairGraphDataset(Dataset):
    def __init__(self, pair_table: pd.DataFrame, graphs: dict[str, dgl.DGLGraph]) -> None:
        self.pairs = pair_table.reset_index(drop=True).copy()
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.pairs.iloc[index]
        donor_smiles = row["donor_smiles"]
        acceptor_smiles = row["acceptor_smiles"]
        return {
            "donor_graph": self.graphs[donor_smiles],
            "acceptor_graph": self.graphs[acceptor_smiles],
            "target": float(row["pce"]),
            "donor_smiles": donor_smiles,
            "acceptor_smiles": acceptor_smiles,
        }


def collate_pair_graphs(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty pair batch")
    return {
        "donor_graph": dgl.batch([sample["donor_graph"] for sample in samples]),
        "acceptor_graph": dgl.batch([sample["acceptor_graph"] for sample in samples]),
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
        "donor_smiles": [sample["donor_smiles"] for sample in samples],
        "acceptor_smiles": [sample["acceptor_smiles"] for sample in samples],
    }


class DeviceGraphDataset(Dataset):
    """Row-level PCE records with molecular graphs and fitted context tensors."""

    def __init__(
        self,
        device_table: pd.DataFrame,
        graphs: dict[str, dgl.DGLGraph],
        numeric_context: torch.Tensor,
        categorical_context: torch.Tensor,
    ) -> None:
        self.devices = device_table.reset_index(drop=True).copy()
        self.graphs = graphs
        if numeric_context.ndim != 2 or categorical_context.ndim != 2:
            raise ValueError("context tensors must have shape [records, features]")
        if len(self.devices) != numeric_context.shape[0] or len(self.devices) != categorical_context.shape[0]:
            raise ValueError("device rows and context tensors must have the same length")
        if numeric_context.device != categorical_context.device:
            raise ValueError("numeric and categorical context tensors must use the same device")
        self.numeric_context = numeric_context
        self.categorical_context = categorical_context
        self.targets = torch.as_tensor(
            self.devices["pce"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=numeric_context.device,
        )

    def __len__(self) -> int:
        return len(self.devices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.devices.iloc[index]
        donor_smiles = row["donor_smiles"]
        acceptor_smiles = row["acceptor_smiles"]
        return {
            "donor_graph": self.graphs[donor_smiles],
            "acceptor_graph": self.graphs[acceptor_smiles],
            "numeric_context": self.numeric_context[index],
            "categorical_context": self.categorical_context[index],
            "target": self.targets[index],
            "record_id": row.get("record_id"),
            "doi": row.get("doi"),
            "donor_smiles": donor_smiles,
            "acceptor_smiles": acceptor_smiles,
        }


def collate_device_graphs(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty device batch")
    return {
        "donor_graph": dgl.batch([sample["donor_graph"] for sample in samples]),
        "acceptor_graph": dgl.batch([sample["acceptor_graph"] for sample in samples]),
        "numeric_context": torch.stack([sample["numeric_context"] for sample in samples]),
        "categorical_context": torch.stack(
            [sample["categorical_context"] for sample in samples]
        ),
        "target": torch.stack([sample["target"] for sample in samples]),
        "record_id": [sample["record_id"] for sample in samples],
        "doi": [sample["doi"] for sample in samples],
        "donor_smiles": [sample["donor_smiles"] for sample in samples],
        "acceptor_smiles": [sample["acceptor_smiles"] for sample in samples],
    }
