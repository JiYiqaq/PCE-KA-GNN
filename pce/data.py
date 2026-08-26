from __future__ import annotations

from typing import Dict, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset


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
    selected["record_id"] = frame["id"] if "id" in frame.columns else np.arange(input_rows)
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
        pd.concat([selected["donor_smiles"], selected["acceptor_smiles"]], ignore_index=True)
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

    pair_table = (
        device_table.groupby(list(PAIR_COLUMNS), as_index=False, sort=True)
        .size()
        .rename(columns={"size": "row_count"})
    )
    active_splits = int(np.count_nonzero(ratios))
    if len(pair_table) < active_splits:
        raise ValueError("not enough unique molecular pairs to populate every requested split")

    split_names = ("train", "validation", "test")
    active_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]
    target_rows = ratios * len(device_table)
    assigned_rows = np.zeros(3, dtype=int)
    pair_assignments: list[int] = []
    rng = np.random.default_rng(seed)
    pair_table = pair_table.assign(_tie_break=rng.random(len(pair_table))).sort_values(
        ["row_count", "_tie_break"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    for pair_index, pair in pair_table.iterrows():
        remaining_pairs = len(pair_table) - pair_index
        empty_splits = [index for index in active_indices if assigned_rows[index] == 0]
        candidates = empty_splits if remaining_pairs == len(empty_splits) else active_indices
        deficits = target_rows - assigned_rows
        chosen = max(candidates, key=lambda index: (deficits[index], -index))
        pair_assignments.append(chosen)
        assigned_rows[chosen] += int(pair["row_count"])
    pair_table["_split_index"] = pair_assignments
    pair_partitions = {
        name: pair_table.loc[pair_table["_split_index"] == index, list(PAIR_COLUMNS)]
        for index, name in enumerate(split_names)
    }

    row_keys = pd.MultiIndex.from_frame(device_table[list(PAIR_COLUMNS)])
    splits: Dict[str, pd.DataFrame] = {}
    for name, pairs in pair_partitions.items():
        allowed = pd.MultiIndex.from_frame(pairs[list(PAIR_COLUMNS)])
        splits[name] = device_table.loc[row_keys.isin(allowed)].reset_index(drop=True).copy()
    return splits


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
        "categorical_context": torch.stack([sample["categorical_context"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "record_id": [sample["record_id"] for sample in samples],
        "doi": [sample["doi"] for sample in samples],
        "donor_smiles": [sample["donor_smiles"] for sample in samples],
        "acceptor_smiles": [sample["acceptor_smiles"] for sample in samples],
    }
