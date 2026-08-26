from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import dgl
import dgl.function as fn
import torch
from rdkit import Chem, rdBase

from utils.graph_path import encode_bond_14, get_node_attributes


TOPOLOGY_CACHE_VERSION = 4
MAX_HEAVY_ATOMS = 500
NODE_FEATURE_WIDTH = 113
ATOM_FEATURE_WIDTH = 92
EDGE_FEATURE_WIDTH = 21


def graph_has_finite_features(graph: dgl.DGLGraph) -> bool:
    tensors = [graph.ndata[name] for name in graph.ndata.keys()]
    tensors.extend(graph.edata[name] for name in graph.edata.keys())
    return bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _add_incident_edge_features(graph: dgl.DGLGraph) -> None:
    if graph.num_edges() == 0:
        aggregate = torch.zeros(
            graph.num_nodes(),
            EDGE_FEATURE_WIDTH,
            dtype=graph.ndata["feat"].dtype,
        )
    else:
        graph.update_all(fn.copy_e("feat", "m"), fn.mean("m", "edge_mean"))
        aggregate = graph.ndata.pop("edge_mean")
    graph.ndata["feat"] = torch.cat([graph.ndata["feat"], aggregate], dim=1)


def build_topology_graph(
    smiles: str,
    encoder_atom: str = "cgcnn",
    encoder_bond: str = "dim_14",
) -> dgl.DGLGraph:
    if encoder_atom != "cgcnn":
        raise ValueError(f"unsupported atom encoder for topology graphs: {encoder_atom}")
    if encoder_bond != "dim_14":
        raise ValueError(f"unsupported bond encoder for topology graphs: {encoder_bond}")
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
    if molecule is None:
        raise ValueError("RDKit could not parse the SMILES")
    dummy_atoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if dummy_atoms:
        raise ValueError(
            "dummy atom placeholders cannot be assigned a physical element feature; "
            f"atom indices={dummy_atoms}"
        )
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
    if heavy_atoms > MAX_HEAVY_ATOMS:
        raise ValueError(
            f"molecule exceeds the {MAX_HEAVY_ATOMS}-heavy-atom production complexity limit; "
            f"observed={heavy_atoms}"
        )
    with rdBase.BlockLogs():
        Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule)

    atom_features = torch.tensor(
        [list(get_node_attributes(atom.GetSymbol(), atom_features=encoder_atom)) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    if atom_features.ndim != 2 or atom_features.shape[1] != ATOM_FEATURE_WIDTH:
        raise ValueError(
            f"atom encoder produced {tuple(atom_features.shape)}; "
            f"expected (*, {ATOM_FEATURE_WIDTH})"
        )

    sources: list[int] = []
    destinations: list[int] = []
    encoded_edges: list[list[float]] = []
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        destination = bond.GetEndAtomIdx()
        feature = list(encode_bond_14(bond))
        if len(feature) != EDGE_FEATURE_WIDTH:
            raise ValueError(
                f"bond encoder produced {len(feature)} values; expected {EDGE_FEATURE_WIDTH}"
            )
        sources.extend([source, destination])
        destinations.extend([destination, source])
        encoded_edges.extend([feature, feature])

    graph = dgl.graph(
        (sources, destinations),
        num_nodes=molecule.GetNumAtoms(),
    )
    graph.ndata["feat"] = atom_features
    graph.edata["feat"] = (
        torch.tensor(encoded_edges, dtype=torch.float32)
        if encoded_edges
        else torch.empty((0, EDGE_FEATURE_WIDTH), dtype=torch.float32)
    )
    _add_incident_edge_features(graph)
    if tuple(graph.ndata["feat"].shape[1:]) != (NODE_FEATURE_WIDTH,):
        raise ValueError(f"topology node features must have width {NODE_FEATURE_WIDTH}")
    if not graph_has_finite_features(graph):
        raise ValueError("topology graph contains non-finite features")
    return graph


def _configure_graph_worker() -> None:
    torch.set_num_threads(1)


def _build_default_graph_task(
    arguments: tuple[str, str, str],
) -> tuple[str, dgl.DGLGraph | None, str | None]:
    smiles, encoder_atom, encoder_bond = arguments
    try:
        graph = build_topology_graph(smiles, encoder_atom, encoder_bond)
        if not graph_has_finite_features(graph):
            raise ValueError("graph contains non-finite features")
        return smiles, graph, None
    except Exception as error:
        return smiles, None, f"{type(error).__name__}: {error}"


def build_topology_graph_cache(
    smiles_values: Iterable[str],
    cache_path: str | Path,
    encoder_atom: str,
    encoder_bond: str,
    graph_builder: Callable[[str, str, str], dgl.DGLGraph] | None = None,
    num_workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    checkpoint_interval: int = 0,
) -> tuple[dict[str, dgl.DGLGraph], dict[str, object]]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_workers > 1 and graph_builder is not None:
        raise ValueError("parallel cache construction requires the production graph builder")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval cannot be negative")
    cache_path = Path(cache_path)
    metadata = {
        "version": TOPOLOGY_CACHE_VERSION,
        "builder": "rdkit_topology_explicit_hydrogen",
        "encoder_atom": encoder_atom,
        "encoder_bond": encoder_bond,
        "node_feature_width": NODE_FEATURE_WIDTH,
        "edge_feature_width": EDGE_FEATURE_WIDTH,
        "max_heavy_atoms": MAX_HEAVY_ATOMS,
    }
    graphs: dict[str, dgl.DGLGraph] = {}
    failure_reasons: dict[str, str] = {}
    cache_current = False
    cache_dirty = False
    loaded_cached_molecules = 0

    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") == metadata:
            cache_current = True
            graphs = dict(payload.get("graphs", {}))
            failure_reasons = dict(payload.get("failure_reasons", {}))
            invalid_cached = [
                smiles
                for smiles, graph in graphs.items()
                if tuple(graph.ndata.get("feat", torch.empty(0)).shape[1:])
                != (NODE_FEATURE_WIDTH,)
                or not graph_has_finite_features(graph)
            ]
            for smiles in invalid_cached:
                del graphs[smiles]
                failure_reasons[smiles] = "cached graph failed finite feature validation"
            cache_dirty = bool(invalid_cached)
            loaded_cached_molecules = len(graphs)
    if not cache_current:
        cache_dirty = True

    requested = sorted({str(value).strip() for value in smiles_values if str(value).strip()})
    builder = graph_builder or build_topology_graph
    pending = [
        smiles for smiles in requested if smiles not in graphs and smiles not in failure_reasons
    ]

    def build_one(smiles: str) -> tuple[str, dgl.DGLGraph | None, str | None]:
        try:
            graph = builder(smiles, encoder_atom, encoder_bond)
            if not graph_has_finite_features(graph):
                raise ValueError("graph contains non-finite features")
            return smiles, graph, None
        except Exception as error:
            return smiles, None, f"{type(error).__name__}: {error}"

    built_molecules = 0

    def save_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(cache_path.name + ".tmp")
        torch.save(
            {
                "metadata": metadata,
                "graphs": {smiles: graphs[smiles] for smiles in sorted(graphs)},
                "failure_reasons": {
                    smiles: failure_reasons[smiles] for smiles in sorted(failure_reasons)
                },
            },
            temporary_path,
        )
        temporary_path.replace(cache_path)

    def retain_result(result: tuple[str, dgl.DGLGraph | None, str | None], completed: int) -> None:
        nonlocal built_molecules, cache_dirty
        smiles, graph, failure = result
        if graph is not None:
            graphs[smiles] = graph
            built_molecules += 1
        else:
            failure_reasons[smiles] = str(failure)
        cache_dirty = True
        if progress_callback is not None:
            progress_callback(completed, len(pending))
        if checkpoint_interval and completed % checkpoint_interval == 0:
            save_cache()

    if num_workers == 1:
        for completed, result in enumerate(map(build_one, pending), start=1):
            retain_result(result, completed)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_configure_graph_worker,
        ) as executor:
            futures = [
                executor.submit(
                    _build_default_graph_task,
                    (smiles, encoder_atom, encoder_bond),
                )
                for smiles in pending
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                retain_result(future.result(), completed)

    graphs = {smiles: graphs[smiles] for smiles in sorted(graphs)}
    failure_reasons = {smiles: failure_reasons[smiles] for smiles in sorted(failure_reasons)}

    if cache_dirty:
        save_cache()

    requested_failures = {
        smiles: failure_reasons[smiles]
        for smiles in requested
        if smiles in failure_reasons
    }
    audit: dict[str, object] = {
        "builder": metadata["builder"],
        "num_workers": int(num_workers),
        "checkpoint_interval": int(checkpoint_interval),
        "requested_molecules": len(requested),
        "loaded_cached_molecules": loaded_cached_molecules,
        "built_molecules": built_molecules,
        "usable_molecules": sum(smiles in graphs for smiles in requested),
        "failed_molecules": len(requested_failures),
        "failure_reasons": requested_failures,
    }
    return {smiles: graphs[smiles] for smiles in requested if smiles in graphs}, audit
