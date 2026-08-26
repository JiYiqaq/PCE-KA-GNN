# Multimodal PCE Regression Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production CUDA KA-GNN that predicts per-device PCE from robust donor/acceptor topology graphs and leakage-free experimental context.

**Architecture:** A deterministic topology graph builder supplies shared donor/acceptor KA-GNN encoders. A train-only context preprocessor produces numeric values, missingness masks, and categorical indices for a Fourier-KAN context encoder; pair and context embeddings feed the final regression head. Device rows are split by ordered molecular pair and all cache/output artifacts carry audit metadata.

**Tech Stack:** Python 3.10, RDKit, pandas, NumPy, PyTorch 2.1.2+cu118, DGL 2.2.1+cu118, unittest, YAML.

---

### Task 1: Robust topology graph builder

**Files:**
- Create: `pce/graphs.py`
- Create: `tests/test_pce_graphs.py`
- Modify: `utils/graph_path.py`

1. Write failing tests that require deterministic topology graphs for valid SMILES, 113 finite node features, 21 finite edge features, bidirectional bonds, useful failure reasons, and unchanged-cache reuse.
2. Run `python -m unittest tests.test_pce_graphs -v` and confirm the missing API failures.
3. Implement the minimal topology builder and versioned cache. Correct bond-length lookup to use the RDKit bond type string.
4. Run the focused and full test suites.
5. Commit `feat: build robust molecular topology graphs`.

### Task 2: Per-device data preparation and grouped split

**Files:**
- Modify: `pce/data.py`
- Modify: `tests/test_pce_data.py`

1. Write failing tests requiring repeated molecular pairs to remain separate, target-derived columns to be absent, finite PCE validation, and pair-disjoint grouped splits.
2. Confirm focused RED failures.
3. Implement `prepare_device_table` and a deterministic row-balanced pair-grouped split.
4. Run focused and full tests.
5. Commit `feat: preserve device records with pair-grouped splits`.

### Task 3: Leakage-free context preprocessing

**Files:**
- Create: `pce/context.py`
- Create: `tests/test_pce_context.py`

1. Write failing tests for ratio parsing, additive percentage parsing, train-only robust numeric scaling, missing masks, missing/unknown categorical separation, finite tensors, and serialization round trips.
2. Confirm focused RED failures.
3. Implement the minimal parsers and `ContextPreprocessor`.
4. Run focused and full tests.
5. Commit `feat: encode leakage-free device context`.

### Task 4: CUDA-ready device dataset and batching

**Files:**
- Modify: `pce/data.py`
- Modify: `tests/test_pce_data.py`

1. Write failing tests for per-device graph/context/target samples and batched tensor shapes.
2. Confirm RED.
3. Implement `DeviceGraphDataset` and `collate_device_graphs` without target-derived inputs.
4. Run focused and full tests.
5. Commit `feat: batch multimodal device records`.

### Task 5: Multimodal KA-GNN regressor

**Files:**
- Modify: `model/pce_ka_gnn.py`
- Modify: `tests/test_pce_model.py`

1. Write a CUDA failing test requiring finite per-device outputs, gradients through graph and context branches, unrestricted regression output, and a material-only ablation path.
2. Confirm RED on the verified GTX 1650 Ti environment.
3. Implement categorical embeddings, the Fourier-KAN context encoder, and multimodal fusion.
4. Run CUDA-focused and full tests.
5. Commit `feat: fuse molecular graphs with device context`.

### Task 6: Production training entry point

**Files:**
- Modify: `main_pce.py`
- Modify: `pce/training.py`
- Modify: `tests/test_main_pce.py`
- Modify: `tests/test_pce_training.py`

1. Write failing tests for device-cache fingerprinting, train-only preprocessing, CUDA graph preloading, checkpoint preprocessing metadata, portable summaries, and new output names.
2. Confirm RED.
3. Replace the active pair-median workflow with the per-device multimodal pipeline while retaining deterministic CUDA fail-fast behavior.
4. Run focused and full tests.
5. Commit `feat: train multimodal PCE models on CUDA`.

### Task 7: Configurations and project contract

**Files:**
- Modify: `config/pce.yaml`
- Modify: `config/pce_quick.yaml`
- Modify: `config/pce_smoke.yaml`
- Create: `config/pce_material_only.yaml`
- Modify: `tests/test_project_layout.py`

1. Write failing layout tests for the context fields, new caches, CUDA requirement, topology builder, and ablation configuration.
2. Confirm RED.
3. Update the configurations with identical data/split contracts and production CUDA defaults.
4. Run focused and full tests.
5. Commit `config: define multimodal and material-only experiments`.

### Task 8: Data/cache generation and GPU experiments

**Files:**
- Generate: `data/processed/device_records.csv`
- Generate: `data/processed/device_records.meta.json`
- Generate locally: `data/processed/pce_topology_graphs.pt`
- Generate locally: `outputs/pce_multimodal/`
- Generate locally: `outputs/pce_material_only/`

1. Verify `kagnn`, CUDA, DGL CUDA graph creation, GPU name, and free memory before execution.
2. Generate the fingerprinted device cache and topology graph cache; report elapsed time and graph success rate.
3. Run the production-stack smoke configuration on CUDA.
4. Benchmark safe graph preloading and batch size on GTX 1650 Ti without reducing precision or reproducibility.
5. Run formal multimodal and material-only experiments on the same grouped split.
6. Recompute metrics from exported predictions and verify runtime/audit metadata.

### Task 9: Documentation, results, and integration

**Files:**
- Modify: `README.md`
- Modify: `results/baseline/README.md`
- Create: `results/multimodal/` verified lightweight artifacts

1. Document the device-level task, included/excluded features, missing-data policy, grouped split, graph success audit, GPU command, and limitations.
2. Store verified summaries, histories, predictions, and checkpoints when within repository limits.
3. Run `python -m unittest discover -s tests -v`, `python -m pip check`, CUDA graph verification, `git diff --check`, cache fingerprint verification, and a fresh production command.
4. Use `superpowers:finishing-a-development-branch`, fast-forward merge to `main`, verify again, push `origin/main`, and clean the isolated worktree.

