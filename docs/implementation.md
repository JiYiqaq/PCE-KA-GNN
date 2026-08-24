# PCE Regression Implementation Plan

**Goal:** Build and verify an independent donor-plus-acceptor KA-GNN workflow for continuous PCE prediction.

**Architecture:** Canonicalize and aggregate ordered molecular pairs, cache one graph per unique molecule, encode both graphs with a shared Fourier KA-GNN encoder, fuse their embeddings, and train an unconstrained regression head. Keep the regression entry point and configuration self-contained.

**Tech Stack:** Python 3.10, PyTorch 2.0.1 CPU, DGL 2.0.0, RDKit, pandas, NumPy, scikit-learn, unittest.

---

### Task 1: Pair-table preparation

**Files:**
- Create: `pce/data.py`
- Create: `pce/__init__.py`
- Test: `tests/test_pce_data.py`

1. Write tests asserting that missing inputs are removed, SMILES are canonicalized, duplicate ordered pairs are reduced to median PCE, and replicate counts are retained.
2. Run `python -m unittest tests.test_pce_data -v`; expect a failure because `pce.data` is absent.
3. Implement `prepare_pair_table` and `canonicalize_smiles` with explicit column validation and a returned audit dictionary.
4. Add a test that `split_pair_table` covers every pair exactly once and produces disjoint pair-key sets.
5. Run the data tests; expect all to pass.

### Task 2: Shared dual-graph KA-GNN model

**Files:**
- Create: `model/pce_ka_gnn.py`
- Test: `tests/test_pce_model.py`

1. Write a failing test using two small batched DGL graphs with 113-dimensional node features.
2. Assert that `DualKAGNNRegressor` returns a finite `(batch_size,)` tensor and supports backpropagation.
3. Implement `KAGraphEncoder` from the author's `KAN_linear` and `NaiveFourierKANLayer` classes, followed by the ordered pair fusion and a KAN regression head without sigmoid.
4. Run the model tests; expect all to pass.

### Task 3: Regression utilities

**Files:**
- Create: `pce/training.py`
- Test: `tests/test_pce_training.py`

1. Write failing tests for training-only target standardization and known MAE/RMSE/R-squared values.
2. Implement `TargetScaler`, `regression_metrics`, one-epoch training, and evaluation functions.
3. Ensure batches move both DGL graphs and labels to the selected device.
4. Run the training utility tests; expect all to pass.

### Task 4: Graph cache and datasets

**Files:**
- Extend: `pce/data.py`
- Extend: `tests/test_pce_data.py`

1. Add a failing test that edge aggregation yields a 113-dimensional node feature exactly once.
2. Implement graph construction around `path_complex_mol`, per-SMILES cache reuse, failed-graph reporting, `PairGraphDataset`, and a DGL batch collator.
3. Save caches with metadata identifying encoder settings.
4. Run data tests; expect all to pass.

### Task 5: CLI and configuration

**Files:**
- Create: `main_pce.py`
- Create: `config/pce.yaml`

1. Parse `--config` without overriding the user-supplied path.
2. Load/prepare the pair table, create or load graph cache, split data, standardize targets from training data, train with validation checkpoint selection, evaluate the test split, and save predictions plus checkpoint.
3. Print an audit summary and final MAE, RMSE, and R-squared in PCE units.
4. Use the bundled `data/raw/Active_Database.csv` by default and fail clearly when it is missing.

### Task 6: End-to-end verification

**Files:**
- Create: `tests/fixtures/pce_smoke.csv`
- Create: `config/pce_smoke.yaml`

1. Run `python main_pce.py --config config/pce_smoke.yaml`; expect graph cache creation, at least one completed epoch, prediction CSV creation, checkpoint creation, and exit code 0.
2. Run `python -m unittest discover -s tests -v`; expect zero failures.
3. Run `python -m pip check`; expect no broken requirements.
4. Run `python -m py_compile main_pce.py pce/*.py model/pce_ka_gnn.py`; expect exit code 0.
5. Run a short real-data configuration and record counts and metrics.
