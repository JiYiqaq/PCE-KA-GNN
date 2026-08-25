# GPU Production Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the bundled PCE KA-GNN workflow run reproducibly on the NVIDIA GPU without silent CPU fallback or repeated SMILES preparation.

**Architecture:** Use an explicit CUDA-only device resolver, a fingerprinted canonical-pair cache stored under `data/processed`, and runtime metadata in every summary. Keep the existing graph cache format and validate that it loads under the production CUDA DGL build before any formal run.

**Tech Stack:** Python 3.10, PyTorch 2.1.2+cu118, DGL 2.2.1+cu118, torchdata 0.7.1, RDKit, pandas, unittest.

---

### Task 1: Add fingerprinted canonical-pair caching

**Files:**
- Modify: `tests/test_main_pce.py`
- Modify: `main_pce.py`

**Steps:**
1. Add tests proving a valid cache skips raw CSV preparation and a changed source fingerprint invalidates the cache.
2. Run the focused tests and confirm they fail because automatic cache reuse is absent.
3. Implement SHA256-backed cache metadata and atomic cache writes.
4. Run the focused and full test suites.

### Task 2: Enforce and report the production CUDA device

**Files:**
- Modify: `tests/test_main_pce.py`
- Modify: `main_pce.py`
- Modify: `config/pce.yaml`
- Modify: `config/pce_quick.yaml`
- Modify: `config/pce_smoke.yaml`

**Steps:**
1. Add tests for explicit CUDA selection, fail-fast behavior, and runtime metadata.
2. Run the focused tests and confirm they fail under the current silent CPU fallback.
3. Implement the CUDA resolver and metadata report; set tracked configs to `device: cuda`.
4. Run the focused and full test suites.

### Task 3: Pin and document the verified GPU environment

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-gpu.txt`
- Modify: `README.md`
- Modify: `tests/test_project_layout.py`

**Steps:**
1. Add layout tests for the CUDA dependency pins and production configs.
2. Run the tests and confirm they fail with the CPU-oriented documentation and DGL pin.
3. Add the exact verified CUDA pins and concise installation/verification commands.
4. Run the full suite and `pip check` in the GPU environment.

### Task 4: Verify the real production path

**Files:**
- Runtime artifacts only under ignored `data/processed` and `outputs`.

**Steps:**
1. Verify PyTorch CUDA and a DGL graph on `cuda:0`.
2. Load the existing 131 MB graph cache under the GPU environment.
3. Seed and validate the canonical-pair cache from the already generated canonical pairs.
4. Run the smoke configuration on CUDA and confirm `summary.json` reports CUDA runtime metadata.
5. Run the formal configuration on CUDA, monitor memory/runtime, and verify output artifacts.
6. Run all tests, `pip check`, `git diff --check`, and inspect the final diff.
7. Commit, merge to `main`, push to GitHub, and remove the temporary worktree.
