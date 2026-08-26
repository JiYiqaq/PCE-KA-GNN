# Multimodal PCE Regression Design

## Objective

Replace the pair-median, molecule-only workflow with a production workflow that predicts the PCE of each device record from donor and acceptor molecular graphs plus independently available material, formulation, device, and processing variables. Preserve the verified molecule-only baseline under `results/baseline/`, exclude target leakage, retain every valid device record that can be represented, and run formal training on the verified CUDA stack.

## Data contract

Each retained row is one device observation. Required fields are `donor_smiles`, `acceptor_smiles`, and finite `pce`. Donor and acceptor SMILES are canonicalized once and cached with a source-file fingerprint. Repeated donor-acceptor pairs are not collapsed because their conditions and targets may differ. The active predictors are:

- Numeric: `homo_d`, `lumo_d`, `homo_a`, `lumo_a`, `active_layer_thickness`, `annealing_temp`, parsed donor-acceptor log ratio, and parsed additive percentage.
- Categorical: `device_type`, `etl_canonical`, `htl_canonical`, `solvent_canonical`, and `additive_canonical`.
- Molecular: canonical donor and acceptor graphs.

`voc`, `jsc`, `ff`, recomputed/average/best PCE fields, identifiers, DOI values, and raw material names are excluded from the feature tensors. Missing numeric values receive a training-set median plus an explicit availability mask. Missing categorical values and categories unseen in training use separate tokens. Preprocessing statistics and vocabularies are fitted on training rows only and serialized with the checkpoint.

## Molecular graph representation

The old builder made 3D embedding a prerequisite even though the active PCE model did not consume coordinates directly. The replacement constructs a deterministic bidirectional topology graph from every RDKit-valid canonical SMILES, without requiring a conformer. It preserves the author's 92-dimensional CGCNN element representation and 21-dimensional bond representation; mean incident-bond features are concatenated to produce the existing 113-dimensional node input.

The cache records its builder version, encoders, successful graphs, and exact failure reasons. A row is removed only when either SMILES is invalid or topology feature construction genuinely fails. No geometry failure can remove a row. Three-dimensional learning is intentionally not mixed into this change: adding a scientifically validated geometry branch requires a separate conformer protocol and ablation rather than zero-filled pseudo-geometry.

## Model and execution

The shared KA-GNN encoder produces one 64-dimensional vector for each molecule. Pair features remain `[donor, acceptor, |donor-acceptor|, donor*acceptor]`. A context encoder embeds categorical variables, combines them with robust-scaled numeric values and missingness masks, and maps the result through a Fourier-KAN layer. The pair and context vectors feed a KAN regression head with no sigmoid.

All rows sharing the same ordered canonical donor-acceptor pair are assigned to the same train, validation, or test split. This prevents direct pair leakage while retaining condition-level observations. Context preprocessing is fitted after splitting. The primary checkpoint minimizes validation MAE; final metrics are reported once on the held-out test set.

Formal training requires `cuda:0`, deterministic CUDA algorithms, and the verified PyTorch/DGL builds. If the topology graph collection fits within a conservative fraction of free GPU memory, all unique graphs are preloaded to CUDA once; otherwise the run fails with a concrete memory report instead of silently taking a slower execution path. RDKit topology construction remains CPU-bound because this code path has no GPU implementation, and its result is cached.

## Validation and acceptance criteria

- All valid input rows are preserved until graph validation; repeated pairs remain separate records.
- At least 99% of RDKit-valid unique molecules build topology graphs, or the remaining failures are individually reported.
- Train/validation/test rows are complete and donor-acceptor-pair disjoint.
- No target-derived field enters numeric or categorical feature tensors.
- Missing and unknown values transform to finite, reproducible tensors using training-only statistics.
- Unit tests exercise the actual CUDA build for model forward/backward behavior.
- A production-stack smoke run and the formal command complete on GTX 1650 Ti with runtime metadata and cache audits.
- The formal multimodal result is compared with a material-only ablation using the identical rows and split.

