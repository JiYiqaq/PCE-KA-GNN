# PCE Regression Design

## Goal

Provide an independent regression workflow that predicts a continuous PCE value from an ordered donor-SMILES/acceptor-SMILES pair. Reuse only the attributed Fourier-KAN and molecular graph construction components required from KA-GNN.

## Data choice

The source table contains repeated measurements for many donor-acceptor pairs. A model that only sees the two SMILES strings cannot distinguish processing conditions for otherwise identical inputs. The first baseline therefore canonicalizes both SMILES strings and aggregates each ordered pair to its median PCE. The retained `replicate_count` and within-pair PCE standard deviation document label uncertainty for later analysis.

Alternative approaches considered:

1. Train on every device row and group-split by pair. This retains more rows but presents conflicting targets for identical inputs.
2. Add device/process descriptors to the model. This is scientifically useful but is a separate multimodal experiment and exceeds the present two-SMILES objective.

The pair-median baseline is selected because it gives each input one defensible target and prevents duplicate-pair leakage.

## Model

Each molecule is converted with the repository's existing `path_complex_mol` graph builder. Edge features are averaged into each destination node exactly once, producing the same 113-dimensional node input used by the original program. A shared KA-GNN graph encoder, built from the author's Fourier KAN layers, processes donor and acceptor graphs independently. Shared weights reduce overfitting while the ordered concatenation keeps donor and acceptor roles distinct.

The pair representation concatenates donor embedding, acceptor embedding, absolute difference, and elementwise product. A Fourier-KAN regression head produces one unconstrained scalar. It deliberately omits sigmoid because PCE is continuous rather than a probability.

## Training and evaluation

Unique pairs are shuffled with a fixed seed and split 80/10/10. Pair keys are checked to be disjoint. Targets are standardized using training-set mean and standard deviation only. The model optimizes MSE in standardized space; predictions are converted back to PCE units for MAE, RMSE, and R-squared reporting.

Validation MAE controls checkpoint selection. The saved checkpoint includes model state, target scaler, configuration, data summary, and final metrics. Unique molecular graphs are cached so repeated molecules are not rebuilt across pairs or runs.

## Error handling and verification

Rows with missing/non-numeric PCE, missing SMILES, invalid RDKit SMILES, or failed graph construction are counted and excluded with a summary. Too-small datasets and empty splits fail with actionable errors. Unit tests cover aggregation, split disjointness, model output shape, target scaling, and metrics. An end-to-end smoke run on a small synthetic CSV verifies graph creation, training, checkpoint writing, and metrics before the real dataset is used.

The public project bundles its source dataset and uses only project-relative paths. Large generated graph caches stay local and can always be rebuilt from the bundled CSV.
