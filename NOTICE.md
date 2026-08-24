# Attribution and modification notice

This repository is adapted from [LongLee220/KA-GNN](https://github.com/LongLee220/KA-GNN), released under the MIT License. The upstream copyright and license text are preserved in `LICENSE`.

The following files contain or directly reuse upstream implementation ideas:

- `model/ka_gnn.py`: copied from the upstream KA-GNN implementation and kept as the source of the Fourier-KAN layers;
- `utils/graph_path.py`: copied from the upstream molecular graph construction implementation;
- `model/pce_ka_gnn.py`: an added dual-encoder regression model built using the upstream Fourier-KAN layers.

Major modifications and additions in this repository are:

- replacement of a single-molecule classification workflow with ordered donor-acceptor pair regression;
- canonicalization and median aggregation of repeated donor-acceptor observations;
- a shared two-branch molecular graph encoder and interaction fusion features;
- unconstrained continuous PCE output with train-only target standardization;
- MAE, RMSE and R² evaluation, finite-value validation, graph caching and exported predictions;
- self-contained project configuration, tests, documentation and bundled OPV-DB source data.

Upstream project:

- Repository: https://github.com/LongLee220/KA-GNN
- Article: Li, L., Zhang, Y., Wang, G. et al. *Kolmogorov–Arnold graph neural networks for molecular property prediction*. Nature Machine Intelligence 7, 1346–1354 (2025).
- DOI: https://doi.org/10.1038/s42256-025-01087-7
