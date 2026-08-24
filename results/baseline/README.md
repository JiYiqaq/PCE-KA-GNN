# Verified quick baseline

These files preserve the successful two-epoch CPU run performed with seed 42 and the compact model configuration in `config/pce_quick.yaml`.

The first full preprocessing pass produced 5,877 canonical unique donor-acceptor pairs from 38,849 rows. The upstream 3D graph builder yielded 495 graph-usable pairs. Finite-feature validation removed another 25 pairs, leaving 470 pairs for the recorded 376/47/47 split.

The large generated graph cache is not included. `best_model.pt` contains the model weights, model configuration, target scaler and best epoch; embedded paths were changed to project-relative paths without changing any tensors.
