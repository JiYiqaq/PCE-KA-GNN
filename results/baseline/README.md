# Verified deterministic GPU baseline

These files preserve the reproducible CUDA run performed with seed 42 and `config/pce.yaml`. The verified runtime was PyTorch 2.1.2+cu118 and DGL 2.2.1+cu118 on an NVIDIA GeForce GTX 1650 Ti. Training stopped after epoch 57; the validation-MAE checkpoint was epoch 37.

Preprocessing produced 5,877 canonical donor-acceptor pairs from 38,849 rows. The upstream 3D graph builder and finite-feature validation left 470 usable pairs for the recorded 376/47/47 split. The test metrics are MAE 2.2126, RMSE 2.7700 and R2 0.2163.

The large generated graph cache is not included because it exceeds GitHub's 100 MB single-file limit. `best_model.pt` contains model weights, model configuration, the target scaler and the best epoch. The result metadata uses project-relative paths.
