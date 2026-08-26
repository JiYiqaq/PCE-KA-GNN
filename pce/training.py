from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite values detected in {name}")


def _predict_batch(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    donor_graph: object,
    acceptor_graph: object,
    device: torch.device,
) -> torch.Tensor:
    has_numeric = "numeric_context" in batch
    has_categorical = "categorical_context" in batch
    if has_numeric != has_categorical:
        raise ValueError("numeric and categorical context must be supplied together")
    if not has_numeric:
        return model(donor_graph, acceptor_graph)
    numeric_context = batch["numeric_context"].to(device=device, dtype=torch.float32)
    categorical_context = batch["categorical_context"].to(device=device, dtype=torch.long)
    _require_finite(numeric_context, "numeric context")
    return model(donor_graph, acceptor_graph, numeric_context, categorical_context)


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: torch.Tensor | Iterable[float]) -> "TargetScaler":
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.numel() == 0:
            raise ValueError("cannot fit target scaler on an empty training set")
        mean = float(tensor.mean())
        std = float(tensor.std(unbiased=False))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(
    targets: Iterable[float],
    predictions: Iterable[float],
) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=float)
    y_pred = np.asarray(list(predictions), dtype=float)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and have matching shapes")

    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(residual)) / total) if y_true.size >= 2 and total > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        donor_graph = batch["donor_graph"].to(device)
        acceptor_graph = batch["acceptor_graph"].to(device)
        target = batch["target"].to(device=device, dtype=torch.float32)
        standardized_target = scaler.transform(target)
        _require_finite(target, "training targets")
        _require_finite(standardized_target, "standardized training targets")

        optimizer.zero_grad()
        prediction = _predict_batch(model, batch, donor_graph, acceptor_graph, device)
        _require_finite(prediction, "training predictions")
        loss = F.mse_loss(prediction, standardized_target)
        _require_finite(loss, "training loss")
        loss.backward()
        optimizer.step()

        batch_size = int(target.numel())
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("training loader produced no examples")
    return total_loss / total_examples


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, object]],
    scaler: TargetScaler,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[float] = []
    predictions: list[float] = []

    with torch.no_grad():
        for batch in loader:
            donor_graph = batch["donor_graph"].to(device)
            acceptor_graph = batch["acceptor_graph"].to(device)
            target = batch["target"].to(device=device, dtype=torch.float32)
            standardized_target = scaler.transform(target)
            _require_finite(target, "evaluation targets")
            _require_finite(standardized_target, "standardized evaluation targets")
            standardized_prediction = _predict_batch(
                model, batch, donor_graph, acceptor_graph, device
            )
            _require_finite(standardized_prediction, "evaluation predictions")
            loss = F.mse_loss(standardized_prediction, standardized_target)
            _require_finite(loss, "evaluation loss")
            prediction = scaler.inverse(standardized_prediction)
            _require_finite(prediction, "evaluation predictions in PCE units")

            batch_size = int(target.numel())
            total_loss += float(loss) * batch_size
            total_examples += batch_size
            targets.extend(target.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())

    if total_examples == 0:
        raise ValueError("evaluation loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "targets": targets,
        "predictions": predictions,
        "metrics": regression_metrics(targets, predictions),
    }
