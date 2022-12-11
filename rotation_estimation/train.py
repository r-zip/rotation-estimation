from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .constants import DEFAULT_EPOCHS, DEFAULT_LR, MODEL_PATH
from .metrics import so3_distance

METRICS = ["so3", "euler"]


def compute_metrics(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        try:
            so3 = so3_distance(pred, truth).sum().item()
        except ValueError:
            so3 = np.nan
        return {"so3": so3, "euler": np.nan, "n": pred.shape[0]}


def avg_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    output_dict = {}
    for key in METRICS:
        output_dict[key] = sum([m * n for m, n in zip([m[key] for m in metrics], [m["n"] for m in metrics])]) / sum(
            [m["n"] for m in metrics]
        )
    return output_dict


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    val_every: int = 100,
    model_path: Optional[Path] = None,
    loss_fn: Callable = F.mse_loss,
) -> Dict[str, Dict[str, List[float]]]:
    optimizer = Adam(model.parameters(), lr=lr)

    # this loop is based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
    history = {
        "train": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
        "val": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
    }
    sum_steps = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_metrics = []
        for step, (inputs, labels) in enumerate(train_data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            raw, projected = model(inputs)
            loss = loss_fn(raw, projected, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_metrics.append(compute_metrics(projected, labels))

            val_metrics = []
            if (sum_steps + 1) % val_every == 0:
                avg_train_loss = running_loss / val_every
                print(f"[step: {sum_steps + 1:5d}] train loss: {avg_train_loss:.3f}")
                running_loss = 0.0

                model.eval()
                with torch.no_grad():
                    avg_val_loss = 0
                    n_val = 0
                    for val_input, val_label in val_data_loader:
                        val_raw, val_projected = model(val_input)
                        avg_val_loss += loss_fn(val_raw, val_projected, val_label).item()
                        n_val += 1
                        val_metrics.append(compute_metrics(val_projected, val_label))

                    avg_val_loss /= n_val

                model.train()

                print(f"[step: {sum_steps + 1:5d}] val loss: {avg_val_loss:.3f}")

                # validation here
                history["train"]["mse"].append(avg_train_loss)
                history["val"]["mse"].append(avg_val_loss)

                # average metrics
                avg_train_metrics = avg_metrics(train_metrics)
                avg_val_metrics = avg_metrics(val_metrics)

                for key in METRICS:
                    history["train"][key].append(avg_train_metrics[key])
                    history["val"][key].append(avg_val_metrics[key])

                # clear metrics
                train_metrics.clear()
                val_metrics.clear()

            sum_steps += 1

    if model_path is None:
        model_path = MODEL_PATH / "model.pt"

    torch.save(model, model_path)

    return history
