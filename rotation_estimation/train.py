from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .constants import DEFAULT_EPOCHS, DEFAULT_LR, METRICS, MODEL_PATH
from .metrics import avg_metrics, compute_metrics


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
    """
    This function's main loop used the PyTorch tutorial at
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network as a starting point.
    """

    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
        "val": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
    }
    sum_steps = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_metrics = []
        n_train = 0
        for step, (original, rotated, labels) in enumerate(train_data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            raw, projected = model(original, rotated)
            loss = loss_fn(raw, projected, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_metrics.append(compute_metrics(projected, labels))
            n_train += 1

            val_metrics = []
            if (sum_steps + 1) % val_every == 0:
                avg_train_loss = running_loss / n_train
                print(f"[step: {sum_steps + 1:5d}] train loss: {avg_train_loss:.3f}")
                n_train = 0
                running_loss = 0.0

                model.eval()
                with torch.no_grad():
                    avg_val_loss = 0
                    n_val = 0
                    for val_original, val_rotated, val_label in val_data_loader:
                        val_raw, val_projected = model(val_original, val_rotated)
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

                print(f"[step: {sum_steps + 1:5d}] train so3: {avg_train_metrics['so3']:.3f}")
                print(f"[step: {sum_steps + 1:5d}] val so3: {avg_val_metrics['so3']:.3f}")

                for key in METRICS:
                    history["train"][key].append(avg_train_metrics[key])
                    history["val"][key].append(avg_val_metrics[key])

                # clear metrics
                train_metrics.clear()
                val_metrics.clear()

            sum_steps += 1

    return history, model
