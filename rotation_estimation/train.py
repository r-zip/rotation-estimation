from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .metrics import so3_distance


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    lr: float,
    epochs: int = 3,
    val_every: int = 100,
    model_path: Path = Path("./saved_model.pt"),
) -> Dict[str, Dict[str, List[float]]]:
    optimizer = Adam(model.parameters(), lr=lr)

    # this loop is based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
    history = {"train": {"mse": [], "so3": []}, "val": {"mse": [], "so3": []}}
    sum_steps = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if (sum_steps + 1) % val_every == 0:
                avg_train_loss = running_loss / val_every
                print(f"[step: {sum_steps + 1:5d}] train loss: {avg_train_loss:.3f}")
                running_loss = 0.0

                model.eval()
                with torch.no_grad():
                    avg_val_loss = 0
                    n_val = 0
                    for val_batch_idx, (val_input, val_label) in enumerate(val_data_loader):
                        val_output = model(val_input)
                        avg_val_loss += F.mse_loss(val_output, val_label).item()
                        n_val += 1
                        if val_batch_idx >= 1:
                            break

                    avg_val_loss /= n_val

                model.train()

                print(f"[step: {sum_steps + 1:5d}] val loss: {avg_val_loss:.3f}")

                # validation here
                history["train"]["mse"].append(avg_train_loss)
                history["val"]["mse"].append(avg_val_loss)

            sum_steps += 1

    torch.save(model, model_path)

    return history
