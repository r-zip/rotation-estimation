"""Basic neural network building blocks"""

from typing import List, Optional

import torch.nn as nn


def build_mlp(
    input_dimension: int,
    output_dimension: int,
    hidden_layer_sizes: List[int],
    layer_norm: bool = True,
    final_activation: Optional[nn.Module] = None,
    final_layer_norm: bool = False,
) -> nn.Module:
    input_sizes = [input_dimension, *hidden_layer_sizes]
    output_sizes = [*hidden_layer_sizes, output_dimension]

    # reference for layer norm/layer/activation fct order:
    # https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1
    layers = []
    for k, (input_size, output_size) in enumerate(zip(input_sizes, output_sizes)):
        layers.append(nn.Linear(input_size, output_size))
        if k < min(len(input_sizes), len(output_sizes)) - 1:
            if layer_norm:
                layers.append(nn.LayerNorm(output_size))
            layers.append(nn.ReLU())

    if final_layer_norm:
        layers.append(nn.LayerNorm(output_dimension))

    if final_activation:
        layers.append(final_activation)

    return nn.Sequential(*layers)
