from typing import List, Optional

import pytest
import torch.nn as nn

from rotation_estimation.blocks import build_mlp


@pytest.mark.parametrize(
    "layer_sizes,layer_norm,final_layer_norm,final_activation",
    [
        ([256, 512, 128], True, False, nn.ReLU()),
        ([256, 512, 128], True, True, None),
        ([256, 512, 128], False, False, None),
        ([256, 512], False, False, None),
    ],
)
def test_mlp(layer_sizes: List[int], layer_norm: bool, final_layer_norm: bool, final_activation: Optional[nn.Module]):
    mlp = build_mlp(
        input_dimension=100,
        output_dimension=10,
        hidden_layer_sizes=layer_sizes,
        layer_norm=layer_norm,
        final_activation=final_activation,
        final_layer_norm=final_layer_norm,
    )
    expected_num_layers = len(layer_sizes) + 1
    expected_num_relus = expected_num_layers - 1 + (1 if final_activation else 0)
    expected_num_norms = (expected_num_layers - 1 if layer_norm else 0) + (1 if final_layer_norm else 0)

    assert sum([isinstance(m, nn.Linear) for m in list(mlp.modules())]) == expected_num_layers
    assert sum([isinstance(m, nn.ReLU) for m in list(mlp.modules())]) == expected_num_relus
    assert sum([isinstance(m, nn.LayerNorm) for m in list(mlp.modules())]) == expected_num_norms
