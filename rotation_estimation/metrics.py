from typing import Dict, List

import numpy as np
import torch

from .constants import METRICS


def _arccos_trace(R: torch.Tensor, unit: str = "radians") -> torch.Tensor:
    if unit == "degrees":
        scale = 180 / np.pi
    else:
        scale = 1
    return scale * torch.arccos((torch.trace(R) - 1) / 2)


# http://www.boris-belousov.net/2016/12/01/quat-dist/
def so3_distance(P: torch.Tensor, Q: torch.Tensor, unit: str = "radians") -> torch.Tensor:
    R = torch.matmul(P, Q.transpose(2, 1))
    if len(P.shape) == 3:
        num_matrices = P.shape[0]
        distances = []
        for k in range(num_matrices):
            distances.append(_arccos_trace(R[k], unit=unit))
        return torch.Tensor(distances)
    return _arccos_trace(R, unit=unit)


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
