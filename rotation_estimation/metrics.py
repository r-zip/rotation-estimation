import numpy as np
import torch


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
