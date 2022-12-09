import pytorch3d.transforms.rotation_conversions  # # 6D AUX functions
import torch
from torch.nn.modules.loss import _Loss


class OrthogonalMSELoss(_Loss):
    def __init__(self, weight: float = 0.01) -> None:
        super(OrthogonalMSELoss, self).__init__()
        self.weight = weight

    def forward(self, x, y) -> torch.Tensor:
        return torch.mean(
            torch.sum((x - y[:, :2, :]) ** 2, dim=(1, 2))
            + self.weight * torch.flatten(torch.matmul(x[:, 0, :][:, None, :], x[:, 1, :][:, :, None]))
        )
