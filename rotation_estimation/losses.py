import torch
from torch.nn.modules.loss import _Loss


class OrthogonalMSELoss(_Loss):
    def __init__(self, six_d: bool = False, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight
        self.six_d = six_d

    def forward(self, output: torch.Tensor, pred_matrix: torch.Tensor, true_matrix: torch.Tensor) -> torch.Tensor:
        if self.six_d:
            return torch.mean(
                torch.sum((output - true_matrix[:, :2, :]) ** 2, dim=(1, 2))
                + self.weight * torch.mean(torch.abs(output[:, 0, :] * output[:, 1, :]).sum(dim=1))
            )
        return torch.mean(
            torch.sum((pred_matrix - true_matrix) ** 2)
            + self.weight
            * (
                torch.mean(torch.abs(output[:, 0, :] * output[:, 1, :]).sum(dim=1))
                + torch.mean(torch.abs(output[:, 0, :] * output[:, 2, :]).sum(dim=1))
                + torch.mean(torch.abs(output[:, 1, :] * output[:, 2, :]).sum(dim=1))
            )
        )
