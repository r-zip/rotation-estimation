import torch
from torch.nn.modules.loss import _Loss


class OrthogonalMSELoss(_Loss):
    def __init__(self, six_d: bool = False, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight
        self.six_d = six_d

    def forward(self, output: torch.Tensor, pred_matrix: torch.Tensor, true_matrix: torch.Tensor) -> torch.Tensor:
        if self.six_d:
            # we consulted Adam W's old implementation for the next two lines of this function's code
            regularizer = torch.abs(output[:, 0, :] * output[:, 1, :]).sum(dim=1)
            return torch.mean(torch.mean((output - true_matrix[:, :2, :]) ** 2, dim=(1, 2)) + self.weight * regularizer)

        regularizer = (
            torch.abs(output[:, 0, :] * output[:, 1, :]).sum(dim=1)
            + torch.abs(output[:, 0, :] * output[:, 2, :]).sum(dim=1)
            + torch.abs(output[:, 1, :] * output[:, 2, :]).sum(dim=1)
        ) / 3
        return torch.mean(torch.mean((pred_matrix - true_matrix) ** 2, dim=(1, 2)) + self.weight * regularizer)
