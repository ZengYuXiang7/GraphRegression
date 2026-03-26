import torch
import torch.nn as nn
from torch import Tensor


class NARLoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_rank: float = 0.2,
        lambda_consist: float = 1.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank
        self.lambda_consist = lambda_consist
        self.loss_mse = nn.MSELoss()
        self.loss_rank = nn.L1Loss()
        self.loss_consist = nn.L1Loss()

    def _rank_loss(self, predict: Tensor, target: Tensor) -> Tensor:
        index = torch.randperm(predict.shape[0], device=predict.device)
        v1 = predict - predict[index]
        v2 = target - target[index]
        return self.loss_rank(v1, v2)

    def forward(self, predict: Tensor, target: Tensor):
        loss_mse = self.loss_mse(predict, target) * self.lambda_mse
        loss_rank = self._rank_loss(predict, target) * self.lambda_rank

        loss_consist = 0
        if self.lambda_consist > 0:
            source_pred, aug_pred = predict.chunk(2, 0)
            loss_consist = (
                self.loss_consist(source_pred, aug_pred) * self.lambda_consist
            )
        loss = loss_mse + loss_rank + loss_consist
        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_rank": loss_rank,
            "loss_consist": loss_consist,
        }
