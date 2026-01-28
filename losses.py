import random
import torch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class PairwiseDiffLoss(nn.Module):
    def __init__(self, loss_type="l1"):
        """
        :param loss_type: 'l1', 'l2', or 'kldiv'
        """
        super(PairwiseDiffLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == "l1":
            self.base_loss = nn.L1Loss()
        elif loss_type == "l2":
            self.base_loss = nn.MSELoss()
        elif loss_type == "kldiv":
            self.base_loss = nn.KLDivLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predicts, targets):
        """
        :param predicts: Tensor of shape (B,) or (B, 1)
        :param targets: Tensor of shape (B,) or (B, 1)
        :return: Pairwise difference loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if predicts.ndim == 2 and predicts.shape[1] == 1:
            predicts = predicts.squeeze(1)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if predicts.ndim != 1 or targets.ndim != 1:
            raise ValueError("Both predicts and targets must be 1D tensors.")

        B = predicts.size(0)
        idx = list(range(B))
        random.shuffle(idx)
        shuffled_preds = predicts[idx]
        shuffled_targs = targets[idx]

        diff_preds = predicts - shuffled_preds
        diff_targs = targets - shuffled_targs

        return self.base_loss(diff_preds, diff_targs)


class ACLoss(nn.Module):
    def __init__(self, loss_type="l1", reduction="mean"):
        """
        Architecture Consistency Loss
        :param loss_type: 'l1' or 'l2'
        :param reduction: 'mean' or 'sum'
        """
        super(ACLoss, self).__init__()
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, predictions):
        """
        :param predictions: Tensor of shape [2B, 1] or [2B]
                            where front B are source, back B are augmented
        :return: scalar loss
        """
        N = predictions.shape[0]
        B = N // 2  # 自动 floor 向下取整

        # 仅使用前 B 和后 B，丢弃中间多出的一个（若存在）
        source = predictions[:B]
        augmented = predictions[-B:]

        # Ensure shape is [B]
        source = (
            source.squeeze(-1) if source.ndim == 2 and source.shape[1] == 1 else source
        )
        augmented = (
            augmented.squeeze(-1)
            if augmented.ndim == 2 and augmented.shape[1] == 1
            else augmented
        )

        return self.criterion(source, augmented)


class SoftRankLoss(torch.nn.Module):
    def __init__(self, config):
        super(SoftRankLoss, self).__init__()
        self.config = config

    # Kendall
    def diffkendall_tau(self, x, y, alpha=10.0, jitter=0.0):
        x = x.view(-1)
        y = y.view(-1)
        if jitter > 0:
            x = x + jitter * torch.randn_like(x)
            y = y + jitter * torch.randn_like(y)
        n = x.numel()

        dx = x.unsqueeze(0) - x.unsqueeze(1)  # [n,n]
        dy = y.unsqueeze(0) - y.unsqueeze(1)

        sx = torch.tanh(alpha * dx)
        sy = torch.tanh(alpha * dy)

        mask = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
        )
        concord = (sx * sy)[mask].sum()

        num_pairs = torch.tensor(n * (n - 1) / 2, device=x.device, dtype=concord.dtype)
        return concord / num_pairs

    def diffkendall_loss(self, pred, target, alpha=10.0, jitter=0.0):
        return 1.0 - self.diffkendall_tau(pred, target, alpha, jitter=jitter)

    # Spearman
    def soft_rank(self, v, tau=1.6):
        v = v.view(-1, 1)
        P = torch.sigmoid((v.T - v) / tau)  # [n,n]
        r = 1.0 + P.sum(dim=1)  # 近似秩
        return (r - r.mean()) / (r.std() + 1e-8)

    def spearman_loss(self, a, b, tau=1.0):
        ra, rb = self.soft_rank(a, tau), self.soft_rank(b, tau)
        rho = (ra * rb).mean()
        return 1.0 - rho

    def forward(self, preds, labels):
        """
        :param preds: Tensor of shape (B,) or (B, 1)
        :param labels: Tensor of shape (B,) or (B, 1)
        :return: Rank loss
        """
        # 自动 squeeze 支持 [B, 1] 输入
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        if preds.ndim != 1 or labels.ndim != 1:
            raise ValueError("Both preds and labels must be 1D tensors.")

        loss_spearman = self.spearman_loss(preds, labels, tau=1.6)
        loss_kendall = self.diffkendall_loss(preds, labels, alpha=10, jitter=0)

        return loss_spearman, loss_kendall

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


# --------- 你已有的三个类，原样放进来即可 ----------
# PairwiseDiffLoss, ACLoss, SoftRankLoss
# （这里假设你已经定义了它们）


@dataclass
class RankLossWeights:
    spearman: float = 1.0
    kendall: float = 1.0
    sr: float = 0.5
    ac: float = 0.1


def _reshape_like(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return pred.reshape(label.shape)


class RankLossPack(nn.Module):
    """
    把四个“排序相关”loss封装到一个类里：
      - SoftRankLoss: 返回 (spearman, kendall)
      - SR: 用 PairwiseDiffLoss 作为 sr_loss（按你 compute_loss 里的写法乘 0.5）
      - AC: ACLoss（按你 compute_loss 里的写法乘 0.1）

    你只需要调用：rank_loss = loss_pack(pred, label)
    """

    def __init__(
        self,
        config,
        *,
        weights: Optional[RankLossWeights] = None,
        # SR 用 PairwiseDiffLoss，默认 l1，和你原逻辑一致你也可以改
        sr_loss_type: str = "l1",
        # AC 默认 l1/mean
        ac_loss_type: str = "l1",
        ac_reduction: str = "mean",
        # SoftRankLoss 超参如果你想暴露也可以继续加
    ):
        super().__init__()
        self.config = config
        self.weights = weights or RankLossWeights()

        self.soft_rank = SoftRankLoss(config)
        self.sr_loss = PairwiseDiffLoss(loss_type=sr_loss_type)
        self.ac_loss = ACLoss(loss_type=ac_loss_type, reduction=ac_reduction)

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        pred = _reshape_like(pred, label)

        # 保持你原逻辑：nnlqp 不加 rank/sr/ac
        if getattr(self.config, "dataset", None) == "nnlqp":
            zero = pred.new_tensor(0.0)
            return (zero, {}) if return_dict else zero

        # SoftRankLoss 返回 spearman & kendall
        loss_sp, loss_kd = self.soft_rank(pred, label)

        # SR（用 PairwiseDiffLoss 作为 sr_loss）
        loss_sr = self.sr_loss(pred, label)

        # AC（只看 pred）
        loss_ac = self.ac_loss(pred)

        total = (
            self.weights.spearman * loss_sp
            + self.weights.kendall * loss_kd
            + self.weights.sr * loss_sr
            + self.weights.ac * loss_ac
        )

        if not return_dict:
            return total

        detail = {
            "spearman": loss_sp,
            "kendall": loss_kd,
            "sr": loss_sr,
            "ac": loss_ac,
            "rank_total": total,
        }
        return total, detail