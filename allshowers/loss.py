from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["MSELoss", "MSEAndDirectionLoss", "Set2SetLoss", "losses", "build_loss"]


class MSELoss(nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        loss_weights: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)
        self.loss_weights = nn.Buffer(
            loss_weights if loss_weights is not None else torch.ones(1),
            persistent=False,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ):
        if mask is not None:
            pred = pred * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)
        mse_loss = (
            self.loss_weights * F.mse_loss(pred, target, reduction="none")
        ).sum()
        if self.reduction == "sum":
            return mse_loss
        elif self.reduction == "mean":
            denominator = (
                np.prod(pred.shape) if mask is None else mask.sum() * pred.shape[-1]
            )
            return mse_loss / denominator


class MSEAndDirectionLoss(nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(
        self,
        cosine_sim_dim: int = 2,
        reduction: Literal["mean", "sum"] = "mean",
        loss_weights: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        assert cosine_sim_dim > 0, "cannot be batch dimension"
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.cosine_sim_dim = cosine_sim_dim
        self.reduction = reduction
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)
        self.loss_weights = nn.Buffer(
            loss_weights if loss_weights is not None else torch.ones(1),
            persistent=False,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ):
        # Adjust cosine similarity dimension for 2D tensors
        cosine_dim = (
            1 if pred.dim() == 2 and self.cosine_sim_dim == 2 else self.cosine_sim_dim
        )

        # Apply mask if provided
        if mask is not None:
            pred_masked = pred * mask.unsqueeze(-1)
            target_masked = target * mask.unsqueeze(-1)
        else:
            pred_masked = pred
            target_masked = target

        # Compute losses
        mse_loss = (
            self.loss_weights * F.mse_loss(pred_masked, target_masked, reduction="none")
        ).sum()
        direction_loss = 1.0 - F.cosine_similarity(
            pred_masked, target_masked, dim=cosine_dim
        )

        if mask is not None:
            direction_loss = direction_loss * mask
        direction_loss = direction_loss.sum()

        # Apply reduction
        if self.reduction == "sum":
            return mse_loss + direction_loss

        # Mean reduction
        mse_denominator = (
            np.prod(pred.shape) if mask is None else mask.sum() * pred.shape[-1]
        )
        cosine_denominator = np.prod(pred.shape[:-1]) if mask is None else mask.sum()

        return mse_loss / mse_denominator + direction_loss / cosine_denominator


class Set2SetLoss(nn.Module):
    """
    Bipartite-matching MSE loss for unordered point sets (Hungarian assignment
    on pt/eta/phi features at indices 0/1/2). Not wired into CNF.loss, which
    already has point-to-point correspondence between prediction and target;
    use directly for models that predict an unordered set without correspondence.
    """

    def __init__(self):
        super().__init__()
        self.regression_loss = nn.MSELoss(reduction="none")

    def forward(self, input, target, mask):
        bs = len(input)
        new_mask = mask.unsqueeze(1).expand(-1, target.size(1), -1).cpu()

        new_input = input.cpu().unsqueeze(1).expand(-1, target.size(1), -1, -1)
        new_target = target.cpu().unsqueeze(2).expand(-1, -1, input.size(1), -1)

        pdist = self.regression_loss(new_input, new_target)

        pdist_pt = pdist[..., 0]
        pdist_eta = pdist[..., 1]
        pdist_phi = pdist[..., 2]

        pdist = pdist.mean(-1)
        pdist = pdist * new_mask
        pdist_pt = pdist_pt * new_mask
        pdist_eta = pdist_eta * new_mask
        pdist_phi = pdist_phi * new_mask

        pdist = torch.nan_to_num(pdist, nan=0.0, posinf=0.0, neginf=0.0)

        pdist_ = pdist.detach().numpy()
        indices = np.array(
            [linear_sum_assignment(p) for p in pdist_]
        )  # indices shape (b,2,N)

        total_losses = torch.zeros((bs), device=pdist.device)
        pt_losses = torch.zeros((bs), device=pdist.device)
        eta_losses = torch.zeros((bs), device=pdist.device)
        phi_losses = torch.zeros((bs), device=pdist.device)
        for idx_i in range(bs):
            indices_i = indices.shape[2] * indices[idx_i, 0] + indices[idx_i, 1]
            matched_indices = torch.from_numpy(indices_i).to(device=pdist.device)

            total_losses[idx_i] = torch.gather(
                pdist[idx_i].flatten(0, 1), 0, matched_indices
            ).mean(0)
            pt_losses[idx_i] = torch.gather(
                pdist_pt[idx_i].flatten(0, 1), 0, matched_indices
            ).mean(0)
            eta_losses[idx_i] = torch.gather(
                pdist_eta[idx_i].flatten(0, 1), 0, matched_indices
            ).mean(0)
            phi_losses[idx_i] = torch.gather(
                pdist_phi[idx_i].flatten(0, 1), 0, matched_indices
            ).mean(0)

        return {
            "total_loss": total_losses.mean().to(input.device),
            "pt_loss": pt_losses.mean(),
            "eta_loss": eta_losses.mean(),
            "phi_loss": phi_losses.mean(),
        }


losses: dict[str, type[nn.Module]] = {
    "MSELoss": MSELoss,
    "MSEAndDirectionLoss": MSEAndDirectionLoss,
    "Set2SetLoss": Set2SetLoss,
}


def build_loss(config: dict[str, Any] | str) -> nn.Module:
    """Build a loss module from a config dict (with a 'name' key) or a bare name string."""
    if isinstance(config, str):
        config = {"name": config}
    config = dict(config)
    name = config.pop("name")
    if name not in losses:
        raise ValueError(f"Loss '{name}' is not registered. Available losses: {list(losses)}")
    return losses[name](**config)
