import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch


FlowType = Literal["icfm", "rcfm", "vp"]


def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t
    return t.reshape(-1, *([1] * (x.ndim - 1)))


@dataclass(frozen=True)
class PathCoefficients:
    alpha: torch.Tensor
    sigma: torch.Tensor
    alpha_dot: torch.Tensor
    sigma_dot: torch.Tensor


class ProbabilityPath(ABC):
    """Path x_t = alpha(t) x0 + sigma(t) x1."""

    @property
    @abstractmethod
    def reverse_time(self) -> bool:
        pass

    @property
    def time_range(self) -> tuple[float, float]:
        return (1.0, 0.0) if self.reverse_time else (0.0, 1.0)

    @abstractmethod
    def coefficients(self, t: torch.Tensor, x: torch.Tensor) -> PathCoefficients:
        pass

    def interpolate(
        self,
        source: torch.Tensor,
        data: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = self.coefficients(t, data)
        return coeffs.alpha * source + coeffs.sigma * data

    def velocity(
        self,
        source: torch.Tensor,
        data: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = self.coefficients(t, data)
        return coeffs.alpha_dot * source + coeffs.sigma_dot * data

    def model_time(self, solver_time: torch.Tensor) -> torch.Tensor:
        """Map DPM's source-to-data clock onto the model path time."""
        if self.reverse_time:
            return solver_time
        return 1 - solver_time

    def sampling_time_range(
        self,
        prediction_type: str,
        epsilon: float,
    ) -> tuple[float, float]:
        start, end = self.time_range
        direction = 1 if end > start else -1

        start_t = torch.tensor(start)
        end_t = torch.tensor(end)
        if prediction_type == "noise":
            sigma_start = self.coefficients(start_t, start_t).sigma
            if sigma_start.abs() < epsilon:
                start += direction * epsilon
        elif prediction_type == "data":
            alpha_end = self.coefficients(end_t, end_t).alpha
            if alpha_end.abs() < epsilon:
                end -= direction * epsilon
        return start, end


@dataclass(frozen=True)
class LinearPath(ProbabilityPath):
    reverse: bool = False

    @property
    def reverse_time(self) -> bool:
        return self.reverse

    def coefficients(self, t: torch.Tensor, x: torch.Tensor) -> PathCoefficients:
        t = pad_t_like_x(t, x)

        if self.reverse:
            alpha, sigma = t, 1 - t
            alpha_dot = torch.ones_like(t)
            sigma_dot = -alpha_dot
        else:
            alpha, sigma = 1 - t, t
            alpha_dot = -torch.ones_like(t)
            sigma_dot = torch.ones_like(t)

        return PathCoefficients(alpha, sigma, alpha_dot, sigma_dot)


@dataclass(frozen=True)
class TrigonometricPath(ProbabilityPath):
    @property
    def reverse_time(self) -> bool:
        return False

    def coefficients(self, t: torch.Tensor, x: torch.Tensor) -> PathCoefficients:
        t = pad_t_like_x(t, x)
        angle = math.pi * t / 2
        alpha, sigma = torch.cos(angle), torch.sin(angle)
        alpha_dot = -math.pi / 2 * sigma
        sigma_dot = math.pi / 2 * alpha
        return PathCoefficients(alpha, sigma, alpha_dot, sigma_dot)


def make_probability_path(flow_type: FlowType) -> ProbabilityPath:
    if flow_type == "icfm":
        return LinearPath()
    if flow_type == "rcfm":
        return LinearPath(reverse=True)
    if flow_type == "vp":
        return TrigonometricPath()
    raise ValueError(f"Unknown flow type: {flow_type}")
