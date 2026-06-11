import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from allshowers import ode_solvers
from allshowers.transformer import Transformer, compute_mask

__all__ = ["TimeSchedule", "CNF"]

_SCHEDULES = ("uniform", "lognorm", "pow", "mode")


class TimeSchedule:
    """Samples flow-matching time steps t ∈ [0, 1] from a chosen distribution.

    Parameters
    ----------
    schedule : str
        One of ``"uniform"`` (default), ``"lognorm"``, ``"pow"``, ``"mode"``.
    lognorm_mu : float
        Mean of the underlying normal for logit-normal sampling (default 0).
    lognorm_sigma : float
        Std of the underlying normal for logit-normal sampling (default 1).
    power_alpha : float
        Exponent for power-law sampling; higher → more mass near 0 (default 3).
    mode_s : float
        Shape parameter for the mode schedule (default -0.54).
    """

    def __init__(
        self,
        schedule: str = "uniform",
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        power_alpha: float = 3.0,
        mode_s: float = -0.54,
    ):
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"Unknown time schedule '{schedule}'. "
                f"Choose one of {_SCHEDULES}."
            )
        self.schedule = schedule
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.power_alpha = power_alpha
        self.mode_s = mode_s

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a (batch_size,) float tensor of time steps on *device*."""
        match self.schedule:
            case "uniform":
                t = np.random.rand(batch_size)
            case "pow":
                t = np.random.power(self.power_alpha, size=batch_size)
            case "lognorm":
                z = norm.rvs(loc=self.lognorm_mu, scale=self.lognorm_sigma, size=batch_size)
                t = 1.0 / (1.0 + np.exp(-z))
            case "mode":
                u = np.random.rand(batch_size)
                s = self.mode_s
                t = 1.0 - u - s * (np.cos(np.pi / 2 * u) ** 2 - 1.0 + u)
        return torch.from_numpy(t.astype(np.float32)).to(device)

    def __repr__(self) -> str:
        match self.schedule:
            case "uniform":
                detail = ""
            case "lognorm":
                detail = f", mu={self.lognorm_mu}, sigma={self.lognorm_sigma}"
            case "pow":
                detail = f", alpha={self.power_alpha}"
            case "mode":
                detail = f", s={self.mode_s}"
        return f"TimeSchedule(schedule={self.schedule!r}{detail})"


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Transformer,
        frequencies: int = 3,
        solver: str = "heun",
        time_schedule: str = "uniform",
        time_schedule_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Buffer(
            (torch.arange(1, frequencies + 1) * torch.pi).reshape(1, -1)
        )
        self.num_layer_cond = network.num_layer_cond
        self.network = network
        self.set_solver(solver)
        self.time_schedule = TimeSchedule(
            schedule=time_schedule,
            **(time_schedule_kwargs or {}),
        )

    def set_solver(self, solver: str) -> None:
        if solver not in ode_solvers.integrators:
            raise ValueError(
                f"Solver '{solver}' is not registered. "
                f"Available solvers: {list(ode_solvers.integrators.keys())}"
            )
        self.solver = ode_solvers.integrators[solver]

    def forward(self, t: Tensor, x: Tensor, **kwargs) -> Tensor:
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(x.shape[0], -1)

        return self.network(t, x, **kwargs)

    def __calculate_block_mask(self, kwargs: dict[str, Tensor | BlockMask]) -> None:
        if "layer" not in kwargs or "mask" not in kwargs:
            raise ValueError(
                "The 'layer' and 'mask' arguments must be provided in kwargs."
                "This implementation of a CNF only supports our transformers"
                "implementation as the network."
            )
        if not isinstance(kwargs["layer"], Tensor) or not isinstance(
            kwargs["mask"], Tensor
        ):
            raise TypeError(
                "Both 'layer' and 'mask' must be of type Tensor. "
                f"Got {type(kwargs['layer'])} and {type(kwargs['mask'])}."
            )
        mask = kwargs["mask"]
        del kwargs["mask"]
        kwargs["block_mask"] = compute_mask(
            padding_mask=mask,
            layer=kwargs["layer"],
            num_layer_cond=self.num_layer_cond,
        )

    def encode(self, x: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        self.__calculate_block_mask(kwargs)
        return self.solver(self, x, 0.0, 1.0, num_timesteps, **kwargs)

    def decode(self, z: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        self.__calculate_block_mask(kwargs)
        return self.solver(self, z, 1.0, 0.0, num_timesteps, **kwargs)

    def loss(self, x: Tensor, noise: Tensor | None, **kwargs) -> Tensor:
        self.__calculate_block_mask(kwargs)
        t = self.time_schedule.sample(x.shape[0], device=x.device).to(x.dtype)
        t = t.reshape([x.shape[0]] + [1] * (x.dim() - 1))
        z = noise if noise is not None else torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self(t.reshape(-1, 1), y, **kwargs) - u).square()

    def sample(
        self, shape: tuple[int, ...], num_timesteps: int = 200, **kwargs
    ) -> Tensor:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.decode(z, num_timesteps, **kwargs)

    def __repr__(self) -> str:
        network = self.network.__repr__().replace("\n", "\n  ")
        return f"""\
{self.__class__.__name__}(
  (network): {network}
  frequencies/pi={(self.frequencies[0] / torch.pi).tolist()}
  time_schedule={self.time_schedule!r}
)"""