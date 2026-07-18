import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from allshowers import ode_solvers
from allshowers.transformer import Transformer, compute_mask

__all__ = ["CNF"]


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Transformer,
        frequencies: int = 3,
        solver: str = "heun",
        # --- Internal Guidance (IG) -----------------------------------------
        # Weight lambda of the auxiliary intermediate-layer supervision loss
        # added on top of the final-layer denoising loss during training (see
        # allshowers.transformer.Transformer's intermediate_layer_idx, which
        # must be set >= 0 for this to have any effect -- otherwise the
        # network never produces an intermediate output and this is ignored).
        # 0.0 (default) disables the auxiliary loss entirely.
        ig_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Buffer(
            (torch.arange(1, frequencies + 1) * torch.pi).reshape(1, -1)
        )
        self.num_layer_cond = network.num_layer_cond
        self.network = network
        self.set_solver(solver)
        self.ig_loss_weight = ig_loss_weight

        # --- IG sampling-time state -----------------------------------------
        # Off by default (ig_scale=1.0 is a no-op: forward() then always
        # returns the final head's output, identical to pre-IG behaviour).
        # Enabled via enable_internal_guidance(); disabled again via
        # disable_internal_guidance(). Deliberately NOT touched by loss(),
        # so a generator process that turns IG on for sampling can never
        # accidentally leak it into a gradient computation, and vice versa.
        self.ig_scale = 1.0
        self.ig_t_min = 0.0
        self.ig_t_max = 1.0

    def set_solver(self, solver: str) -> None:
        if solver not in ode_solvers.integrators:
            raise ValueError(
                f"Solver '{solver}' is not registered. "
                f"Available solvers: {list(ode_solvers.integrators.keys())}"
            )
        self.solver = ode_solvers.integrators[solver]

    def enable_internal_guidance(
        self,
        scale: float = 1.4,
        t_min: float = 0.3,
        t_max: float = 1.0,
    ) -> None:
        """Turn on Internal Guidance extrapolation for subsequent forward()
        calls (and therefore for encode/decode/sample, and for every
        external solver in dpm.py / generator.py's PNDMSolver, since they
        all call this module directly).

        D_w(x; c) = D_i(x; c) + w * (D_f(x; c) - D_i(x; c))

        where D_i is the intermediate ("weak") head's output and D_f is the
        final head's output (IG paper, Eq. 5). Guidance is only applied
        while t lies in (t_min, t_max) (the "guidance interval", Eq. 6);
        outside that range the plain final-head output is used.

        Requires the network to have been built with intermediate_layer_idx
        >= 0 (allshowers.transformer.Transformer) -- otherwise the
        intermediate head doesn't exist and this call has no effect at
        sampling time (forward() silently falls back to the final head).

        Note on t_min/t_max: in this codebase's rectified-flow convention,
        t=0 is pure data and t=1 is pure noise. The IG paper found guidance
        should be skipped near the *low-noise* end (t close to data), so the
        default t_min=0.3 leaves the first part of denoising unguided.
        """
        self.ig_scale = scale
        self.ig_t_min = t_min
        self.ig_t_max = t_max

    def disable_internal_guidance(self) -> None:
        """Restore forward() to always return the final head's output."""
        self.ig_scale = 1.0
        self.ig_t_min = 0.0
        self.ig_t_max = 1.0

    def _embed_time(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        return t.expand(x.shape[0], -1)

    def _forward_raw(self, t: Tensor, x: Tensor, **kwargs) -> tuple[Tensor, Tensor | None]:
        """Run the network and return both heads' raw outputs, unblended.
        Used by loss() (which always needs both heads separately) and by
        forward() (which optionally blends them for IG sampling)."""
        t_emb = self._embed_time(t, x)
        return self.network(t_emb, x, **kwargs)

    def forward(self, t: Tensor, x: Tensor, **kwargs) -> Tensor:
        final, inter = self._forward_raw(t, x, **kwargs)
        if inter is None or self.ig_scale == 1.0:
            return final

        # All of this module's solvers (euler/heun/midpoint here, plus the
        # standalone DPM_Solver and PNDMSolver in dpm.py / generator.py)
        # advance every element of the batch on the same schedule, so t is
        # uniform across the batch at every call; using the first entry to
        # gate the guidance interval is therefore exact, not an approximation.
        t_scalar = t.reshape(-1)[0]
        if self.ig_t_min < t_scalar < self.ig_t_max:
            return inter + self.ig_scale * (final - inter)
        return final

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
        t = torch.rand(
            [x.shape[0]] + [1] * (x.dim() - 1), device=x.device, dtype=x.dtype
        )
        z = noise if noise is not None else torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        # Bypass forward()'s IG blending here on purpose: training always
        # needs the two heads' *raw*, unblended outputs so each can be
        # supervised against the same target u independently (IG paper,
        # Eq. 3-4). This also means loss() is unaffected by whatever
        # ig_scale a generator process may have left set on this instance.
        final, inter = self._forward_raw(t.reshape(-1, 1), y, **kwargs)
        loss = (final - u).square()
        if inter is not None and self.ig_loss_weight > 0.0:
            loss = loss + self.ig_loss_weight * (inter - u).square()
        return loss

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
  ig_loss_weight={self.ig_loss_weight}
)"""