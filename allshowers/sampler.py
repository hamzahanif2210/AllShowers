from typing import Optional

import numpy as np
import torch
from abc import ABC, abstractmethod
from tqdm import tqdm


class Sampler(ABC):
    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        reverse_time=False,
    ):
        """
        Args:
            n_steps: Number of steps to take
            use_edm_schedule: Whether to use the EDM schedule
            heavy_tail: Whether to use a heavy-tailed distribution
            zero_init_padded: Whether to zero out the padded values
            edm_config: Configuration for the EDM schedule (sigma_max, sigma_min, rho)
            random_seed: Random seed for reproducibility
            reverse_time: Whether to reverse the time direction
        """
        self.n_steps = n_steps
        self.use_edm_schedule = use_edm_schedule
        self.heavy_tail = heavy_tail
        self.zero_init_padded = zero_init_padded
        self.reverse_time = reverse_time

        assert not use_edm_schedule or edm_config is not None, (
            "EDM config must be provided if using EDM schedule"
        )

        if self.use_edm_schedule:
            print("Using EDM schedule")
            idxs = torch.arange(n_steps)
            sigma_max = edm_config["sigma_max"]
            sigma_min = edm_config["sigma_min"]
            rho = edm_config["rho"]
            t_steps = (
                sigma_max ** (1 / rho)
                + idxs
                / (n_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            t_steps = torch.flip(t_steps, [0])
        else:
            t_steps = torch.linspace(0, 1, n_steps)
        self.t_steps = torch.cat([t_steps, torch.ones_like(t_steps)[:1]])
        if reverse_time:
            self.t_steps = torch.flip(self.t_steps, [0])

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        self.random_seed = random_seed

        if self.heavy_tail:
            self.distribution = torch.distributions.StudentT(7)
        else:
            self.distribution = torch.distributions.Normal(0, 1)

    def reset(self, random_seed=None):
        if random_seed is None:
            assert self.random_seed is not None, "Random seed not provided"
            random_seed = self.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _init_fastsim(self, shape, mask=None, device=None):
        if device is None:
            device = torch.device("cpu")
        fastsim = self.distribution.sample(shape).to(device)
        if mask is not None and self.zero_init_padded and mask.shape[-1] == 2:
            fastsim[~mask[..., 1]] = 0
        return fastsim

    @torch.no_grad()
    def _transfer(self, x_curr, d_curr, dt):
        return x_curr + d_curr * dt

    @torch.no_grad()
    def _step(self, model, fastsim, timestep, ctxt_dict, dt, **kwargs):
        deriv, _ = model.forward(
            fastsim,
            timestep=timestep.expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt)
        return fastsim_next, deriv

    @torch.no_grad()
    def _midpoint_step(self, model, fastsim, timestep, ctxt_dict, dt, **kwargs):
        deriv = model.forward(
            fastsim,
            timestep=timestep.expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt / 2)
        deriv_next = model.forward(
            fastsim_next,
            timestep=(timestep + dt / 2).expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv_next, dt)
        return fastsim_next, deriv_next

    @torch.no_grad()
    def _heun_step(self, model, fastsim, ctxt_dict, t_cur, t_next, is_last=False, **kwargs):
        fastsim_next, deriv_cur = self._step(
            model, fastsim, timestep=t_cur, dt=t_next - t_cur, ctxt_dict=ctxt_dict,
        )
        deriv_prev = deriv_cur
        if not is_last:
            _, deriv_prime = self._step(
                model, fastsim_next, timestep=t_next, dt=t_next - t_cur, ctxt_dict=ctxt_dict,
            )
            deriv_prev = (1 / 2) * (deriv_cur + deriv_prime)
            fastsim = self._transfer(fastsim, deriv_prev, t_next - t_cur)
        else:
            fastsim = fastsim_next
        return fastsim, deriv_prev

    @torch.no_grad()
    def _rk4_step(self, model, fastsim, t_list, *, ctxt_dict, **kwargs):
        x_2, e_1 = self._step(
            model, fastsim, timestep=t_list[0], dt=t_list[1] - t_list[0], ctxt_dict=ctxt_dict,
        )
        x_3, e_2 = self._step(
            model, x_2, timestep=t_list[1], dt=t_list[1] - t_list[0], ctxt_dict=ctxt_dict,
        )
        x_4, e_3 = self._step(
            model, x_3, timestep=t_list[1], dt=t_list[2] - t_list[0], ctxt_dict=ctxt_dict,
        )
        _, e_4 = self._step(
            model, x_4, timestep=t_list[2], dt=t_list[2] - t_list[0], ctxt_dict=ctxt_dict,
        )
        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        fastsim_next = self._transfer(fastsim, et, t_list[2] - t_list[0])
        return fastsim_next, et

    @abstractmethod
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        to_cpu=True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass


class EulerSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: Optional[torch.Tensor] = None,
        to_cpu=True,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        dtype, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device) if init_fastsim is None else init_fastsim
        fastsim = fastsim.to(dtype)
        t_steps = self.t_steps.to(device).to(dtype)
        seq = [fastsim.cpu()]
        for t_cur, t_next in tqdm(zip(t_steps[:-1], t_steps[1:]), total=self.n_steps):
            fastsim, _ = self._step(
                model, fastsim, timestep=t_cur, dt=t_next - t_cur, ctxt_dict=ctxt_dict,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            return fastsim.cpu(), torch.stack(seq)
        return fastsim.cpu() if to_cpu else fastsim


class NonSingularSampler(Sampler):
    """From http://arxiv.org/abs/2410.02217"""

    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        reverse_time=False,
        g_scale=0.0,
        n=1.0,
        m=0.0,
    ):
        super().__init__(
            n_steps=n_steps,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
            reverse_time=reverse_time,
        )
        self.g_scale = g_scale
        self.n = n
        self.m = m
        print(f"Using Non-Singular sampler with g_scale={g_scale}, n={n}, m={m}")

    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        dtype, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device) if init_fastsim is None else init_fastsim
        fastsim = fastsim.to(dtype)
        t_steps = self.t_steps.to(device).to(dtype)
        seq = [fastsim.cpu()]
        g_scale, n, m = self.g_scale, self.n, self.m

        for t_cur, t_next in tqdm(zip(t_steps[:-1], t_steps[1:]), total=self.n_steps):
            tb = t_cur.view(-1, 1, 1)
            v, _ = model(fastsim, t_cur.expand(fastsim.shape[0]), ctxt_dict)
            dt = t_next - t_cur

            g = g_scale * torch.pow(tb, n / 2) * torch.pow(1.0 - tb, m / 2)
            s_u = -((1.0 - tb) * v + fastsim)
            fr = v - (g_scale**2 * torch.pow(tb, n - 1) * torch.pow(1.0 - tb, m) * s_u / 2.0)
            dbt = torch.sqrt(torch.abs(dt)) * torch.randn_like(fastsim)
            fastsim = fastsim + fr * dt + g * dbt
            if save_seq:
                seq.append(fastsim.cpu())

        if save_seq:
            return fastsim.cpu(), torch.stack(seq)
        return fastsim.cpu() if to_cpu else fastsim


class SDESampler(Sampler):
    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        reverse_time=False,
        eta: float = 1.0,
        last_step_size: float = 1e-3,
        diffusion_mode: str = "linear",
    ):
        super().__init__(
            n_steps=n_steps,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
            reverse_time=reverse_time,
        )
        assert reverse_time, "SDE sampler currently only supports reverse time (RCFM)"

        self.eta = eta
        self.last_step_size = last_step_size
        t0, t1 = 1, last_step_size
        self.t_steps = torch.linspace(t0, t1, n_steps)
        self.t_steps = torch.cat([torch.ones_like(self.t_steps)[:1], self.t_steps])
        print(
            f"Using SDE sampler with eta={eta}, last_step_size={last_step_size}, diffusion_mode={diffusion_mode}"
        )
        print("Using t_steps:", self.t_steps)
        self.diffusion_mode = diffusion_mode

    def get_diffusion_coeff(self, t):
        if self.diffusion_mode == "linear":
            return t
        elif self.diffusion_mode == "sdbm":
            return 2 * (t + t**2 / (1 - t))
        elif self.diffusion_mode == "increasing-decreasing":
            return torch.sin(np.pi * t) ** 2
        else:
            raise ValueError(f"Invalid diffusion mode: {self.diffusion_mode}")

    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: Optional[torch.Tensor] = None,
        to_cpu=True,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        dtype, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device) if init_fastsim is None else init_fastsim
        fastsim = fastsim.to(dtype)
        t_steps = self.t_steps.to(device).to(dtype)

        seq = [fastsim.cpu()]
        for t_cur, t_next in tqdm(zip(t_steps[:-1], t_steps[1:]), total=self.n_steps):
            dt = torch.abs(t_next - t_cur)
            v_theta, _ = model.forward(
                fastsim,
                timestep=t_cur.expand(fastsim.shape[0]),
                ctxt_dict=ctxt_dict,
            )
            eps_hat = fastsim + (1.0 - t_cur) * v_theta
            score = -eps_hat / t_cur
            g_t = self.get_diffusion_coeff(t_cur) * self.eta
            drift = v_theta + 0.5 * (g_t**2) * score
            noise = self.distribution.sample(fastsim.shape).to(device=device, dtype=dtype)
            fastsim = fastsim - drift * dt + g_t * torch.sqrt(dt) * noise
            if save_seq:
                seq.append(fastsim.cpu())

        t_final = torch.tensor(self.last_step_size, device=device, dtype=dtype)
        v_theta, _ = model.forward(
            fastsim,
            timestep=t_final.expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim = fastsim - v_theta * t_final

        if save_seq:
            return fastsim.cpu(), torch.stack(seq)
        return fastsim.cpu() if to_cpu else fastsim


class MidpointSampler(Sampler):
    @torch.inference_mode()
    def sample(self, model, pflow_shape, *, ctxt_dict, save_seq=False, **kwargs):
        mask = ctxt_dict["fs_mask"]
        device = mask.device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        seq = [fastsim.cpu()]
        for t_cur, t_next in tqdm(zip(t_steps[:-1], t_steps[1:]), total=self.n_steps):
            fastsim, _ = self._midpoint_step(
                model, fastsim, timestep=t_cur, dt=t_next - t_cur, ctxt_dict=ctxt_dict,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            return fastsim.cpu(), torch.stack(seq)
        return fastsim.cpu()


class HeunSampler(Sampler):
    @torch.inference_mode()
    def sample(self, model, pflow_shape, *, ctxt_dict, save_seq=False, **kwargs):
        mask = ctxt_dict["fs_mask"]
        _, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._heun_step(
                model, fastsim, t_cur=t_cur, t_next=t_next,
                is_last=(i == self.n_steps - 1), ctxt_dict=ctxt_dict,
            )
        return fastsim.cpu()


class RK4Sampler(Sampler):
    def sample(self, model, pflow_shape, *, ctxt_dict, save_seq=False, **kwargs):
        mask = ctxt_dict["fs_mask"]
        _, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        for t_cur, t_next in tqdm(zip(t_steps[:-1], t_steps[1:]), total=self.n_steps):
            fastsim, _ = self._rk4_step(
                model,
                fastsim,
                t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                ctxt_dict=ctxt_dict,
            )
        return fastsim.cpu()


class PNDMSampler(Sampler):
    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        reverse_time=False,
        init_step="rk4",
    ):
        self.init_step = init_step
        assert self.init_step in ["rk4", "heun", "euler"], "Invalid init_step"
        super().__init__(
            n_steps=n_steps,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
            reverse_time=reverse_time,
        )
        self.step_fn = {
            "rk4": self._rk4_step,
            "heun": self._heun_step,
            "euler": self._step,
        }[self.init_step]

    @torch.inference_mode()
    def sample(self, model, pflow_shape, *, ctxt_dict, save_seq=False, **kwargs):
        mask = ctxt_dict["fs_mask"]
        _, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        ets = []
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            if len(ets) > 2:
                deriv_, _ = model.forward(
                    fastsim,
                    timestep=t_cur.expand(fastsim.shape[0]),
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_)
                deriv = (1 / 24) * (
                    55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]
                )
                fastsim = self._transfer(fastsim, deriv, t_next - t_cur)
            else:
                fastsim, deriv_prev = self.step_fn(
                    model,
                    fastsim,
                    t_cur=t_cur,
                    t_next=t_next,
                    t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                    is_last=(i == self.n_steps - 1),
                    dt=t_next - t_cur,
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_prev)
            if len(ets) > 4:
                ets.pop(0)
        return fastsim.cpu()
