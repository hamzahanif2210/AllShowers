from typing import final

import torch
from tqdm import tqdm

# Taken from https://github.com/NVlabs/Sana/tree/main, adapted to call
# allshowers.flow_matching.CNF's native signature directly, and hardcoded to
# the exact rectified-flow schedule allshowers.flow_matching.CNF.loss() uses:
#
#   y = (1 - t) * x + (eps + (1 - eps) * t) * z         (x = data, z = noise)
#   u = (1 - eps) * z - x                                (velocity target)
#
# i.e. data coefficient c_d(t) = 1 - t, noise coefficient c_n(t) = eps + (1-eps)*t,
# t=0 is pure data, t=1 is pure noise. No separate FlowPrediction/ProbabilityPath
# abstraction is used here since this file only ever needs this one schedule.


def _pad_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample time tensor to broadcast against x's other dims."""
    return t.reshape(-1, *([1] * (x.ndim - 1)))


class NoiseScheduleFlow:
    def __init__(self, eps: float = 1e-4, schedule: str = "discrete_flow") -> None:
        self.total_N = 1000
        self.T = 1 - 1 / self.total_N
        self.t0 = 1 / self.total_N
        self.schedule = schedule
        self.eps = eps

    def marginal_log_mean_coeff(self, t):
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        """Data (signal) coefficient c_d(t) = 1 - t."""
        return 1 - t

    def marginal_std(self, t):
        """Noise coefficient c_n(t) = eps + (1 - eps) * t."""
        return self.eps + (1 - self.eps) * t

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        low = torch.full_like(lamb, self.t0)
        high = torch.full_like(lamb, self.T)
        for _ in range(64):
            mid = (low + high) / 2
            move_right = self.marginal_lambda(mid) > lamb
            low = torch.where(move_right, mid, low)
            high = torch.where(move_right, high, mid)
        return (low + high) / 2


@final
class DPM_Solver:
    def __init__(self, model_fn, eps: float = 1e-4) -> None:
        """model_fn must have the same call signature as CNF.__call__:
        model_fn(t, x, cond, num_points, layer, block_mask, label=None) -> velocity

        eps must match the eps used in allshowers.flow_matching.CNF.loss()
        (default 1e-4); it's the numerical floor keeping the noise coefficient
        off exactly zero at t=0.
        """
        self.model = model_fn
        self.eps = eps
        self.noise_schedule = NoiseScheduleFlow(eps=eps)

    def data_prediction_fn(self, x, t, cond, num_points, layer, block_mask, label=None):
        """Return the data prediction (x0-estimate) from the CNF's velocity output.

        Derivation: given y = c_d*x + c_n*z and u = (1-eps)*z - x, solving for x:
            x = [(1-eps)*y - c_n*u] / D,   D = c_n + (1-eps)*c_d
        For this affine schedule D is identically 1 for all t (verified
        symbolically), so no division-by-zero guard is needed here.
        """
        velocity = self.model(
            t.expand(x.shape[0]),
            x,
            cond=cond,
            num_points=num_points,
            layer=layer,
            block_mask=block_mask,
            label=label,
        )
        t_ = _pad_like(t, x)
        c_n = self.eps + (1 - self.eps) * t_
        return (1 - self.eps) * x - c_n * velocity

    def model_fn(self, x, t, cond, num_points, layer, block_mask, label=None):
        return self.data_prediction_fn(x, t, cond, num_points, layer, block_mask, label)

    def get_time_steps(self, t_T, t_0, N, device, shift=1.0):
        timesteps = torch.linspace(t_T, t_0, N + 1, device=device)
        return shift * timesteps / (1 + (shift - 1) * timesteps)

    def denoise_to_zero_fn(self, x, s, cond, num_points, layer, block_mask, label=None):
        return self.data_prediction_fn(x, s, cond, num_points, layer, block_mask, label)

    def dpm_solver_first_update(
        self,
        x,
        s,
        t,
        cond,
        num_points,
        layer,
        block_mask,
        label=None,
        model_s=None,
    ):
        """DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`."""
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s, cond, num_points, layer, block_mask, label)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            + (alpha_t * phi_2) * D1
            - (alpha_t * phi_3) * D2
        )
        return x_t

    def multistep_dpm_solver_update(
        self, x, model_prev_list, t_prev_list, t, cond, num_points, layer, block_mask, label, order
    ):
        if order == 1:
            return self.dpm_solver_first_update(
                x, t_prev_list[-1], t, cond, num_points, layer, block_mask, label,
                model_s=model_prev_list[-1],
            )
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    @torch.no_grad()
    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        cond,
        num_points,
        layer,
        block_mask,
        label=None,
        n_steps=20,
        t_start=None,
        t_end=None,
        order=2,
        lower_order_final=True,
        denoise_to_zero=False,
        return_intermediate=False,
        flow_shift=1.0,
        to_cpu=True,
    ):
        """Compute the sample at time `t_end` by DPM-Solver, given the initial
        noise at time `t_start`. `cond`/`num_points`/`layer`/`block_mask`/`label`
        are passed straight through to the CNF on every model call.
        """
        t_0 = self.noise_schedule.t0 if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, (
            "Time range must be positive and lie within the configured schedule"
        )
        x = torch.randn(target_shape, device=cond.device, dtype=cond.dtype)
        device = x.device
        intermediates = []

        def call_model(x_, t_, model_s=None):
            return self.model_fn(x_, t_, cond, num_points, layer, block_mask, label)

        with torch.no_grad():
            assert n_steps >= order
            timesteps = self.get_time_steps(t_T=t_T, t_0=t_0, N=n_steps, device=device, shift=flow_shift)
            assert timesteps.shape[0] - 1 == n_steps
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [call_model(x, t)]
            if return_intermediate:
                intermediates.append(x.cpu())
            for step in range(1, order):
                t = timesteps[step]
                x = self.multistep_dpm_solver_update(
                    x, model_prev_list, t_prev_list, t, cond, num_points, layer, block_mask, label, step,
                )
                if return_intermediate:
                    intermediates.append(x.cpu())
                t_prev_list.append(t)
                model_prev_list.append(call_model(x, t))
            for step in tqdm(range(order, n_steps + 1)):
                t = timesteps[step]
                step_order = min(order, n_steps + 1 - step) if lower_order_final else order
                x = self.multistep_dpm_solver_update(
                    x, model_prev_list, t_prev_list, t, cond, num_points, layer, block_mask, label, step_order,
                )
                if return_intermediate:
                    intermediates.append(x.cpu())
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                if step < n_steps:
                    model_prev_list[-1] = call_model(x, t)
            if denoise_to_zero:
                t = torch.ones((1,), device=device) * t_0
                x = self.denoise_to_zero_fn(x, t, cond, num_points, layer, block_mask, label)
                if return_intermediate:
                    intermediates.append(x.cpu())
        if to_cpu:
            x = x.cpu()
        if return_intermediate:
            return x, torch.stack(intermediates, dim=0)
        return x