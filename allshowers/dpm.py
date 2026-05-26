from typing import final

import torch
from tqdm import tqdm

# Taken from https://github.com/NVlabs/Sana/tree/main


class NoiseScheduleFlow:
    def __init__(self, schedule="discrete_flow"):
        self.T = 1
        self.t0 = 0.001
        self.schedule = schedule
        self.total_N = 1000

    def marginal_log_mean_coeff(self, t):
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        return 1 - t

    @staticmethod
    def marginal_std(t):
        return t

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    @staticmethod
    def inverse_lambda(lamb):
        return 1 / (torch.exp(lamb) + 1)

    def edm_sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        sigma = edmsigma
        lambda_t = torch.log(1 / sigma)
        return self.inverse_lambda(lambda_t)


def get_noise_from_velocity(velocity, x, t):
    return (1 - t) * velocity - x


def expand_dims(v, dims):
    """Expand tensor v of shape [N] to [N, 1, 1, ..., 1] with total `dims` dimensions."""
    return v[(...,) + (None,) * (dims - 1)]


@final
class DPM_Solver:
    def __init__(self, model_fn):
        """Construct a DPM-Solver."""
        self.model = lambda x, timestep, ctxt_dict: model_fn(
            x, timestep.expand(x.shape[0]), ctxt_dict
        )
        self.noise_schedule = NoiseScheduleFlow()

    def noise_prediction_fn(self, x, t, ctxt_dict):
        return self.model(x, timestep=t, ctxt_dict=ctxt_dict)

    def data_prediction_fn(self, x, t, ctxt_dict):
        noise = self.noise_prediction_fn(x, t, ctxt_dict)
        alpha_t, sigma_t = (
            self.noise_schedule.marginal_alpha(t),
            self.noise_schedule.marginal_std(t),
        )
        x0 = (x - sigma_t * noise) / alpha_t
        return x0

    def model_fn(self, x, t, ctxt_dict):
        return self.data_prediction_fn(x, t, ctxt_dict)

    def get_time_steps(self, t_T, t_0, N, device, shift=1.0):
        betas = torch.linspace(t_T, t_0, N + 1).to(device)
        sigmas = 1.0 - betas
        sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
        return sigmas

    def denoise_to_zero_fn(self, x, s, ctxt_dict):
        return self.data_prediction_fn(x, s, ctxt_dict)

    def dpm_solver_first_update(self, x, s, t, ctxt_dict, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s, ctxt_dict)
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

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, ctxt_dict, order):
        if order == 1:
            return self.dpm_solver_first_update(
                x, t_prev_list[-1], t, ctxt_dict, model_s=model_prev_list[-1],
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
        x,
        ctxt_dict,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        lower_order_final=True,
        denoise_to_zero=False,
        return_intermediate=False,
        flow_shift=1.0,
    ):
        t_0 = self.noise_schedule.t0 if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0
        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= order
            timesteps = self.get_time_steps(t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            assert timesteps.shape[0] - 1 == steps
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [self.model_fn(x, t, ctxt_dict)]
            if return_intermediate:
                intermediates.append(x.cpu())
            for step in range(1, order):
                t = timesteps[step]
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, ctxt_dict, step)
                if return_intermediate:
                    intermediates.append(x.cpu())
                t_prev_list.append(t)
                model_prev_list.append(self.model_fn(x, t, ctxt_dict))
            for step in tqdm(range(order, steps + 1)):
                t = timesteps[step]
                step_order = min(order, steps + 1 - step) if lower_order_final else order
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, ctxt_dict, step_order)
                if return_intermediate:
                    intermediates.append(x.cpu())
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, t, ctxt_dict)
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t, ctxt_dict)
                if return_intermediate:
                    intermediates.append(x.cpu())
        if return_intermediate:
            return x, torch.stack(intermediates, dim=0)
        return x
