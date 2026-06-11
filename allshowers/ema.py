# Adapted from NVIDIA NeMo (Apache 2.0 License).
# Original: https://github.com/NVIDIA/NeMo
# Changes: removed Lightning/Callback/EMAModelCheckpoint dependencies;
# kept EMAOptimizer and ema_update for use with the plain-PyTorch Trainer.

import contextlib
import copy
import threading
from typing import Any

import torch

__all__ = ["EMAOptimizer", "ema_update"]


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def _run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    Wraps any torch.optim.Optimizer and maintains an Exponential Moving
    Average (EMA) of all its parameters.

    After every optimizer step the EMA shadow weights are updated:
        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    Use the ``swap_ema_weights()`` context manager to temporarily replace the
    model's live weights with the EMA weights (e.g. for evaluation or saving).

    Args:
        optimizer:      the optimizer to wrap
        device:         device on which to keep the EMA shadow copy
        decay:          EMA decay factor (default 0.9999)
        every_n_steps:  update EMA every N optimizer steps (default 1)
        current_step:   starting step count, useful when resuming (default 0)
        ema_start_step: do not start accumulating EMA until this step (default 0)

    Example::

        opt = EMAOptimizer(torch.optim.Adam(model.parameters()), device, decay=0.9999)
        for batch in loader:
            loss.backward()
            opt.step()
            opt.zero_grad()

        # evaluate with EMA weights
        with opt.swap_ema_weights():
            val_loss = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
        ema_start_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.ema_start_step = ema_start_step

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None
        self.ema_params: tuple[torch.Tensor, ...] = ()

    # ------------------------------------------------------------------ #
    #  torch.optim.Optimizer interface                                     #
    # ------------------------------------------------------------------ #

    def all_parameters(self) -> list[torch.Tensor]:
        return [param for group in self.param_groups for param in group["params"]]

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()
            self.first_iteration = False

        # Lazily initialise EMA shadow copy after ema_start_step.
        if self.current_step >= self.ema_start_step and self.rebuild_ema_params:
            opt_params = list(self.all_parameters())
            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params):]
            )
            self.rebuild_ema_params = False

        # Actual optimizer step.
        if (
            getattr(self.optimizer, "_step_supports_amp_scaling", False)
            and grad_scaler is not None
        ):
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        # EMA update.
        if (
            self.current_step >= self.ema_start_step
            and self.current_step % self.every_n_steps == 0
            and not self.rebuild_ema_params
        ):
            self._update()

        self.current_step += 1
        return loss

    @torch.no_grad()
    def _update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )
            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=_run_ema_update_cpu,
                args=(self.ema_params, current_model_state, self.decay, self.stream),
            )
            self.thread.start()

    # ------------------------------------------------------------------ #
    #  Weight swapping                                                     #
    # ------------------------------------------------------------------ #

    def _swap_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self):
        """In-place swap model weights ↔ EMA shadow weights."""
        self.join()
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self._swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        """Context manager: temporarily replace model weights with EMA weights."""
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        self.join()
        return {
            "opt": self.optimizer.state_dict(),
            "ema": self.ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
            "ema_start_step": self.ema_start_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.join()
        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(
            param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
        )
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.ema_start_step = state_dict.get("ema_start_step", 0)  # backwards compat
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True

    # ------------------------------------------------------------------ #
    #  Delegate everything else to the inner optimizer                    #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)

    def join(self):
        """Wait for any async CUDA/CPU EMA update to finish."""
        if self.stream is not None:
            self.stream.synchronize()
        if self.thread is not None:
            self.thread.join()