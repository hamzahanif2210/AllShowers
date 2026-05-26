# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0
import contextlib
import copy
import threading
from typing import Any, Dict, Optional

import torch


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(
    ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None
):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    Wraps a torch.optim.Optimizer and maintains an Exponential Moving Average
    of the model parameters:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    Use the ``swap_ema_weights()`` context manager to temporarily swap the live
    model weights with the EMA weights for evaluation.

    Args:
        optimizer: the optimizer to wrap
        device: device for EMA parameter storage
        decay: EMA decay factor
        every_n_steps: apply EMA update every N optimizer steps
        current_step: initial step counter (useful when resuming)
        ema_start_step: delay EMA until this many steps have elapsed
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
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> list[torch.Tensor]:
        return [param for group in self.param_groups for param in group["params"]]

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()
            self.first_iteration = False

        if self.current_step >= self.ema_start_step and self.rebuild_ema_params:
            opt_params = list(self.all_parameters())
            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params):]
            )
            self.rebuild_ema_params = False

        if (
            getattr(self.optimizer, "_step_supports_amp_scaling", False)
            and grad_scaler is not None
        ):
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if (
            self.current_step >= self.ema_start_step
            and self._should_update_at_step()
            and not self.rebuild_ema_params
        ):
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
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
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        """Context manager: temporarily swap model weights with EMA weights."""
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()
        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        ema_params = (
            self.ema_params
            if not self.in_saving_ema_model_context
            else list(self.all_parameters())
        )
        return {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
            "ema_start_step": self.ema_start_step,
        }

    def load_state_dict(self, state_dict):
        self.join()
        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(
            param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
        )
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.ema_start_step = state_dict.get("ema_start_step", 0)
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True


try:
    import os
    import lightning.pytorch as pl
    from lightning.pytorch import Callback
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
    from lightning.pytorch.utilities.rank_zero import rank_zero_info

    class EMA(Callback):
        """
        Lightning callback that maintains Exponential Moving Averages of model parameters.

        During validation/test, EMA weights are swapped in (unless validate_original_weights=True).
        Saves an extra EMA checkpoint alongside each regular checkpoint.

        Args:
            decay: EMA decay in [0, 1].
            validate_original_weights: if True, validate with live weights instead of EMA.
            every_n_steps: update EMA every N optimiser steps.
            cpu_offload: store EMA params on CPU.
            ema_start_step: delay EMA until this step.
        """

        def __init__(
            self,
            decay: float,
            validate_original_weights: bool = False,
            every_n_steps: int = 1,
            cpu_offload: bool = False,
            ema_start_step: int = 0,
        ):
            if not (0 <= decay <= 1):
                raise MisconfigurationException("EMA decay value must be between 0 and 1")
            self.decay = decay
            self.validate_original_weights = validate_original_weights
            self.every_n_steps = every_n_steps
            self.cpu_offload = cpu_offload
            self.ema_start_step = ema_start_step

        def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            device = pl_module.device if not self.cpu_offload else torch.device("cpu")
            trainer.optimizers = [
                optim
                if isinstance(optim, EMAOptimizer)
                else EMAOptimizer(
                    optim,
                    device=device,
                    decay=self.decay,
                    every_n_steps=self.every_n_steps,
                    current_step=trainer.global_step,
                    ema_start_step=self.ema_start_step,
                )
                for optim in trainer.optimizers
            ]

        def on_validation_start(self, trainer, pl_module) -> None:
            if self._should_validate_ema_weights(trainer):
                self.swap_model_weights(trainer)

        def on_validation_end(self, trainer, pl_module) -> None:
            if self._should_validate_ema_weights(trainer):
                self.swap_model_weights(trainer)

        def on_test_start(self, trainer, pl_module) -> None:
            if self._should_validate_ema_weights(trainer):
                self.swap_model_weights(trainer)

        def on_test_end(self, trainer, pl_module) -> None:
            if self._should_validate_ema_weights(trainer):
                self.swap_model_weights(trainer)

        def _should_validate_ema_weights(self, trainer) -> bool:
            return not self.validate_original_weights and self._ema_initialized(trainer)

        def _ema_initialized(self, trainer) -> bool:
            return any(isinstance(opt, EMAOptimizer) for opt in trainer.optimizers)

        def swap_model_weights(self, trainer, saving_ema_model: bool = False):
            for optimizer in trainer.optimizers:
                assert isinstance(optimizer, EMAOptimizer)
                optimizer.switch_main_parameter_weights(saving_ema_model)

        @contextlib.contextmanager
        def save_ema_model(self, trainer):
            self.swap_model_weights(trainer, saving_ema_model=True)
            try:
                yield
            finally:
                self.swap_model_weights(trainer, saving_ema_model=False)

        @contextlib.contextmanager
        def save_original_optimizer_state(self, trainer):
            for optimizer in trainer.optimizers:
                assert isinstance(optimizer, EMAOptimizer)
                optimizer.save_original_optimizer_state = True
            try:
                yield
            finally:
                for optimizer in trainer.optimizers:
                    optimizer.save_original_optimizer_state = False

        def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
            checkpoint_callback = trainer.checkpoint_callback
            ckpt_path = trainer.ckpt_path

            if ckpt_path and checkpoint_callback is not None:
                ext = checkpoint_callback.FILE_EXTENSION
                if ckpt_path.endswith(f"-EMA{ext}"):
                    rank_zero_info(
                        "Loading EMA weights. They will be treated as the main weights "
                        "and a new EMA copy will be created when training resumes."
                    )
                    return
                ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
                if os.path.exists(ema_path):
                    ema_state_dict = torch.load(
                        ema_path, map_location=torch.device("cpu"), weights_only=False
                    )
                    checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                    del ema_state_dict
                    rank_zero_info("EMA state has been restored.")
                else:
                    raise MisconfigurationException(
                        "Unable to find the associated EMA weights when re-loading. "
                        f"Expected them at: {ema_path}"
                    )

    class EMAModelCheckpoint(ModelCheckpoint):
        """ModelCheckpoint that also saves a separate EMA copy of the model."""

        def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional["EMA"]:
            for callback in trainer.callbacks:
                if isinstance(callback, EMA):
                    return callback
            return None

        def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
            ema_callback = self._get_ema_callback(trainer)
            if ema_callback is not None:
                with ema_callback.save_original_optimizer_state(trainer):
                    super()._save_checkpoint(trainer, filepath)
                with ema_callback.save_ema_model(trainer):
                    ema_filepath = self._ema_format_filepath(filepath)
                    if self.verbose:
                        rank_zero_info(f"Saving EMA weights to {ema_filepath}")
                    super()._save_checkpoint(trainer, ema_filepath)
            else:
                super()._save_checkpoint(trainer, filepath)

        def _ema_format_filepath(self, filepath: str) -> str:
            return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

except ImportError:
    pass
