"""
python /n/home04/hhanif/AllShowers/allshowers/generator.py \
  --run-dir  /n/home04/hhanif/AllShowers/results/20260715_053633_CNF-Transformer \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_test_data_with_num_points.h5 \
  --num-samples 6141 \
  --num-timesteps 16 \
  --device cuda:0 \
  --solver midpoint \
  --pdgs 0 1 \
  --max-points 4096

python /n/home04/hhanif/AllShowers/allshowers/generator.py   --run-dir   /n/home04/hhanif/AllShowers/results/20260715_183520_Photons   --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons_test_data_with_num_points.h5   --num-samples 6141  --device cuda:0   --solver dpm --dpm-order 2   --pdgs 0 1   --max-points 6016 --num-timesteps 4

python /n/home04/hhanif/AllShowers/allshowers/generator.py \
  --run-dir   /n/home04/hhanif/AllShowers/results/20260715_183520_Photons \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons_test_data_with_num_points.h5 \
  --num-samples 6141 \
  --num-timesteps 32 \
  --device cuda:0 \
  --solver midpoint \
  --pdgs 0 1 \
  --max-points 6016 



python /n/home04/hhanif/AllShowers/allshowers/generator.py \
  --run-dir  /n/home04/hhanif/AllShowers/results/20260520_160031_Muons-Allshower \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_muons_test_data_with_num_points.h5 \
  --num-samples 6141 \
  --num-timesteps 16 \
  --device cuda:0 \
  --solver midpoint \
  --pdgs 0 1 \
  --max-points 25088 --batch-size 128

"""


import argparse
import os
import platform
import sys
import time
import warnings
from typing import Any
from tqdm import tqdm

import showerdata
import torch
import yaml
from torch import Tensor, nn

from allshowers import flow_matching as fm
from allshowers import transformer
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose
from allshowers.dpm import DPM_Solver
from allshowers.transformer import compute_mask

start = time.perf_counter()

# Solvers handled by allshowers.ode_solvers / CNF.set_solver (heun, euler, midpoint, ...)
# vs. the standalone DPM-Solver, which bypasses CNF.sample entirely. DPM_Solver
# hardcodes the rcfm schedule allshowers.flow_matching.CNF was trained with
# (see allshowers/dpm.py), so there's no separate path/prediction object to
# configure here.
DPM_SOLVER_NAME = "dpm"
PNDM_SOLVER_NAME = "pndm"


# ---------------------------------------------------------------------------
# PNDM (pseudo-numerical, 4th-order linear multistep) solver, adapted from
# parnassus_core.flow.utils.sampler.{Sampler, PNDMSampler}.
#
# Simplifications vs. the original:
#   - No FlowPrediction/ProbabilityPath: your CNF already predicts velocity
#     directly, and FlowPrediction.to_velocity() is a no-op whenever
#     prediction_type == "velocity" (it just returns the model output
#     unchanged) -- so _velocity_fn below calls the CNF directly, with no
#     schedule conversion math needed at all (unlike DPM_Solver above, whose
#     data-prediction parameterization genuinely needed one).
#   - Time steps are plain torch.linspace(1.0, 0.0, n_steps+1) -- exactly the
#     same (t0=1.0, t1=0.0) convention allshowers.flow_matching.CNF.decode
#     already uses for its own euler/heun/midpoint solvers, since the original
#     Sampler's sampling_time_range() for prediction_type="velocity" doesn't
#     adjust the endpoints (that adjustment only applies for "data"/"noise"
#     parameterizations, which don't apply here).
#   - No EDM schedule, heavy-tail noise init, save_seq, or random_seed
#     bookkeeping -- none of your other solvers (heun/midpoint/dpm) have these
#     either, so PNDM is kept consistent with them.
# ---------------------------------------------------------------------------
class PNDMSolver:
    def __init__(self, model_fn, n_steps: int, init_step: str = "rk4") -> None:
        """model_fn must have the same call signature as CNF.__call__:
        model_fn(t, x, cond, num_points, layer, block_mask, label=None) -> velocity
        """
        if init_step not in ("rk4", "heun", "euler"):
            raise ValueError(f"Invalid init_step: {init_step}")
        self.model = model_fn
        self.n_steps = n_steps
        self.init_step = init_step

    def _velocity_fn(self, x, t, cond, num_points, layer, block_mask, label=None):
        return self.model(
            t.expand(x.shape[0]),
            x,
            cond=cond,
            num_points=num_points,
            layer=layer,
            block_mask=block_mask,
            label=label,
        )

    def _transfer(self, x, deriv, dt):
        return x + deriv * dt

    def _step(self, x, t, dt, cond, num_points, layer, block_mask, label=None):
        deriv = self._velocity_fn(x, t, cond, num_points, layer, block_mask, label)
        return self._transfer(x, deriv, dt), deriv

    def _heun_step(self, x, t_cur, t_next, cond, num_points, layer, block_mask, label=None, is_last=False):
        x_next, deriv_cur = self._step(x, t_cur, t_next - t_cur, cond, num_points, layer, block_mask, label)
        if is_last:
            return x_next, deriv_cur
        _, deriv_prime = self._step(x_next, t_next, t_next - t_cur, cond, num_points, layer, block_mask, label)
        deriv = 0.5 * (deriv_cur + deriv_prime)
        x = self._transfer(x, deriv, t_next - t_cur)
        return x, deriv

    def _rk4_step(self, x, t_list, cond, num_points, layer, block_mask, label=None):
        x_2, e_1 = self._step(x, t_list[0], t_list[1] - t_list[0], cond, num_points, layer, block_mask, label)
        x_3, e_2 = self._step(x_2, t_list[1], t_list[1] - t_list[0], cond, num_points, layer, block_mask, label)
        x_4, e_3 = self._step(x_3, t_list[1], t_list[2] - t_list[0], cond, num_points, layer, block_mask, label)
        _, e_4 = self._step(x_4, t_list[2], t_list[2] - t_list[0], cond, num_points, layer, block_mask, label)
        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        x_next = self._transfer(x, et, t_list[2] - t_list[0])
        return x_next, et

    def _initial_step(self, x, t_cur, t_next, cond, num_points, layer, block_mask, label, is_last):
        if self.init_step == "rk4":
            return self._rk4_step(x, [t_cur, (t_cur + t_next) / 2, t_next], cond, num_points, layer, block_mask, label)
        if self.init_step == "heun":
            return self._heun_step(x, t_cur, t_next, cond, num_points, layer, block_mask, label, is_last=is_last)
        return self._step(x, t_cur, t_next - t_cur, cond, num_points, layer, block_mask, label)

    @torch.no_grad()
    def sample(
        self,
        target_shape,
        cond,
        num_points,
        layer,
        block_mask,
        label=None,
        n_steps: int | None = None,
        to_cpu: bool = True,
    ):
        n_steps = self.n_steps if n_steps is None else n_steps
        t_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=cond.device, dtype=cond.dtype)
        x = torch.randn(target_shape, device=cond.device, dtype=cond.dtype)

        ets = []
        for i, (t_cur, t_next) in enumerate(
            tqdm(list(zip(t_steps[:-1], t_steps[1:])), total=n_steps)
        ):
            if len(ets) > 2:
                deriv_ = self._velocity_fn(x, t_cur, cond, num_points, layer, block_mask, label)
                ets.append(deriv_)
                deriv = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
                x = self._transfer(x, deriv, t_next - t_cur)
            else:
                x, deriv_prev = self._initial_step(
                    x, t_cur, t_next, cond, num_points, layer, block_mask, label,
                    is_last=(i == n_steps - 1),
                )
                ets.append(deriv_prev)
            if len(ets) > 4:
                ets.pop(0)

        if to_cpu:
            x = x.cpu()
        return x


class Generator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        compile: bool = False,
        solver: str = "heun",
        resize_factor: float = 1.0,
        max_points: int | None = None,
        checkpoint: str | None = None,
        dpm_order: int = 2,
        dpm_eps: float = 1e-4,
        pndm_init_step: str = "rk4",
    ) -> None:
        super().__init__()

        run_params_file = os.path.join(run_dir, "conf.yaml")
        if checkpoint is not None:
            state_dict_file = checkpoint
        else:
            import glob
            matches = sorted(glob.glob(os.path.join(run_dir, "checkpoints/best*.pt")))
            if not matches:
                raise FileNotFoundError(
                    f"no checkpoint matching checkpoints/best*.pt found in {run_dir}"
                )
            state_dict_file = matches[-1]
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        if not os.path.exists(trafo_file):
            trafo_file = os.path.join(run_dir, "preprocessing/trafos-all.pt")
        self.result_dir = run_dir
        self.num_timesteps = num_timesteps
        self.do_compile = compile
        self.resize_factor = resize_factor

        # DPM-Solver and PNDM aren't in the ode_solvers registry CNF.set_solver
        # uses, so they're handled as separate code paths in forward() instead
        # of via CNF.sample(). Both hardcode the rcfm schedule your CNF was
        # trained with (dpm_eps default matches CNF.loss()'s 1e-4 floor); PNDM
        # needs no such parameter since it works directly on velocity, which is
        # exactly what your CNF already outputs.
        self.solver_name = solver
        self.dpm_order = dpm_order
        self.dpm_eps = dpm_eps
        self.pndm_init_step = pndm_init_step
        if solver in (DPM_SOLVER_NAME, PNDM_SOLVER_NAME):
            # CNF still needs *some* registered integrator for its own internal
            # solver attribute, even though we never call flow.sample() when
            # using dpm/pndm — it's just never exercised on this path.
            model_solver = "heun"
        else:
            model_solver = solver

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        self.__init_model(run_params["model"], state_dict_file, solver=model_solver)
        self.__init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())
        self.feature_last = run_params["data"].get("feature_last", False)
        self.num_layers = run_params["model"].get("num_layers", None)
        self.max_points = max_points if max_points is not None else run_params["data"].get("max_num_points", 6016)
        self.expects_angles = run_params["model"]["dim_inputs"][-1] > 1

        # Auto-detect time mode from config — no CLI flag needed.
        # If the model was trained with samples_time_trafo, dim_inputs[0] == 4.
        self.with_time = run_params["model"]["dim_inputs"][0] == 4

    def __init_model(
        self, params: dict[str, Any], state_file: str, solver: str = "heun"
    ) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in key for key in state_dict)
        if trained_compiled and not self.do_compile:
            for k in list(state_dict.keys()):
                if "_orig_mod." in k:
                    new_k = k.replace("_orig_mod.", "")
                    state_dict[new_k] = state_dict.pop(k)
        elif not trained_compiled and self.do_compile:
            for k in list(state_dict.keys()):
                if "network." in k:
                    new_k = k.replace("network.", "network._orig_mod.")
                    state_dict[new_k] = state_dict.pop(k)
        if self.do_compile:
            network = torch.compile(network)
        self.flow = fm.CNF(network, **flow_config)  # type: ignore
        self.flow.load_state_dict(state_dict)

    def __init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo = compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo = compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo = compose(params.get("cond_trafo"))

        # Time trafo — only present when model was trained with time
        if params.get("samples_time_trafo") is not None:
            self.samples_time_trafo = compose(params.get("samples_time_trafo"))
        else:
            self.samples_time_trafo = None

        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])

        # Load time trafo state if saved in the trafos file
        if self.samples_time_trafo is not None and "samples_time_trafo" in state:
            self.samples_time_trafo.load_state_dict(state["samples_time_trafo"])

    def forward(
        self,
        energies: Tensor,
        num_points: Tensor,
        angles: Tensor,
        label: Tensor | None = None,
    ) -> Tensor:
        if self.expects_angles:
            condition = torch.concatenate(
                [self.cond_trafo(energies * self.resize_factor), angles], dim=-1
            )
        else:
            condition = self.cond_trafo(energies)
        layer = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.int32)
        mask = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.bool)
        for i in range(condition.shape[0]):
            total_points = torch.sum(num_points[i])
            layer_i = torch.repeat_interleave(num_points[i])
            if total_points > self.max_points:
                warnings.warn(
                    f"num points {total_points} exceeds max points {self.max_points}, truncating"
                )
                total_points = self.max_points
                layer_i = layer_i[: self.max_points]
            layer[i, :total_points, 0] = layer_i
            mask[i, :total_points, 0] = True
        layer = layer.to(condition.device)
        mask = mask.to(condition.device)
        num_raw_features = 4 if self.with_time else 3

        raw_samples = self.__sample_raw(
            shape=(condition.shape[0], self.max_points, num_raw_features),
            condition=condition,
            num_points=num_points,
            layer=layer,
            mask=mask,
            label=label,
        )

        if self.with_time:
            # Reconstruct 5-column output: x, y, z(layer), e, t
            samples = torch.zeros(
                (condition.shape[0], self.max_points, 5), device=raw_samples.device
            )
            samples[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
            samples[:, :, 2]  = layer.squeeze(2)
            samples[:, :, 3]  = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
            samples[:, :, 4]  = self.samples_time_trafo.inverse(raw_samples[:, :, 3])
            samples[~mask.repeat(1, 1, 5)] = 0
        else:
            # Reconstruct 4-column output: x, y, z(layer), e
            samples = torch.zeros(
                (condition.shape[0], self.max_points, 4), device=raw_samples.device
            )
            samples[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
            samples[:, :, 2]  = layer.squeeze(2)
            samples[:, :, 3]  = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
            samples[~mask.repeat(1, 1, 4)] = 0

        return samples

    def __sample_raw(
        self,
        shape: tuple[int, int, int],
        condition: Tensor,
        num_points: Tensor,
        layer: Tensor,
        mask: Tensor,
        label: Tensor | None,
    ) -> Tensor:
        """Draw raw (pre-inverse-transform) samples using whichever solver was
        selected at construction time: one of CNF's built-in ODE integrators
        (heun/euler/midpoint/...), the standalone DPM-Solver, or PNDM."""
        if self.solver_name not in (DPM_SOLVER_NAME, PNDM_SOLVER_NAME):
            return self.flow.sample(
                shape=shape,
                num_timesteps=self.num_timesteps,
                cond=condition,
                num_points=num_points,
                layer=layer,
                mask=mask,
                label=label,
            )

        block_mask = compute_mask(
            padding_mask=mask, layer=layer, num_layer_cond=self.flow.num_layer_cond
        )

        if self.solver_name == DPM_SOLVER_NAME:
            solver = DPM_Solver(model_fn=self.flow, eps=self.dpm_eps)
            return solver.sample(
                target_shape=shape,
                cond=condition,
                num_points=num_points,
                layer=layer,
                block_mask=block_mask,
                label=label,
                n_steps=self.num_timesteps,
                order=self.dpm_order,
                to_cpu=False,  # keep on condition's device; forward() runs the
                               # inverse trafo (self.samples_coordinate_trafo etc.)
                               # on-device right after this, same as the ODE path.
            )

        pndm_solver = PNDMSolver(
            model_fn=self.flow, n_steps=self.num_timesteps, init_step=self.pndm_init_step
        )
        return pndm_solver.sample(
            target_shape=shape,
            cond=condition,
            num_points=num_points,
            layer=layer,
            block_mask=block_mask,
            label=label,
            n_steps=self.num_timesteps,
            to_cpu=False,  # keep on condition's device, same reason as DPM above.
        )


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def generate(
    generator: Generator,
    energies: Tensor,
    num_points: Tensor,
    angles: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
    labels: Tensor | None = None,
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_num_points = torch.split(num_points, batch_size, dim=0)
    split_angles = torch.split(angles, batch_size, dim=0)
    if labels is not None:
        split_labels = torch.split(labels, batch_size, dim=0)
    else:
        split_labels = [None] * len(split_energies)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, batch in enumerate(
        zip(split_energies, split_num_points, split_angles, split_labels)
    ):
        print_time(f"start batch {i:3d}")
        batch = [e.to(device) if e is not None else None for e in batch]
        samples_l = generator(*batch).cpu()
        samples.append(samples_l)
    samples = torch.cat(samples)
    print_time("generation done")
    return samples


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new samples")
    parser.add_argument(
        "--run-dir",
        help="directory that contains the model's weights and where the generated samples should be saved",
    )
    parser.add_argument(
        "--cond_file",
        help="file with the conditioning information (e.g. energies, number of points)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1,
        type=int,
        help="number of samples to generate. default: 1",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1024, type=int, help="default: 1024"
    )
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None, help="device for computations")
    parser.add_argument(
        "--num-timesteps",
        default=200,
        type=int,
        help="number of timesteps for the ODE solver. default: 200",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="data type for the generated samples. default: float32",
    )
    parser.add_argument(
        "-r",
        "--rescale-factor",
        default=1.0,
        type=float,
        help="energy rescale factor applied during generation. default: 1.0",
    )
    parser.add_argument(
        "--solver",
        default="heun",
        type=str,
        help=(
            "Sampler to use during generation. Either an ODE integrator name "
            "registered in allshowers.ode_solvers (e.g. heun, euler, midpoint), "
            "'dpm' for the standalone multistep DPM-Solver, or 'pndm' for the "
            "standalone 4th-order linear multistep PNDM solver. default: heun"
        ),
    )
    parser.add_argument(
        "--dpm-order",
        default=2,
        type=int,
        choices=[1, 2, 3],
        help="Order of the multistep DPM-Solver update, only used with --solver dpm. default: 2",
    )
    parser.add_argument(
        "--dpm-eps",
        default=1e-4,
        type=float,
        help=(
            "Numerical floor on the noise coefficient, must match the eps used "
            "in allshowers.flow_matching.CNF.loss() during training. Only used "
            "with --solver dpm. default: 1e-4"
        ),
    )
    parser.add_argument(
        "--pndm-init-step",
        default="rk4",
        type=str,
        choices=["rk4", "heun", "euler"],
        help=(
            "Single-step method used to warm up PNDM's history buffer for its "
            "first few steps, only used with --solver pndm. default: rk4"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="path to a specific checkpoint .pt file. overrides the default weights/best.pt",
    )
    parser.add_argument(
        "--max-points",
        default=None,
        type=int,
        help="maximum number of points per shower, overrides the value in conf.yaml (default: use conf.yaml value, usually 6016)",
    )
    parser.add_argument(
        "--pdgs",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        nargs="+",
        type=int,
        help="list of pdg codes for the labels. default: [11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212]",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    parsed_args.pdgs.sort(key=lambda x: (abs(x), -x))
    print_time("start main")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if parsed_args.dtype not in dtypes:
        raise ValueError(f"invalid dtype: {parsed_args.dtype}")
    dtype = dtypes[parsed_args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    print(yaml.dump(vars(parsed_args)), end="")
    if parsed_args.device:
        device = parsed_args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    if "cuda" in device.lower():
        print("device:", torch.cuda.get_device_name(torch.device(device)))
    elif device.lower() == "cpu":
        print("device:", platform.processor())
    print("num threads:", torch.get_num_threads())
    sys.stdout.flush()

    generator = Generator(
        run_dir=parsed_args.run_dir,
        num_timesteps=parsed_args.num_timesteps,
        compile=("cuda" in device.lower()),
        solver=parsed_args.solver,
        resize_factor=parsed_args.rescale_factor,
        max_points=parsed_args.max_points,
        checkpoint=parsed_args.checkpoint,
        dpm_order=parsed_args.dpm_order,
        dpm_eps=parsed_args.dpm_eps,
        pndm_init_step=parsed_args.pndm_init_step,
    )

    print_time(f"time mode: {'ON (x,y,e,t)' if generator.with_time else 'OFF (x,y,e)'}")

    cond_data = showerdata.observables.read_observables_from_file(
        parsed_args.cond_file,
        observables=[
            "incident_energies",
            "incident_pdg",
            "incident_directions",
            "num_points_per_layer",
        ],
        start=-parsed_args.num_samples,
    )
    energies = torch.from_numpy(cond_data["incident_energies"])
    num_points = torch.from_numpy(cond_data["num_points_per_layer"])
    angle = torch.from_numpy(cond_data["incident_directions"])
    pdg = torch.from_numpy(cond_data["incident_pdg"])
    labels = to_label_tensor(
        pdg=pdg,
        label_list=parsed_args.pdgs,
    )

    energies = energies.to(dtype, copy=False)

    generator.eval()
    generator = generator.to(device)

    samples = generate(
        generator,
        energies,
        num_points,
        angle,
        parsed_args.batch_size,
        device,
        labels,
    )
    showers = showerdata.Showers(
        points=samples.numpy(),
        energies=energies.numpy(),
        directions=angle.numpy(),
        pdg=pdg.numpy(),
    )

    for i in range(100):
        name = f"samples{i:02d}"
        file_path = os.path.join(parsed_args.run_dir, name + ".h5")
        if not os.path.exists(file_path):
            break
    else:
        raise RuntimeError("no free sample file name found")

    showers.save(file_path)
    with open(os.path.join(parsed_args.run_dir, name + ".yaml"), "w") as f:
        yaml.dump(vars(parsed_args), f)

    print(f"saved to {file_path}")
    print_time("all done")


if __name__ == "__main__":
    main()