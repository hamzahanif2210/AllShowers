"""
Sweep Internal Guidance's ig_scale (and optionally ig_t_min/ig_t_max) over a
trained checkpoint, without paying the cost of reloading the model/trafos for
every value the way repeated `python generator.py --ig-scale ...` CLI calls
would.

Loads showerdata/torch/yaml once, builds allshowers.generator.Generator once,
then for each requested ig_scale:
  1. calls flow.enable_internal_guidance(scale=..., t_min=..., t_max=...)
     (or flow.disable_internal_guidance() for scale == 1.0)
  2. runs allshowers.generator.generate(...) as usual
  3. saves the result to <run-dir>/sweep_ig<value>_samplesNN.h5, plus a
     sidecar yaml recording the exact ig settings used, so you can tell runs
     apart later without parsing filenames.

Example
-------
python /n/home04/hhanif/AllShowers/allshowers/sweep_ig_scale.py \
  --run-dir /n/home04/hhanif/AllShowers/results/20260718_010354_Electrons \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_test_data_with_num_points.h5 \
  --num-samples 6141 \
  --solver dpm --dpm-order 2 --num-timesteps 4 \
  --pdgs 0 1 --max-points 4096 \
  --ig-scales 1.0 1.1 1.4 1.8 2.3 \
  --ig-t-min 0.3 --ig-t-max 1.0

ig-scales includes 1.0 on purpose -- that's your no-IG baseline, generated
with the exact same noise-generation code path as the IG runs (only the
enable/disable_internal_guidance() call differs), so comparisons are as
apples-to-apples as possible.
"""

import argparse
import os
import platform
import sys
import time

import showerdata
import torch
import yaml

from allshowers.data_sets import to_label_tensor
from allshowers.generator import Generator, generate, print_time


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Internal Guidance ig_scale over a single loaded checkpoint"
    )
    # --- shared with generator.py --------------------------------------------
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cond_file", required=True)
    parser.add_argument("-n", "--num-samples", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--num-timesteps", default=200, type=int)
    parser.add_argument("--dtype", default="float32", type=str)
    parser.add_argument("-r", "--rescale-factor", default=1.0, type=float)
    parser.add_argument("--solver", default="heun", type=str)
    parser.add_argument("--dpm-order", default=2, type=int, choices=[1, 2, 3])
    parser.add_argument("--dpm-eps", default=1e-4, type=float)
    parser.add_argument(
        "--pndm-init-step", default="rk4", type=str, choices=["rk4", "heun", "euler"]
    )
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--max-points", default=None, type=int)
    parser.add_argument(
        "--pdgs",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        nargs="+",
        type=int,
    )
    # --- sweep-specific -------------------------------------------------------
    parser.add_argument(
        "--ig-scales",
        nargs="+",
        type=float,
        required=True,
        help=(
            "List of ig_scale values to sweep. Include 1.0 to also generate "
            "a no-IG baseline (recommended) using the same loaded model."
        ),
    )
    parser.add_argument(
        "--ig-t-min",
        default=0.3,
        type=float,
        help="Guidance interval lower bound, applied to every ig_scale != 1.0 in the sweep.",
    )
    parser.add_argument(
        "--ig-t-max",
        default=1.0,
        type=float,
        help="Guidance interval upper bound, applied to every ig_scale != 1.0 in the sweep.",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    parsed_args.pdgs.sort(key=lambda x: (abs(x), -x))
    start = time.perf_counter()

    def log(text):
        print(f"[{int(time.perf_counter() - start):6d}s]: {text}")
        sys.stdout.flush()

    log("start sweep")
    dtypes = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
    if parsed_args.dtype not in dtypes:
        raise ValueError(f"invalid dtype: {parsed_args.dtype}")
    dtype = dtypes[parsed_args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)

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
    sys.stdout.flush()

    # Build the model ONCE. ig_scale here is irrelevant -- it's overwritten
    # per-sweep-value below via enable_internal_guidance()/disable_internal_guidance().
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
        ig_scale=1.0,  # placeholder; real values set per-iteration below
    )
    log(f"time mode: {'ON (x,y,e,t)' if generator.with_time else 'OFF (x,y,e)'}")

    if any(s != 1.0 for s in parsed_args.ig_scales) and generator.flow.network.inter_head is None:
        raise RuntimeError(
            "Requested ig_scale != 1.0, but this checkpoint's network has no "
            "intermediate head (it was not trained with intermediate_layer_idx "
            ">= 0). Retrain with that option set before sweeping IG."
        )

    # Conditioning data is identical across the whole sweep -- load it once.
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
    energies = torch.from_numpy(cond_data["incident_energies"]).to(dtype, copy=False)
    num_points = torch.from_numpy(cond_data["num_points_per_layer"])
    angle = torch.from_numpy(cond_data["incident_directions"])
    pdg = torch.from_numpy(cond_data["incident_pdg"])
    labels = to_label_tensor(pdg=pdg, label_list=parsed_args.pdgs)

    generator.eval()
    generator = generator.to(device)

    for ig_scale in parsed_args.ig_scales:
        log(f"=== ig_scale={ig_scale} ===")
        if ig_scale == 1.0:
            generator.flow.disable_internal_guidance()
        else:
            generator.flow.enable_internal_guidance(
                scale=ig_scale, t_min=parsed_args.ig_t_min, t_max=parsed_args.ig_t_max
            )

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

        tag = f"ig{ig_scale:.3g}".replace(".", "p").replace("-", "m")
        for i in range(100):
            name = f"sweep_{tag}_samples{i:02d}"
            file_path = os.path.join(parsed_args.run_dir, name + ".h5")
            if not os.path.exists(file_path):
                break
        else:
            raise RuntimeError(f"no free sample file name found for ig_scale={ig_scale}")

        showers.save(file_path)
        sidecar = dict(vars(parsed_args))
        sidecar["ig_scale"] = ig_scale  # the actual value used this iteration
        with open(os.path.join(parsed_args.run_dir, name + ".yaml"), "w") as f:
            yaml.dump(sidecar, f)
        log(f"saved to {file_path}")

    log("sweep done")


if __name__ == "__main__":
    main()