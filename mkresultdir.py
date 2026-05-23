#!/usr/bin/env python3
"""
mkresultdir.py
"""


"""
python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/electrons.yaml -p gpu_requeue -g 1 -n 1 --mem 64G --time 12:00:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/photons.yaml -p gpu_requeue -g 1 -n 1 --mem 128G --time 4:00:00 -r


python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/muons.yaml -p gpu -g 1 -n 1 --mem 500G --time 12:00:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/geant4.yaml -p gpu_requeue -g 1 -n 1 --mem 128G --time 4:00:00 -r



"""
import argparse
import os
from pathlib import Path

import yaml

from allshowers import util


JOB_SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={name:s}
#SBATCH --mem={mem:s}
#SBATCH --time={time_limit:s}
#SBATCH -p {partition:s}
#SBATCH --constraint="a100|h100|h200"
{gres_line:s}
#SBATCH --nodes={num_nodes:d}
#SBATCH --output={result_path:s}/log/train_%j.out
#SBATCH --error={result_path:s}/log/train_%j.err
{mail_lines:s}

echo "job id: $SLURM_JOB_ID"
echo "node list: $SLURM_JOB_NODELIST"
echo ""

# Launch one task per GPU across all nodes.
srun \\
  --nodes={num_nodes:d} \\
  --ntasks-per-node={num_gpus:d} \\
  --kill-on-bad-exit=1 \\
  bash {result_path:s}/script.sh
"""


WORKER_SCRIPT_TEMPLATE = """\
#!/bin/env bash
set -euo pipefail

cd {repo_path:s}

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate {mamba_env:s}

_MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(python3 -c "
import socket
print(socket.getaddrinfo('$_MASTER_HOSTNAME', None, socket.AF_INET)[0][4][0])
")

export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export GLOO_USE_IPV6=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export GLOO_SOCKET_IFNAME=^lo,docker0

num_cpus=$(nproc --all)
num_gpus=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( num_cpus / num_gpus ))
if [ "$OMP_NUM_THREADS" -lt 1 ]; then
  export OMP_NUM_THREADS=1
fi

echo "node:        $(uname -n)"
echo "rank:        $RANK / $WORLD_SIZE"
echo "local_rank:  $LOCAL_RANK"
echo "master:      $MASTER_ADDR:$MASTER_PORT  (resolved from $_MASTER_HOSTNAME)"
echo "num CPUs:    $num_cpus"
echo "num GPUs:    $num_gpus"
grep MemTotal /proc/meminfo || true

echo ""
echo "config file: {config_rel:s}"
echo "start time: $(date)"
echo ""

python allshowers/train.py --ddp {config_rel:s}
"""


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create result directory + Slurm scripts, optionally submit with sbatch."
    )

    p.add_argument("param_file", help="YAML parameter file input.")

    p.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Submit the job via sbatch after creating scripts.",
    )

    p.add_argument(
        "--resume-training",
        action="store_true",
        help=(
            "Resume training in an existing result directory. "
            "Must be used together with --resume-ckpt-file."
        ),
    )

    p.add_argument(
        "--resume-ckpt-file",
        type=str,
        default="",
        help=(
            "Absolute path to the checkpoint file to resume from. "
            "Required when --resume-training is set."
        ),
    )

    p.add_argument(
        "-p",
        "--partition",
        choices=["gpu", "gpu_requeue", "gpu_h200", "arguelles_delgado_gpu_mixed"],
        default="gpu",
        help="SLURM partition.",
    )

    p.add_argument(
        "-g",
        "--num_gpu",
        type=int,
        default=1,
        help="GPUs per node. Default: 1",
    )

    p.add_argument(
        "-n",
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes. Default: 1",
    )

    p.add_argument(
        "--mem",
        type=str,
        default="300G",
        help='Memory request. Default: "300G"',
    )

    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Ignored. Kept only so old commands do not break.",
    )

    p.add_argument(
        "--time",
        type=str,
        default="2-00:00:00",
        help='Time limit. Default: "2-00:00:00"',
    )

    p.add_argument(
        "--mamba-env",
        type=str,
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/python313/",
        help="Full path to the mamba environment to activate.",
    )

    p.add_argument(
        "--mail",
        type=str,
        default="",
        help="Email address for Slurm notifications. Leave empty to disable.",
    )

    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = get_args()

    # ------------------------------------------------------------------ #
    # Resume mode: reuse an existing result directory instead of creating #
    # a new one. The checkpoint path is written into conf.yaml so that    #
    # train.py picks it up via conf["resume_ckpt_file"].                  #
    # ------------------------------------------------------------------ #
    if args.resume_training:
        if not args.resume_ckpt_file:
            raise ValueError(
                "--resume-ckpt-file is required when --resume-training is set."
            )
        ckpt_path = os.path.abspath(args.resume_ckpt_file)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}"
            )

        with open(args.param_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        # result_path must already be set in the conf (written by the original run)
        if "result_path" not in params:
            raise ValueError(
                "The YAML file does not contain a result_path. "
                "Pass the conf.yaml from an existing result directory."
            )
        result_path = Path(params["result_path"])
        if not result_path.is_dir():
            raise FileNotFoundError(
                f"result_path directory not found: {result_path}"
            )

        # Inject the checkpoint path so train.py knows where to resume from.
        params["resume_ckpt_file"] = ckpt_path

        conf_file = result_path / "conf.yaml"
        with open(conf_file, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

        print(f"Resuming run in: {result_path}")
        print(f"Checkpoint:      {ckpt_path}")

    else:
        # ------------------------------------------------------------------ #
        # Normal mode: create a fresh result directory.                       #
        # ------------------------------------------------------------------ #
        with open(args.param_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params["result_path"] = util.setup_result_path(params["run_name"], args.param_file)
        result_path = Path(params["result_path"])

        for d in ["checkpoints", "weights", "plots", "log", "preprocessing", "data"]:
            ensure_dir(result_path / d)

        conf_file = result_path / "conf.yaml"
        with open(conf_file, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Write / overwrite job scripts (shared by both modes).               #
    # ------------------------------------------------------------------ #
    run_file = result_path / "run.sh"
    worker_file = result_path / "script.sh"

    repo_path = Path(__file__).resolve().parent
    config_rel = os.path.relpath(str(conf_file), str(repo_path))

    mail_lines = ""
    if args.mail.strip():
        mail_lines = (
            "#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user={args.mail.strip()}\n"
        )

    if args.partition == "arguelles_delgado_gpu_mixed":
        gres_line = "#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1"
    else:
        gres_line = f"#SBATCH --gres=gpu:{args.num_gpu}"

    job_script = JOB_SCRIPT_TEMPLATE.format(
        name=params.get("run_name", "allshowers"),
        mem=args.mem,
        time_limit=args.time,
        partition=args.partition,
        gres_line=gres_line,
        num_nodes=args.num_nodes,
        num_gpus=args.num_gpu,
        result_path=str(result_path),
        mail_lines=mail_lines.rstrip("\n"),
    )

    with open(run_file, "w") as f:
        f.write(job_script + "\n")
    os.chmod(run_file, 0o750)

    worker_script = WORKER_SCRIPT_TEMPLATE.format(
        repo_path=str(repo_path),
        mamba_env=args.mamba_env,
        config_rel=config_rel,
    )

    with open(worker_file, "w") as f:
        f.write(worker_script + "\n")
    os.chmod(worker_file, 0o750)

    cmd = f"sbatch {run_file}"
    print(cmd)

    if args.run:
        print(os.popen(cmd).read())


if __name__ == "__main__":
    main()