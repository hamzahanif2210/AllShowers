#!/usr/bin/env python3

'''
python /n/home04/hhanif/AllShowers/allshowers/create_dataset_for_eval.py \
  --input /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons.h5  \
  --num-layers 24 --with-time

python /n/home04/hhanif/AllShowers/allshowers/create_dataset_for_eval.py \
  --input /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons.h5  \
  --num-layers 24 --with-time

python /n/home04/hhanif/AllShowers/allshowers/create_dataset_for_eval.py \
  --input /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_muons.h5  \
  --num-layers 24 --with-time

python /n/home04/hhanif/AllShowers/allshowers/create_dataset_for_eval.py \
  --input /n/home04/hhanif/test_remaining.h5  \
  --num-layers 76 

'''

import argparse
import os
import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


# -------------------------------------------------
# Worker: opens the file once, processes a chunk
# -------------------------------------------------
def _worker_chunk(args_tuple):
    """
    Process a contiguous range of showers [start, stop).
    Opens the HDF5 file once per chunk, not once per shower.
    Returns (start, points_per_layer_2d).
    """
    input_path, start, stop, num_layers, num_cols = args_tuple
    num_showers = stop - start
    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)

    with h5py.File(input_path, "r") as f:
        dataset = f["showers"]
        for i, global_i in enumerate(range(start, stop)):
            shower = np.array(dataset[global_i])
            points = shower.reshape(-1, num_cols)
            layer_idx = np.clip(
                (points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1
            )
            mask = (points[:, 3] > 0).astype(np.int32)
            np.add.at(points_per_layer[i], layer_idx, mask)

    return start, points_per_layer


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Copy HDF5 file (excluding 'showers') and add num_points_per_layer dataset"
    )
    parser.add_argument(
        "--input",
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 file path. Defaults to <input_stem>_data_with_num_points.h5 in the same directory.",
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Number of showers per worker chunk.")
    parser.add_argument("--num-workers", type=int, default=min(mp.cpu_count(), 32),
                        help="Number of parallel worker processes (default: min(cpu_count, 32)).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--with-time",
        action="store_true",
        default=False,
        help=(
            "Treat shower data as 5-column format (x, y, z, e, t). "
            "Without this flag, the original 4-column format (x, y, z, e) is used."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    num_cols = 5 if args.with_time else 4
    print(f"Mode: {'with time (x,y,z,e,t) — 5 cols' if args.with_time else 'original (x,y,z,e) — 4 cols'}")
    print(f"Workers: {args.num_workers}  |  Chunk size: {args.chunk_size}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_data_with_num_points{ext}"

    if os.path.isfile(args.output):
        if not args.overwrite:
            print(f"Output file '{args.output}' already exists. Use --overwrite to recompute.")
            return
        else:
            print(f"Output file '{args.output}' already exists — deleting and recomputing.")
            os.remove(args.output)

    DATASET_NAME = "observables/num_points_per_layer"
    SKIP_DATASETS = {"showers", "target", DATASET_NAME}

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    with h5py.File(args.input, "r") as h_in, h5py.File(args.output, "w") as h_out:

        N = h_in["showers"].shape[0]

        # -------------------------------------------------
        # Copy all datasets except 'showers'
        # -------------------------------------------------
        print("\nCopying datasets (excluding 'showers')...")
        for name in h_in:
            if name in SKIP_DATASETS:
                print(f"  - Skipping '{name}'")
                continue
            item = h_in[name]
            if isinstance(item, h5py.Dataset):
                print(f"  - Copying '{name}': {item.shape}")
            else:
                print(f"  - Copying '{name}' (group)")
            h_in.copy(name, h_out)

        for key, val in h_in.attrs.items():
            h_out.attrs[key] = val

        # -------------------------------------------------
        # Create output dataset
        # -------------------------------------------------
        print(f"\nCreating '{DATASET_NAME}' with shape ({N}, {args.num_layers})...")
        d_np = h_out.create_dataset(
            DATASET_NAME,
            shape=(N, args.num_layers),
            dtype=np.int32,
            chunks=(min(args.chunk_size, N), args.num_layers),
            compression="gzip",
            shuffle=True,
        )

        # -------------------------------------------------
        # Build chunk ranges and dispatch to workers
        # Each worker opens the file once and processes a
        # contiguous block — far fewer file opens than
        # the per-shower approach.
        # -------------------------------------------------
        chunks = [
            (args.input, start, min(N, start + args.chunk_size), args.num_layers, num_cols)
            for start in range(0, N, args.chunk_size)
        ]
        num_chunks = len(chunks)
        print(f"Dispatching {num_chunks} chunks across {args.num_workers} workers...\n")

        with mp.Pool(processes=args.num_workers) as pool:
            with tqdm(total=N, desc="Processing showers", unit="shower") as pbar:
                for start, result in pool.imap_unordered(_worker_chunk, chunks):
                    d_np[start : start + len(result)] = result
                    pbar.update(len(result))

    print("\nDone.")
    print(f"Output file: {args.output}")
    with h5py.File(args.output, "r") as h:
        print("Datasets inside:")
        for name in h:
            item = h[name]
            if isinstance(item, h5py.Dataset):
                print(f"  - {name}: {item.shape}")
            elif isinstance(item, h5py.Group):
                print(f"  - {name}: Group")


if __name__ == "__main__":
    main()