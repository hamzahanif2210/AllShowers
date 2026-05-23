import argparse
import multiprocessing
import os
import pickle
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing

start = time.time()
batch_type = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]
]


def print_time(*args, **kwargs) -> None:
    elapsed = time.time() - start
    print(f"[{elapsed: 5.2f}s]", *args, **kwargs)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match noise to points using OT and save it to the file. "
            "The mapping is done for each shower and each layer separately."
        )
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--with-time",
        action="store_true",
        default=False,
        help=(
            "Include time as a 4th point feature (x, y, e, t) when computing OT. "
            "Requires 'samples_time_trafo' in the config and a 5-column data file. "
            "Without this flag, the original 3-feature mode (x, y, e) is used."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Three-phase / big-file mode                                        #
    # ------------------------------------------------------------------ #
    big_file_group = parser.add_argument_group(
        "three-phase big-file mode",
        description=(
            "Phase 0 – fit transformations and save to pickle (--fit-only)\n"
            "Phase 1 – array job, one chunk per task         (--big-file --chunk-index I)\n"
            "Phase 2 – merge chunks into the data file       (--merge)\n\n"
            "All three phases require --pkl-path to point at the same file."
        ),
    )
    big_file_group.add_argument(
        "--fit-only",
        action="store_true",
        default=False,
        help=(
            "Phase 0: fit transformations on a small sample of the data, "
            "serialize the PreProcessor to --pkl-path, then exit."
        ),
    )
    big_file_group.add_argument(
        "--pkl-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to save (Phase 0) or load (Phases 1 & 2) the fitted "
            "PreProcessor pickle.  Required for all three phases."
        ),
    )
    big_file_group.add_argument(
        "--big-file",
        action="store_true",
        default=False,
        help=(
            "Phase 1: process one chunk of --chunk-size samples "
            "(selected via --chunk-index) and save noise to a chunk HDF5 file."
        ),
    )
    big_file_group.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of samples per chunk (default: 1000).",
    )
    big_file_group.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        metavar="I",
        help=(
            "Zero-based index of the chunk to process in this job. "
            "Set to $SLURM_ARRAY_TASK_ID in the SLURM array script."
        ),
    )
    big_file_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory where per-chunk HDF5 files are written (and read during "
            "merge).  Defaults to <data_file>_chunks/ next to the data file."
        ),
    )
    big_file_group.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help=(
            "Phase 2: merge all per-chunk HDF5 files into the original data file, "
            "then delete chunk files, chunk directory, and the pickle."
        ),
    )

    parsed = parser.parse_args(args)

    # Validation
    if parsed.big_file and parsed.chunk_index is None:
        parser.error("--big-file requires --chunk-index.")
    if parsed.merge and parsed.big_file:
        parser.error("--merge and --big-file are mutually exclusive.")
    if parsed.fit_only and (parsed.big_file or parsed.merge):
        parser.error("--fit-only is mutually exclusive with --big-file and --merge.")
    if (parsed.fit_only or parsed.big_file or parsed.merge) and parsed.pkl_path is None:
        parser.error("--pkl-path is required for --fit-only, --big-file, and --merge.")

    return parsed


# ======================================================================= #
#  PreProcessor                                                            #
# ======================================================================= #

class PreProcessor:
    def __init__(self, config_file: str, with_time: bool = False,
                 fit_stop: int = 10000) -> None:
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.with_time = with_time
        self.num_features = 4 if with_time else 3

        self.samples_energy_trafo = preprocessing.compose(
            transformation=config["data"]["samples_energy_trafo"],
        )
        self.samples_coordinate_trafo = preprocessing.compose(
            transformation=config["data"]["samples_coordinate_trafo"],
        )

        if self.with_time:
            if "samples_time_trafo" not in config["data"]:
                raise KeyError(
                    "'--with-time' was set but 'samples_time_trafo' is missing from "
                    "the config file's 'data' section."
                )
            self.samples_time_trafo = preprocessing.compose(
                transformation=config["data"]["samples_time_trafo"],
            )
        else:
            self.samples_time_trafo = None

        self.file_path, showers, self.data_shape = self.__get_data(config, fit_stop)
        showers = torch.from_numpy(showers)

        mask = showers[:, :, 3] > 0.0

        self.samples_coordinate_trafo.to(showers.dtype)
        self.samples_energy_trafo.to(showers.dtype)

        self.samples_coordinate_trafo.fit(
            x=showers[:, :, :2],
            mask=mask[:, :, None].repeat(1, 1, 2),
        )
        self.samples_energy_trafo.fit(
            x=showers[:, :, 3],
            mask=mask,
        )

        if self.with_time:
            self.samples_time_trafo.to(showers.dtype)
            self.samples_time_trafo.fit(
                x=showers[:, :, 4],
                mask=mask,
            )

        layer = (showers[:, :, 2] + 0.5).to(torch.int64)
        self.num_layers = int(torch.max(layer).item() + 1)

    def __get_data(
        self, config: dict[str, Any], fit_stop: int
    ) -> tuple[str, npt.NDArray[np.float32], tuple[int, ...]]:
        data_shape = showerdata.get_file_shape(config["data"]["path"])
        showers = showerdata.load(
            path=config["data"]["path"],
            stop=fit_stop,
        )
        num_cols = 5 if self.with_time else 4
        return config["data"]["path"], showers.points[:, :, :num_cols], data_shape

    def __call__(
        self,
        x: npt.NDArray[np.float32],
    ) -> batch_type:
        x_tensor = torch.from_numpy(x)

        mask = x_tensor[:, 3] > 0.0

        x_tensor[:, :2] = self.samples_coordinate_trafo(
            x_tensor[:, :2].permute(0, 2, 1)
        ).permute(0, 2, 1)

        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])

        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)

        if self.with_time:
            x_tensor[:, 4] = self.samples_time_trafo(x_tensor[:, 4])
            x_tensor = x_tensor[:, [0, 1, 3, 4]]
        else:
            x_tensor = x_tensor[:, [0, 1, 3]]

        return x_tensor.numpy(), mask.numpy(), layer.numpy()


# ======================================================================= #
#  DataLoader                                                              #
# ======================================================================= #

class DataLoader(Iterable[npt.NDArray[np.float32]]):
    def __init__(self, data_file: str, batch_size: int,
                 start_idx: int = 0, stop_idx: int | None = None) -> None:
        self.file_name = data_file
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.stop_idx = stop_idx

    def __iter__(self) -> Iterator[npt.NDArray[np.float32]]:
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            stop = self.stop_idx if self.stop_idx is not None else len(file)
            stop = min(stop, len(file))
            for batch_start in range(self.start_idx, stop, self.batch_size):
                batch_end = min(batch_start + self.batch_size, stop)
                samples = file[batch_start:batch_end].points
                yield samples.transpose(0, 2, 1)


# ======================================================================= #
#  NoiseMatcher                                                            #
# ======================================================================= #

class NoiseMatcher:
    def __init__(self, pre_processor: PreProcessor) -> None:
        self.__num_layers = pre_processor.num_layers
        self.__num_features = pre_processor.num_features
        self.pre_processor = pre_processor

    def __call__(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        points, mask, layer = self.pre_processor(samples)
        F = self.__num_features
        noise = np.random.randn(points.shape[0], F, points.shape[2])

        for i in range(self.__num_layers):
            mask_local = np.expand_dims(np.logical_and(mask, layer == i), 1)
            for j in range(len(points)):
                points_j = (
                    points[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                    .reshape(-1, F)
                )
                noise_j = (
                    noise[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                    .reshape(-1, F)
                )
                if len(points_j) > 1:
                    N = len(points_j)
                    assert len(noise_j) == N
                    M = np.sqrt(
                        np.sum(
                            (points_j[:, None, :] - noise_j[None, :, :]) ** 2, axis=-1
                        )
                    )
                    wa = np.ones(N) / N
                    wb = np.ones(N) / N
                    T = ot.emd(wa, wb, M, numItermax=1_000_000)
                    noise_j = N * (T @ noise_j)
                    noise[j].T[mask_local[j].repeat(F).reshape(-1, F)] = (
                        noise_j.flatten()
                    )

        noise[(~mask[:, None, :]).repeat(F, axis=1)] = 0.0
        return noise.astype(np.float32, copy=False)


# ======================================================================= #
#  Worker pool helpers  (global state avoids pickling NoiseMatcher)        #
# ======================================================================= #

_noise_matcher: NoiseMatcher | None = None


def _init_worker(nm: NoiseMatcher) -> None:
    global _noise_matcher
    _noise_matcher = nm


def _worker_fn(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return _noise_matcher(samples)


def _make_pool(num_processes: int, noise_matcher: NoiseMatcher) -> multiprocessing.Pool:
    return multiprocessing.Pool(
        num_processes,
        initializer=_init_worker,
        initargs=(noise_matcher,),
    )


# ======================================================================= #
#  Pickle helpers                                                          #
# ======================================================================= #

def save_preprocessor(pre_processor: PreProcessor, pkl_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(pre_processor, f)
    print_time(f"PreProcessor saved to {pkl_path}")


def load_preprocessor(pkl_path: str) -> PreProcessor:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Pickle not found: {pkl_path}\n"
            "Run Phase 0 (--fit-only) first."
        )
    with open(pkl_path, "rb") as f:
        pre_processor = pickle.load(f)
    print_time(f"PreProcessor loaded from {pkl_path}")
    return pre_processor


# ======================================================================= #
#  Helpers                                                                 #
# ======================================================================= #

def _chunk_dir(data_file: str, output_dir: str | None) -> str:
    if output_dir is not None:
        d = output_dir
    else:
        base = os.path.splitext(data_file)[0]
        d = base + "_chunks"
    os.makedirs(d, exist_ok=True)
    return d


def _chunk_path(chunk_dir: str, chunk_index: int) -> str:
    return os.path.join(chunk_dir, f"chunk_{chunk_index:06d}.h5")


def _num_chunks(total: int, chunk_size: int) -> int:
    return -(-total // chunk_size)


# ======================================================================= #
#  Core processing routines                                                #
# ======================================================================= #

def process_file(
    data_file: str,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    batch_size: int = 128,
) -> None:
    """Original whole-file processing (no --big-file flag)."""
    F = pre_processor.num_features
    num_batches = -(-data_shape[0] // batch_size)
    print_time("batch size:", batch_size)
    print_time("number of batches:", num_batches)
    print_time(f"num features: {F}  {'(x, y, e, t)' if F == 4 else '(x, y, e)'}")
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    noise = np.empty((data_shape[0], F, data_shape[1]), dtype=np.float32)
    print_time(f"NoiseMatcher initialized. (noise shape={noise.shape})")
    sys.stdout.flush()

    num_processes = n - 1 if (n := os.cpu_count()) else 1
    with _make_pool(num_processes, noise_matcher) as pool:
        for i, batch in enumerate(
            pool.imap(_worker_fn, DataLoader(data_file, batch_size))
        ):
            noise[i * batch_size : i * batch_size + len(batch)] = batch

    print_time("All batches processed.")
    sys.stdout.flush()

    noise = noise.transpose(0, 2, 1)
    showerdata.save_target(noise, data_file, overwrite=True)
    print_time(f"Noise saved successfully to {data_file} (shape={noise.shape}).")
    sys.stdout.flush()


def process_chunk(
    data_file: str,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    chunk_index: int,
    chunk_size: int,
    chunk_dir: str,
    batch_size: int = 128,
) -> None:
    """Phase 1: process one chunk and save noise to a chunk HDF5 file."""
    F = pre_processor.num_features
    total = data_shape[0]
    num_points = data_shape[1]

    global_start = chunk_index * chunk_size
    global_stop = min(global_start + chunk_size, total)
    chunk_len = global_stop - global_start

    if global_start >= total:
        print_time(f"Chunk {chunk_index}: start={global_start} >= total={total}. Nothing to do.")
        return

    print_time(f"Chunk {chunk_index}: samples [{global_start}, {global_stop}) ({chunk_len} samples)")
    print_time(f"num features: {F}  {'(x, y, e, t)' if F == 4 else '(x, y, e)'}")
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    noise = np.empty((chunk_len, F, num_points), dtype=np.float32)
    print_time(f"NoiseMatcher initialized. (chunk noise shape={noise.shape})")
    sys.stdout.flush()

    loader = DataLoader(data_file, batch_size,
                        start_idx=global_start, stop_idx=global_stop)

    num_processes = n - 1 if (n := os.cpu_count()) else 1
    offset = 0
    with _make_pool(num_processes, noise_matcher) as pool:
        for batch in pool.imap(_worker_fn, loader):
            noise[offset : offset + len(batch)] = batch
            offset += len(batch)

    print_time(f"Chunk {chunk_index}: all batches processed.")
    sys.stdout.flush()

    noise = noise.transpose(0, 2, 1)

    out_path = _chunk_path(chunk_dir, chunk_index)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("noise", data=noise, compression="gzip", compression_opts=4)
        hf.attrs["global_start"] = global_start
        hf.attrs["global_stop"] = global_stop
        hf.attrs["chunk_index"] = chunk_index

    print_time(f"Chunk {chunk_index}: noise saved to {out_path} (shape={noise.shape}).")
    sys.stdout.flush()


def merge_chunks(
    data_file: str,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    chunk_size: int,
    chunk_dir: str,
    pkl_path: str | None = None,
) -> None:
    """Phase 2: assemble all chunk HDF5 files and write noise into the data file."""
    F = pre_processor.num_features
    total = data_shape[0]
    num_points = data_shape[1]
    num_chunks = _num_chunks(total, chunk_size)

    print_time(f"Merging {num_chunks} chunks into {data_file}")
    print_time(f"Expected full noise shape: ({total}, {num_points}, {F})")
    sys.stdout.flush()

    noise_full = np.empty((total, num_points, F), dtype=np.float32)

    missing = []
    for i in range(num_chunks):
        path = _chunk_path(chunk_dir, i)
        if not os.path.exists(path):
            missing.append(i)
            continue

        with h5py.File(path, "r") as hf:
            g_start = int(hf.attrs["global_start"])
            g_stop = int(hf.attrs["global_stop"])
            chunk_noise = hf["noise"][:]

        noise_full[g_start:g_stop] = chunk_noise
        print_time(f"  Loaded chunk {i}: [{g_start}, {g_stop})")
        sys.stdout.flush()

    if missing:
        raise RuntimeError(
            f"Missing chunk files for indices: {missing}. "
            f"Cannot merge. Re-run Phase 1 for these indices."
        )

    print_time("All chunks loaded. Writing to data file …")
    sys.stdout.flush()

    showerdata.save_target(noise_full, data_file, overwrite=True)
    print_time(f"Noise written to {data_file} (shape={noise_full.shape}).")
    sys.stdout.flush()

    # Clean up chunk files and directory
    for i in range(num_chunks):
        path = _chunk_path(chunk_dir, i)
        if os.path.exists(path):
            os.remove(path)
    try:
        os.rmdir(chunk_dir)
        print_time(f"Removed chunk directory: {chunk_dir}")
    except OSError:
        print_time(f"Chunk directory not empty after cleanup (manual check needed): {chunk_dir}")

    # Remove the pickle now that it's no longer needed
    if pkl_path and os.path.exists(pkl_path):
        os.remove(pkl_path)
        print_time(f"Removed pickle: {pkl_path}")

    sys.stdout.flush()


# ======================================================================= #
#  Entry point                                                             #
# ======================================================================= #

@torch.inference_mode()
def main(args: list[str] | None = None):
    torch.set_num_threads(1)

    parsed_args = parse_args(args)
    print_time("Parsing arguments:", parsed_args)
    print_time(
        f"Mode: {'with time (x, y, e, t)' if parsed_args.with_time else 'original (x, y, e)'}"
    )
    sys.stdout.flush()

    # ------------------------------------------------------------------ #
    #  Dispatch                                                           #
    # ------------------------------------------------------------------ #

    if parsed_args.fit_only:
        # ---- Phase 0: fit and save pickle -----------------------------
        print_time("[PHASE 0] Fitting transformations …")
        pre_processor = PreProcessor(
            parsed_args.file,
            with_time=parsed_args.with_time,
            fit_stop=50000,
        )
        print_time(f"PreProcessor fitted. num_layers={pre_processor.num_layers}, "
                   f"data_shape={pre_processor.data_shape}")
        save_preprocessor(pre_processor, parsed_args.pkl_path)

    elif parsed_args.big_file:
        # ---- Phase 1: one array-job task ------------------------------
        print_time("[PHASE 1] Loading preprocessor from pickle …")
        pre_processor = load_preprocessor(parsed_args.pkl_path)
        chunk_dir = _chunk_dir(pre_processor.file_path, parsed_args.output_dir)
        print_time(
            f"chunk_index={parsed_args.chunk_index}, "
            f"chunk_size={parsed_args.chunk_size}, chunk_dir={chunk_dir}"
        )
        process_chunk(
            data_file=pre_processor.file_path,
            data_shape=pre_processor.data_shape,
            pre_processor=pre_processor,
            chunk_index=parsed_args.chunk_index,
            chunk_size=parsed_args.chunk_size,
            chunk_dir=chunk_dir,
            batch_size=128,
        )

    elif parsed_args.merge:
        # ---- Phase 2: merge job ---------------------------------------
        print_time("[PHASE 2] Loading preprocessor from pickle …")
        pre_processor = load_preprocessor(parsed_args.pkl_path)
        chunk_dir = _chunk_dir(pre_processor.file_path, parsed_args.output_dir)
        print_time(f"chunk_dir={chunk_dir}, chunk_size={parsed_args.chunk_size}")
        merge_chunks(
            data_file=pre_processor.file_path,
            data_shape=pre_processor.data_shape,
            pre_processor=pre_processor,
            chunk_size=parsed_args.chunk_size,
            chunk_dir=chunk_dir,
            pkl_path=parsed_args.pkl_path,
        )

    else:
        # ---- Original whole-file mode (unchanged) ---------------------
        print_time("[ORIGINAL MODE] Fitting and processing …")
        pre_processor = PreProcessor(
            parsed_args.file,
            with_time=parsed_args.with_time,
        )
        print_time("PreProcessor initialized.")
        sys.stdout.flush()
        process_file(
            data_file=pre_processor.file_path,
            data_shape=pre_processor.data_shape,
            pre_processor=pre_processor,
            batch_size=128,
        )


if __name__ == "__main__":
    main()