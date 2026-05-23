import os
import time
import warnings
from typing import TypedDict

import showerdata
import torch
from torch import Tensor

from allshowers.data_loader import DataLoader, DictDataSet, ModelInputDict
from allshowers.preprocessing import Identity, Transformation, compose

__all__ = ["create_label_list", "to_label_tensor", "get_data_loaders"]


class ShowerDict(TypedDict):
    shower: Tensor
    energy: Tensor
    direction: Tensor
    pdg: Tensor
    noise: Tensor | None


def batched_histogram(
    data: torch.Tensor, mask: torch.Tensor, num_bins: int = -1
) -> torch.Tensor:
    if num_bins < 0:
        num_bins = int(torch.max(data[mask]).item()) + 1
    histograms = torch.zeros(size=(data.shape[0], num_bins), dtype=torch.int32)
    ones = torch.zeros(size=data.shape, dtype=histograms.dtype)
    ones[mask] = 1
    histograms.scatter_add_(1, data, ones)
    return histograms


@torch.no_grad()
def initialise_trafos(
    energies: Tensor,
    showers: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    samples_time_trafo: Transformation | None,
    cond_trafo: Transformation,
    *,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if trafos_file is None and world_size > 1:
        raise ValueError(
            "If using distributed training, a trafos_file must be provided to save and load the transformations."
        )
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
    if rank != 0:
        torch.distributed.barrier(device_ids=[local_rank])
    if os.path.isfile(trafos_file):
        if world_size > 1 and rank == 0:
            torch.distributed.barrier(device_ids=[local_rank])
        parameters = torch.load(trafos_file, weights_only=True)
        samples_energy_trafo.load_state_dict(parameters["samples_energy_trafo"])
        samples_coordinate_trafo.load_state_dict(parameters["samples_coordinate_trafo"])
        if samples_time_trafo is not None and "samples_time_trafo" in parameters:
            samples_time_trafo.load_state_dict(parameters["samples_time_trafo"])
        cond_trafo.load_state_dict(parameters["cond_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        # Caller is responsible for passing data that spans the full training
        # distribution (see fit_start/fit_stop in load_and_prepare).
        print(f"[rank {rank}] Fitting transformations on {len(energies)} showers")
        cond_trafo.fit(energies)
        samples_coordinate_trafo.fit(showers[:, :, :2], mask)
        samples_energy_trafo.fit(showers[:, :, 3], mask.squeeze())
        if samples_time_trafo is not None:
            samples_time_trafo.fit(showers[:, :, 4], mask.squeeze())
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
            }
            if samples_time_trafo is not None:
                parameters["samples_time_trafo"] = samples_time_trafo.state_dict()
            torch.save(parameters, trafos_file)
            print(f"[rank {rank}] Saved transformations to {trafos_file}")
        if world_size > 1:
            time.sleep(5)  # make sure file is on network drive
            torch.distributed.barrier(device_ids=[local_rank])


def load_data(
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    max_num_points: int | None = None,
) -> ShowerDict:
    showers = showerdata.load(
        path,
        start,
        stop,
        max_points=max_num_points,
    )
    if return_noise:
        noise, _ = showerdata.load_target(path, "target", start=start, stop=stop)
    else:
        noise = None
    if showers.points.shape[2] not in (4, 5):
        raise ValueError(
            f"Expected 4 or 5 components (x, y, layer, energy[, time]), "
            f"got {showers.points.shape[2]}"
        )
    data = ShowerDict(
        shower=torch.from_numpy(showers.points),
        energy=torch.from_numpy(showers.energies),
        direction=torch.from_numpy(showers.directions),
        pdg=torch.from_numpy(showers.pdg),
        noise=torch.from_numpy(noise) if noise is not None else None,
    )
    return data


@torch.no_grad()
def create_label_list(
    pdg: torch.Tensor,
) -> list[int]:
    unique_pdg = pdg.unique().tolist()
    unique_pdg.sort(key=lambda x: (abs(x), -x))
    return unique_pdg


@torch.no_grad()
def to_label_tensor(
    pdg: torch.Tensor | None,
    label_list: list[int] | None = None,
) -> torch.Tensor | None:
    if pdg is None:
        return None
    if label_list is None:
        label_list = create_label_list(pdg)
    if max(pdg.shape, default=1) != pdg.numel():
        raise ValueError("pdg must be a 1D tensor.")
    pdg = pdg.view(-1)
    label_tensor = torch.zeros(pdg.shape[0], dtype=torch.int64)
    for i, label in enumerate(label_list):
        label_tensor[pdg == label] = i
    return label_tensor


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    samples_time_trafo: Transformation | None = None,
    cond_trafo: Transformation = Identity(),
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    return_direction: bool = False,
    max_num_points: int | None = None,
    num_layers: int = -1,
    do_initialise_trafos: bool = True,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    # When provided (rank 0 only), load this index range purely for trafo
    # fitting so the scaler sees the full training distribution instead of only
    # rank 0's shard. This fixes non-deterministic convergence caused by the
    # time distribution varying across the dataset ordering.
    fit_start: int | None = None,
    fit_stop: int | None = None,
) -> ModelInputDict:
    data = load_data(
        path,
        start=start,
        stop=stop,
        return_noise=return_noise,
        max_num_points=max_num_points,
    )
    has_time = data["shower"].shape[2] == 5
    # If user supplied a time trafo but data has no time component, ignore it.
    effective_time_trafo = samples_time_trafo if has_time else None

    mask = data["shower"][:, :, [3]] > 0

    if do_initialise_trafos:
        if fit_start is not None and fit_stop is not None and rank == 0:
            # Load the full training split so the scaler is fit on the complete
            # distribution rather than just rank 0's shard.
            print(
                f"[rank {rank}] Loading fit data [{fit_start}, {fit_stop}) "
                f"for trafo initialisation"
            )
            fit_data = load_data(
                path,
                start=fit_start,
                stop=fit_stop,
                return_noise=False,
                max_num_points=max_num_points,
            )
            fit_mask = fit_data["shower"][:, :, [3]] > 0
            initialise_trafos(
                fit_data["energy"],
                fit_data["shower"],
                fit_mask,
                samples_energy_trafo,
                samples_coordinate_trafo,
                effective_time_trafo,
                cond_trafo,
                trafos_file=trafos_file,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
        else:
            initialise_trafos(
                data["energy"],
                data["shower"],
                mask,
                samples_energy_trafo,
                samples_coordinate_trafo,
                effective_time_trafo,
                cond_trafo,
                trafos_file=trafos_file,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )

    energy = cond_trafo(data["energy"])
    features = [
        samples_coordinate_trafo(data["shower"][:, :, :2]),
        samples_energy_trafo(data["shower"][:, :, [3]]),
    ]
    if has_time and samples_time_trafo is not None:
        features.append(samples_time_trafo(data["shower"][:, :, [4]]))
    x = torch.concat(features, dim=-1)
    x[~mask.repeat(1, 1, x.shape[-1])] = 0.0

    layer = (data["shower"][:, :, [2]] + 0.1).long()
    num_points = batched_histogram(
        data=layer.squeeze(dim=-1),
        mask=mask.squeeze(dim=-1),
        num_bins=num_layers,
    )
    label = to_label_tensor(data["pdg"])

    if return_direction:
        cond = torch.concat([energy, data["direction"]], dim=-1)
    else:
        cond = energy

    return ModelInputDict(
        x=x,
        cond=cond,
        num_points=num_points,
        layer=layer,
        mask=mask,
        label=label if label is not None else torch.zeros(0, dtype=torch.int64),
        noise=data["noise"],
    )


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader, DataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    data_len = showerdata.get_file_shape(config_dataset["path"])[0]
    if "stop" in config_dataset:
        data_len = min(data_len, config_dataset["stop"])
        del config_dataset["stop"]
    if "val_len" in config_dataset:
        val_len = config_dataset.pop("val_len")
        if val_len > data_len // 2:
            warnings.warn(
                f"val_len {val_len} is larger than 50% of data length {data_len // 2},"
                f" reducing to {data_len // 2}.",
                UserWarning,
            )
            val_len = min(val_len, data_len // 2)
    else:
        val_len = data_len // 10
    split = data_len - val_len

    print(
        f"[rank {rank}] data_len={data_len}, val_len={val_len}, split={split}, "
        f"per_rank_train={split // world_size}, world_size={world_size}"
    )

    if "samples_energy_trafo" in config_dataset:
        config_dataset["samples_energy_trafo"] = compose(
            config_dataset["samples_energy_trafo"]
        )
    if "samples_coordinate_trafo" in config_dataset:
        config_dataset["samples_coordinate_trafo"] = compose(
            config_dataset["samples_coordinate_trafo"]
        )
    if "samples_time_trafo" in config_dataset:
        config_dataset["samples_time_trafo"] = compose(
            config_dataset["samples_time_trafo"]
        )
    if "cond_trafo" in config_dataset:
        config_dataset["cond_trafo"] = compose(config_dataset["cond_trafo"])

    start = rank * (split // world_size)
    stop = (rank + 1) * (split // world_size)

    # Rank 0 fits trafos on the full training split (indices 0 to split) so the
    # scaler sees the complete time distribution rather than only rank 0's shard.
    # This is the fix for non-deterministic convergence when the time distribution
    # varies across the dataset ordering.
    data_train = DictDataSet(
        load_and_prepare(
            **config_dataset,
            start=start,
            stop=stop,
            trafos_file=trafos_file,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            fit_start=0 if rank == 0 else None,
            fit_stop=split if rank == 0 else None,
        )
    )
    loader_train = DataLoader(
        data_set=data_train,
        batch_size=batch_size,
        drop_last=(stop - start) > batch_size,
        shuffle=True,
    )
    if rank == 0:
        data_test = DictDataSet(
            load_and_prepare(
                **config_dataset,
                start=split,
                stop=data_len,
                trafos_file=trafos_file,
                do_initialise_trafos=False,
            )
        )
        loader_test = DataLoader(
            data_set=data_test, batch_size=batch_size, drop_last=False, shuffle=False
        )
    else:
        loader_test = DataLoader(
            data_set=DictDataSet(
                ModelInputDict(
                    x=torch.empty(0, 0, 0),
                    cond=torch.empty(0, 0),
                    num_points=torch.empty(0, 0, dtype=torch.int64),
                    layer=torch.empty(0, 0, dtype=torch.int64),
                    mask=torch.empty(0, 0, dtype=torch.bool),
                    label=torch.empty(0, 0, dtype=torch.int64),
                    noise=None,
                )
            ),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
    }
    if "samples_time_trafo" in config_dataset:
        trafos["samples_time_trafo"] = config_dataset["samples_time_trafo"]
    return loader_train, loader_test, trafos