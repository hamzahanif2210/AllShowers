"""
Check whether any tokens end up with empty attention windows
under a given num_layer_cond setting.

Usage:
python /n/home04/hhanif/AllShowers/allshowers/check_empty_windows.py /n/home04/hhanif/AllShowers/conf/electrons.yaml --num-layer-cond 4 --trafos-file /n/home04/hhanif/AllShowers/results/20260519_014007_Electron-Allshower/preprocessing/trafos.pt
"""

import argparse
import os
import sys

import torch
import yaml


def check_empty_windows(
    layer: torch.Tensor,   # [B, N, 1] int32
    mask: torch.Tensor,    # [B, N, 1] bool
    num_layer_cond: int,
) -> dict:
    B, N, _ = layer.shape
    layer_flat = layer.flatten(1)  # [B, N]
    mask_flat  = mask.flatten(1)   # [B, N]
    half = num_layer_cond // 2

    empty_tokens  = 0
    total_tokens  = 0
    empty_batches = 0
    examples      = []

    for b in range(B):
        real     = mask_flat[b]
        lay      = layer_flat[b]
        real_idx = real.nonzero(as_tuple=True)[0]

        batch_has_empty = False
        for qi in real_idx:
            delta     = lay[qi] - lay[real_idx]
            in_window = (delta >= -half) & (delta <= half)
            neighbors = in_window.sum().item() - 1  # exclude self
            total_tokens += 1
            if neighbors == 0:
                empty_tokens += 1
                batch_has_empty = True
                if len(examples) < 10:
                    examples.append((b, qi.item(), lay[qi].item(), neighbors))

        if batch_has_empty:
            empty_batches += 1

    return {
        "total_tokens":  total_tokens,
        "empty_tokens":  empty_tokens,
        "empty_batches": empty_batches,
        "total_batches": B,
        "empty_pct":     100 * empty_tokens / max(total_tokens, 1),
        "examples":      examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to your yaml config file")
    parser.add_argument("--num-layer-cond", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--trafos-file", type=str, default=None,
        help="path to an existing trafos.pt from a previous run of this config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    num_layer_cond = args.num_layer_cond
    num_layers     = conf["model"].get("num_layers", 24)
    batch_size     = args.batch_size or conf["train"]["batch_size"]

    # resolve trafos file
    trafos_file = args.trafos_file
    if trafos_file is None:
        result_path = conf.get("result_path", "")
        for candidate in [
            os.path.join(result_path, "preprocessing/trafos.pt"),
            os.path.join(result_path, "preprocessing/trafos-all.pt"),
        ]:
            if os.path.isfile(candidate):
                trafos_file = candidate
                break
    if trafos_file is None:
        print("ERROR: could not find a trafos file automatically.")
        print("Pass one with --trafos-file /path/to/preprocessing/trafos.pt")
        sys.exit(1)

    print(f"num_layer_cond : {num_layer_cond}  (window +/-{num_layer_cond // 2} layers)")
    print(f"num_layers     : {num_layers}")
    print(f"batch_size     : {batch_size}")
    print(f"trafos_file    : {trafos_file}")
    print(f"checking       : {args.num_batches} batches")
    print()

    sys.path.insert(0, ".")
    from allshowers import data_sets

    conf["data"]["num_layers"] = num_layers
    conf["data"]["val_len"]    = batch_size * args.num_batches

    _, val_loader, _ = data_sets.get_data_loaders(
        conf["data"],
        batch_size,
        rank=0,
        world_size=1,
        local_rank=0,
        trafos_file=trafos_file,
    )

    total_tokens  = 0
    empty_tokens  = 0
    empty_batches = 0
    total_batches = 0
    all_examples  = []

    for i, batch in enumerate(val_loader):
        if i >= args.num_batches:
            break

        stats = check_empty_windows(batch["layer"], batch["mask"], num_layer_cond)

        total_tokens  += stats["total_tokens"]
        empty_tokens  += stats["empty_tokens"]
        empty_batches += stats["empty_batches"]
        total_batches += stats["total_batches"]
        all_examples.extend(stats["examples"])

        print(f"batch {i:3d} | tokens={stats['total_tokens']:6d} | "
              f"empty={stats['empty_tokens']:4d} ({stats['empty_pct']:.1f}%) | "
              f"batches_with_empty={stats['empty_batches']}/{stats['total_batches']}")

    print()
    print("=" * 60)
    print(f"Total tokens checked              : {total_tokens}")
    print(f"Tokens with 0 neighbors in window : "
          f"{empty_tokens} ({100*empty_tokens/max(total_tokens,1):.2f}%)")
    print(f"Batches containing empty tokens   : {empty_batches}/{total_batches}")

    if all_examples:
        print()
        print("Sample empty tokens (batch, token_idx, layer, neighbors):")
        for ex in all_examples[:10]:
            print(f"  batch={ex[0]}, token={ex[1]}, layer={ex[2]}, neighbors={ex[3]}")
    else:
        print()
        print("No empty windows found -- boundary layer issue is NOT present.")
        print("The instability is purely the learning rate; the config fix is sufficient.")


if __name__ == "__main__":
    main()