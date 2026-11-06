from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import torch
from fairchem.core.common.utils import match_state_dict
from fairchem.core.models.escn.escn import eSCN


def model_config(checkpoint: dict) -> dict:
    config = checkpoint["config"]
    if isinstance(config["model"], str):
        model = copy.deepcopy(config["model_attributes"])
        model["name"] = config["model"]
    else:
        model = copy.deepcopy(config["model"])
    if model.pop("name") != "escn":
        raise ValueError("Expected an eSCN checkpoint.")
    if len(model.get("lmax_list", [6])) != 1 or len(model.get("mmax_list", [2])) != 1:
        raise ValueError("Only single-resolution eSCN checkpoints are supported.")
    return model


def prune_state(source: dict, target: dict, source_l: int, target_l: int,
                channels: int) -> dict:
    source = match_state_dict(target, source)
    if source.keys() != target.keys():
        raise ValueError("Source and target parameter names do not match.")
    result = {}
    for key, dst in target.items():
        src = source[key]
        if src.shape == dst.shape:
            result[key] = src.clone()
            continue
        match = re.search(r"\.so2_block_(?:source|target)\.(?:so2_conv\.(\d+)\.)?fc([12])_(m0|r|i)\.weight$", key)
        if match:
            m = 0 if match[1] is None else int(match[1]) + 1
            axis = 1 if match[2] == "1" else 0
            old_size = (source_l - m + 1) * channels
            new_size = (target_l - m + 1) * channels
        elif key == "sphharm_weights.0":
            axis = 1
            old_size = (source_l + 1) ** 2
            new_size = (target_l + 1) ** 2
        else:
            raise ValueError(f"Unexpected shape change for {key}: {src.shape} -> {dst.shape}")
        expected = list(src.shape)
        expected[axis] = new_size
        if src.shape[axis] != old_size or tuple(expected) != tuple(dst.shape):
            raise ValueError(f"Unexpected parameter layout for {key}.")
        # Each fixed-m block stores increasing l, with channels contiguous within l.
        result[key] = src.narrow(axis, 0, new_size).clone()
    return result


def prune(model_path: str, target_l: int, output: str) -> None:
    print(f"Loading: {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = model_config(checkpoint)
    source_l = config.get("lmax_list", [6])[0]
    mmax = config.get("mmax_list", [2])[0]
    if not mmax <= target_l < source_l:
        raise ValueError(f"target-l must satisfy {mmax} <= target-l < {source_l}.")
    config["lmax_list"] = [target_l]
    print(f"Pruning: L{source_l} -> L{target_l} (M{mmax}, channels unchanged)", flush=True)
    model = eSCN(**config)
    state = prune_state(checkpoint["state_dict"], model.state_dict(), source_l,
                        target_l, config.get("sphere_channels", 128))
    model.load_state_dict(state, strict=True)
    checkpoint["state_dict"] = state
    checkpoint["config"]["model"] = {"name": "escn", **config}
    checkpoint["config"].pop("model_attributes", None)
    for key in ("optimizer", "scheduler", "amp", "ema", "val_metrics"):
        checkpoint.pop(key, None)
    checkpoint.update(epoch=0, step=0, best_val_metric=None)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce lmax of a single-resolution eSCN checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-l", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prune(args.model, args.target_l, args.output)
