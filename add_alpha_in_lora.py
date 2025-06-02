#!/usr/bin/env python3

import argparse
import json
from math import sqrt

import safetensors.torch
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--lora_config_path", type=str)
    args = parser.parse_args()

    with open(args.lora_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    rank = config["r"]
    alpha = config["lora_alpha"]
    if config["use_rslora"]:
        alpha /= sqrt(rank)
    else:
        alpha /= rank
    print("alpha", alpha)

    tensors = safetensors.torch.load_file(args.input_name)
    # Copy a list of keys to avoid modifying the iterator
    ks = list(tensors.keys())
    for k in ks:
        if not k.endswith(".lora_A.weight"):
            continue
        k_alpha = k.replace(".lora_A.weight", ".alpha")
        if k_alpha in ks:
            tensors[k_alpha] *= alpha
        else:
            dtype = tensors[k].dtype
            tensors[k_alpha] = torch.tensor(alpha, dtype=dtype)
    safetensors.torch.save_file(tensors, args.output_name)


if __name__ == "__main__":
    main()
