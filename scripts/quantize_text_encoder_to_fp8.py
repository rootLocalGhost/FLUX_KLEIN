#!/usr/bin/env python3
"""Quantize FLUX.2 text encoder checkpoint to FP8 in place-friendly paths.

Will read from: model/FLUX.2-klein-4B-FP16/text_encoder/text-encoder-merged.safetensors (or shards with index)
Write to: model/FLUX.2-klein-4B-FP16/text_encoder_quantized/text-encoder-fp8.safetensors

    python quantize_text_encoder_fp8.py
"""

import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_DIR = ROOT_DIR / "models" / "FLUX.2-klein-4B-FP16"
TEXT_ENCODER_DIR = MODEL_DIR / "text_encoder"
TEXT_ENCODER_MERGED_DIR = TEXT_ENCODER_DIR
TEXT_ENCODER_QUANT_DIR = MODEL_DIR / "text_encoder_quantized"
OUTPUT_FILE = TEXT_ENCODER_QUANT_DIR / "text-encoder-fp8.safetensors"
OUTPUT_INDEX_FILE = TEXT_ENCODER_QUANT_DIR / "model.safetensors.index.json"


def quantize_to_fp8(dtype_name="float8_e4m3fn", verbose=True):
    TEXT_ENCODER_QUANT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_ENCODER_MERGED_DIR.mkdir(parents=True, exist_ok=True)

    if dtype_name == "float8_e4m3fn":
        fp8_dtype = torch.float8_e4m3fn
    else:
        fp8_dtype = torch.float8_e5m2

    source_dir = TEXT_ENCODER_DIR
    source_paths = []

    if (TEXT_ENCODER_MERGED_DIR / "text-encoder-merged.safetensors").exists():
        source_paths = [TEXT_ENCODER_MERGED_DIR / "text-encoder-merged.safetensors"]
        source_dir = TEXT_ENCODER_MERGED_DIR
        if verbose:
            print(f"Using merged text encoder source: {source_paths[0]}")
    else:
        index_path = TEXT_ENCODER_DIR / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No source model found in {TEXT_ENCODER_DIR}")

        with index_path.open("r", encoding="utf-8") as f:
            idx = json.load(f)

        if "weight_map" not in idx:
            raise ValueError("Index missing weight_map")

        source_paths = sorted(
            {TEXT_ENCODER_DIR / path for path in idx["weight_map"].values()}
        )
        if verbose:
            print(f"Using shard sources: {source_paths}")

    merged = {}

    for path in source_paths:
        if verbose:
            print(f"Loading {path}")
        shard = load_file(str(path))

        for name, tensor in shard.items():
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)

            if tensor.dtype != fp8_dtype:
                tensor = tensor.to(dtype=fp8_dtype, device="cpu")

            merged[name] = tensor

        del shard
        gc.collect()

        if verbose:
            print(f"  -> merged tensors: {len(merged)}")

    if verbose:
        print(f"Saving quantized checkpoint to {OUTPUT_FILE}")

    save_file(merged, str(OUTPUT_FILE))

    index = {
        "metadata": {
            "total_parameters": sum(t.numel() for t in merged.values()),
            "total_size": sum(t.numel() * t.element_size() for t in merged.values()),
        },
        "weight_map": {name: OUTPUT_FILE.name for name in merged.keys()},
    }

    with OUTPUT_INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # Copy config and index from main text_encoder to merged and quant dirs
    for target_dir in [TEXT_ENCODER_MERGED_DIR, TEXT_ENCODER_QUANT_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        for cfg in [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ]:
            src_cfg = TEXT_ENCODER_DIR / cfg
            if src_cfg.exists():
                shutil.copy2(src_cfg, target_dir / cfg)

    # Update copied merged index (if from base) to point to merged model if needed
    merged_index_file = TEXT_ENCODER_MERGED_DIR / "model.safetensors.index.json"
    if merged_index_file.exists():
        with merged_index_file.open("r", encoding="utf-8") as f:
            merged_idx = json.load(f)
        merged_idx["weight_map"] = {name: "model.safetensors" for name in merged.keys()}
        with merged_index_file.open("w", encoding="utf-8") as f:
            json.dump(merged_idx, f, indent=2)

    # Update quant index to point to quant file
    quant_index_file = TEXT_ENCODER_QUANT_DIR / "model.safetensors.index.json"
    with quant_index_file.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"Wrote index to {OUTPUT_INDEX_FILE}")
        print("Quantization complete")


if __name__ == "__main__":
    quantize_to_fp8(dtype_name="float8_e4m3fn", verbose=True)
