#!/usr/bin/env python3
"""Merge a shard-split text_encoder safetensors checkpoint into a single file.

Usage:
  python convert_text_encoder_safetensors_single.py

Requirements:
  pip install safetensors

This script reads text_encoder/model.safetensors.index.json and all referenced
model-*.safetensors shard files, merges tensors and writes text_encoder/text-encoder-merged.safetensors.
It can also update the index to point at the new single file.
"""

import gc
import json
import shutil
from pathlib import Path

# safetensors backends
try:
    from safetensors.torch import (
        load_file as torch_load_file,
        save_file as torch_save_file,
    )
except ImportError:
    torch_load_file = torch_save_file = None

try:
    from safetensors.numpy import (
        load_file as numpy_load_file,
        save_file as numpy_save_file,
    )
except ImportError:
    numpy_load_file = numpy_save_file = None

if torch_load_file is None and numpy_load_file is None:
    raise RuntimeError(
        "Please install safetensors package with either torch or numpy backend: pip install safetensors"
    )


def main():
    root_dir = Path(__file__).resolve().parent.parent
    model_dir = root_dir / "model" / "FLUX.2-klein-4B"
    src_dir = model_dir / "text_encoder"
    merged_dir = model_dir / "text_encoder_merged"
    quant_dir = model_dir / "text_encoder_quantized"

    merged_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)

    if "weight_map" not in idx:
        raise ValueError("Index JSON does not contain weight_map")

    weight_map = idx["weight_map"]
    out_file = merged_dir / "text-encoder-merged.safetensors"
    out_index = merged_dir / "model.safetensors.index.json"

    shard_files = sorted({(src_dir / p).resolve() for p in weight_map.values()})
    shard_files = [p for p in shard_files if p != out_file.resolve()]

    print(f"Found {len(shard_files)} shard files")

    merged = {}
    loaded = 0
    for shard_path in shard_files:
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        print(f"Loading shard: {shard_path}")

        shard_tensors = None
        if torch_load_file is not None:
            try:
                shard_tensors = torch_load_file(str(shard_path))
            except Exception as exc:
                print(f"Torch loader failed for {shard_path}: {exc}")
                shard_tensors = None

        if (
            shard_tensors is None
            and "numpy_load_file" in globals()
            and globals()["numpy_load_file"] is not None
        ):
            try:
                shard_tensors = globals()["numpy_load_file"](str(shard_path))
                if torch_load_file is not None:
                    import torch

                    shard_tensors = {
                        k: torch.from_numpy(v) for k, v in shard_tensors.items()
                    }
            except Exception as exc:
                raise RuntimeError(f"Failed to load shard {shard_path}: {exc}") from exc

        if shard_tensors is None:
            raise RuntimeError(
                f"No available safetensors loader succeeded on {shard_path}"
            )

        overlap = set(merged.keys()) & set(shard_tensors.keys())
        if overlap:
            raise ValueError(
                f"Duplicate tensor keys found when merging shards: {overlap}"
            )

        merged.update(shard_tensors)
        loaded += len(shard_tensors)

        del shard_tensors
        gc.collect()

    print(f"Total tensors merged: {loaded}")
    print(f"Writing output file: {out_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_out_file = out_file.with_suffix(out_file.suffix + ".tmp")

    if torch_save_file is not None:
        torch_save_file(merged, str(temp_out_file))
    elif "numpy_save_file" in globals() and globals()["numpy_save_file"] is not None:
        import numpy as np

        numpy_payload = {}
        for k, v in merged.items():
            if hasattr(v, "cpu"):
                v = v.cpu()
            if hasattr(v, "numpy"):
                v = v.numpy()
            numpy_payload[k] = np.asarray(v)
        globals()["numpy_save_file"](numpy_payload, str(temp_out_file))
    else:
        raise RuntimeError("No safetensors save backend is available")

    temp_out_file.replace(out_file)

    # Copy config files and index to merged and quant directories
    for target_dir in [merged_dir, quant_dir]:
        for cfg in [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ]:
            src = src_dir / cfg
            if src.exists():
                shutil.copy2(src, target_dir / cfg)

    # Update the merged index in merged directory
    updated_idx = idx.copy()
    updated_idx["weight_map"] = {name: out_file.name for name in merged.keys()}
    updated_idx["metadata"] = updated_idx.get("metadata", {})

    with out_index.open("w", encoding="utf-8") as f:
        json.dump(updated_idx, f, indent=2, ensure_ascii=False)

    # Also update quant index from the merged index just-in-case
    quant_index_file = quant_dir / "model.safetensors.index.json"
    shutil.copy2(out_index, quant_index_file)

    print("Merge completed successfully.")


if __name__ == "__main__":
    main()
