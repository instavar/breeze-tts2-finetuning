from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from training.lora import inject_lora, load_adapter, merge_lora
from training.lora_study import loss_receipt
from training.model_loading import load_tokenizer, load_training_model
from training.real_data import sha256_file
from training.real_lora import move_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freshly reload and merge a completed real Breeze LoRA"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verification-output", type=Path)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-examples", type=int, default=16)
    return parser.parse_args()


def average_loss(
    model: torch.nn.Module, paths: list[Path], device: str
) -> dict[str, float]:
    sums = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
    model.eval()
    with torch.no_grad():
        for path in paths:
            losses = loss_receipt(
                model(**move_example(path, device), use_cache=False, return_dict=True)
            )
            for name, value in losses.items():
                sums[name] += value
    return {name: value / len(paths) for name, value in sums.items()}


def main() -> int:
    args = parse_args()
    receipt = json.loads((args.training_root / "training-receipt.json").read_text())
    configuration = receipt["configuration"]
    checkpoint = args.checkpoint or Path(receipt["final_checkpoint"])
    checkpoint_receipt = json.loads(
        (checkpoint / "checkpoint-receipt.json").read_text()
    )
    adapter_path = checkpoint / "adapter.safetensors"
    if sha256_file(adapter_path) != checkpoint_receipt["roles"]["adapter.safetensors"]:
        raise RuntimeError("final adapter file hash differs from checkpoint receipt")

    cache = json.loads((args.cache_root / "cache-receipt.json").read_text())
    paths = [
        args.cache_root / row["artifact"]
        for row in cache["splits"]["validation"][: args.validation_examples]
    ]
    if not paths:
        raise ValueError("no validation examples selected")

    torch.cuda.set_device(args.device)
    model = load_training_model(args.model_root, device=args.device)
    families = inject_lora(
        model,
        variant=configuration["variant"],
        rank=configuration["rank"],
        alpha=configuration["alpha"],
        seed=configuration["seed"],
    )
    hashes = load_adapter(model, adapter_path)
    if hashes != checkpoint_receipt["adapter_tensor_hashes"]:
        raise RuntimeError("fresh-process adapter tensor hashes differ")
    adapter_loss = average_loss(model, paths, args.device)
    merged_count = merge_lora(model)
    if merged_count != len(families):
        raise RuntimeError("merged module count differs from target count")
    merged_loss = average_loss(model, paths, args.device)
    for name in adapter_loss:
        if not math.isclose(adapter_loss[name], merged_loss[name], abs_tol=0.02):
            raise RuntimeError(
                f"merged {name} differs: adapter={adapter_loss[name]} "
                f"merged={merged_loss[name]}"
            )

    merged_partial = args.merged_output.with_name(args.merged_output.name + ".partial")
    if args.merged_output.exists() or merged_partial.exists():
        raise FileExistsError(
            f"refusing to overwrite merged model package: {args.merged_output}"
        )
    merged_partial.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        merged_partial,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    load_tokenizer(args.model_root).save_pretrained(merged_partial)
    shutil.copytree(
        args.model_root / "audio_tokenizer",
        merged_partial / "audio_tokenizer",
    )
    for name in ("LICENSE", "README.md"):
        source = args.model_root / name
        if source.is_file():
            shutil.copy2(source, merged_partial / name)
    merged_partial.rename(args.merged_output)

    verify_receipt = {
        "schema_version": 1,
        "status": "fresh_process_adapter_reload_and_merge_verified",
        "checkpoint": str(checkpoint),
        "adapter_file_sha256": sha256_file(adapter_path),
        "adapter_tensor_hashes_matched": True,
        "merged_module_count": merged_count,
        "merged_model": str(args.merged_output),
        "merged_model_index_sha256": sha256_file(
            args.merged_output / "model.safetensors.index.json"
        ),
        "validation_examples": len(paths),
        "loss": {"adapter": adapter_loss, "merged": merged_loss},
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
    }
    path = args.verification_output or args.training_root / "verification-receipt.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification receipt: {path}")
    path.write_text(json.dumps(verify_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
