from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from training.lora_study import loss_receipt
from training.model_loading import load_training_model
from training.real_lora import move_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freshly load the merged Breeze model package"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--verification-receipt", type=Path)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-examples", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verification_path = (
        args.verification_receipt or args.training_root / "verification-receipt.json"
    )
    receipt = json.loads(verification_path.read_text())
    if receipt.get("status") != "fresh_process_adapter_reload_and_merge_verified":
        raise RuntimeError(f"unexpected verification status: {receipt.get('status')}")

    cache = json.loads((args.cache_root / "cache-receipt.json").read_text())
    paths = [
        args.cache_root / row["artifact"]
        for row in cache["splits"]["validation"][: args.validation_examples]
    ]
    if not paths:
        raise ValueError("no validation examples selected")

    torch.cuda.set_device(args.device)
    model = load_training_model(args.merged_model, device=args.device)
    sums = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
    model.eval()
    with torch.no_grad():
        for path in paths:
            losses = loss_receipt(
                model(
                    **move_example(path, args.device), use_cache=False, return_dict=True
                )
            )
            for name, value in losses.items():
                sums[name] += value
    merged_package_loss = {name: value / len(paths) for name, value in sums.items()}
    expected = receipt["loss"]["merged"]
    for name, value in merged_package_loss.items():
        if not math.isclose(value, expected[name], abs_tol=0.02):
            raise RuntimeError(
                f"fresh merged package {name} differs: expected={expected[name]} "
                f"actual={value}"
            )

    receipt["status"] = "fresh_process_adapter_and_merged_package_verified"
    receipt["merged_package_loss"] = merged_package_loss
    receipt["merged_package_validation_examples"] = len(paths)
    receipt["merged_package_peak_cuda_memory_bytes"] = int(
        torch.cuda.max_memory_allocated(args.device)
    )
    verification_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
