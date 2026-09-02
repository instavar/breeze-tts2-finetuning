from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from training.lora import inject_lora, load_adapter, merge_lora
from training.lora_study import forward_loss
from training.model_loading import load_training_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload one Breeze LoRA adapter")
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def assert_close_loss(
    actual: dict[str, float], expected: dict[str, float], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if not math.isclose(actual[key], expected_value, abs_tol=0.02):
            raise RuntimeError(
                f"{label} {key} loss mismatch: "
                f"expected={expected_value} actual={actual[key]}"
            )


def main() -> int:
    args = parse_args()
    receipt_path = args.variant_root / "receipt.json"
    reload_path = args.variant_root / "reload-receipt.json"
    if reload_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {reload_path}")
    receipt = json.loads(receipt_path.read_text())
    configuration = receipt["configuration"]
    source = receipt["source"]

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = load_training_model(source["model_root"], device=args.device)
    families = inject_lora(
        model,
        variant=receipt["variant"],
        rank=configuration["rank"],
        alpha=configuration["alpha"],
        seed=configuration["seed"],
    )
    hashes = load_adapter(model, receipt["adapter"]["path"])
    if hashes != receipt["adapter"]["hashes"]:
        raise RuntimeError("fresh-process adapter hashes do not match the receipt")

    example = torch.load(source["example"], map_location="cpu", weights_only=True)
    batch = {name: tensor.to(args.device) for name, tensor in example.items()}
    model.eval()
    adapter_loss = forward_loss(model, batch)
    assert_close_loss(adapter_loss, receipt["loss"]["adapted"], label="adapter")
    merged_count = merge_lora(model)
    if merged_count != len(families):
        raise RuntimeError("fresh-process merge count differs from target count")
    merged_loss = forward_loss(model, batch)
    assert_close_loss(merged_loss, receipt["loss"]["merged"], label="merged")

    reload_receipt = {
        "schema_version": 1,
        "status": "fresh_process_adapter_reload_and_merge_verified",
        "variant": receipt["variant"],
        "adapter_hashes_matched": True,
        "merged_module_count": merged_count,
        "loss": {"adapter": adapter_loss, "merged": merged_loss},
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
    }
    reload_path.write_text(json.dumps(reload_receipt, indent=2, sort_keys=True) + "\n")
    receipt["status"] = "adapter_and_fresh_process_merge_verified"
    receipt["reload_receipt"] = str(reload_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(reload_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
