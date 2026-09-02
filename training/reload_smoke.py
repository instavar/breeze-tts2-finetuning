from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from training.full_sft_smoke import SENTINEL_PARAMETERS
from training.model_loading import load_training_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a Breeze SFT feasibility checkpoint"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    args = parse_args()
    receipt_path = args.run_root / "full-sft-receipt.json"
    example_path = args.run_root / "example.pt"
    reload_path = args.run_root / "reload-receipt.json"
    if reload_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {reload_path}")
    receipt = json.loads(receipt_path.read_text())
    checkpoint = Path(receipt["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = load_training_model(checkpoint, device=args.device)
    model.eval()
    parameters = dict(model.named_parameters())
    sentinel_checks = {}
    for family, name in SENTINEL_PARAMETERS.items():
        actual = tensor_sha256(parameters[name])
        expected = receipt["updates"][family]["after_sha256"]
        sentinel_checks[family] = {
            "parameter": name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
    if not all(row["matched"] for row in sentinel_checks.values()):
        raise RuntimeError(f"reloaded sentinel mismatch: {sentinel_checks}")

    example = torch.load(example_path, map_location="cpu", weights_only=True)
    batch = {name: tensor.to(args.device) for name, tensor in example.items()}
    with torch.no_grad():
        outputs = model(**batch, use_cache=False, return_dict=True)
    losses = {
        "total": float(outputs.loss.detach().float().cpu().item()),
        "backbone": float(outputs.backbone_loss.detach().float().cpu().item()),
        "depth_decoder": float(
            outputs.depth_decoder_loss.detach().float().cpu().item()
        ),
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"reloaded model produced non-finite loss: {losses}")

    reload_receipt = {
        "schema_version": 1,
        "status": "fresh_process_reload_verified",
        "python": sys.version,
        "torch": torch.__version__,
        "checkpoint": str(checkpoint),
        "sentinels": sentinel_checks,
        "loss": losses,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
    }
    reload_path.write_text(json.dumps(reload_receipt, indent=2, sort_keys=True) + "\n")
    receipt["status"] = "full_sft_step_and_fresh_process_reload_verified"
    receipt["reload_receipt"] = str(reload_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(reload_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
