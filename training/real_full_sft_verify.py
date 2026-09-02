from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from training.lora_study import loss_receipt
from training.model_loading import load_training_model
from training.real_data import sha256_file
from training.real_full_sft import checkpoint_role_hashes, verify_model_role
from training.real_lora import load_cache, move_example


def copy_inference_export(checkpoint: Path, destination: Path) -> None:
    model_role = json.loads((checkpoint / "model-role.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for relative_name in model_role["files"]:
        source = checkpoint / relative_name
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(checkpoint / "model-role.json", destination / "model-role.json")


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
    parser = argparse.ArgumentParser(
        description="Freshly reload and export a selected Breeze full-SFT checkpoint"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-examples", type=int, default=64)
    args = parser.parse_args()

    checkpoint_receipt = json.loads(
        (args.checkpoint / "checkpoint-receipt.json").read_text()
    )
    if checkpoint_role_hashes(args.checkpoint) != checkpoint_receipt["roles"]:
        raise RuntimeError("selected checkpoint role hashes do not match")
    verify_model_role(args.checkpoint)
    _cache_receipt, _train, validation = load_cache(args.cache_root)
    paths = validation[: args.validation_examples]
    if not paths:
        raise ValueError("no validation examples selected")

    torch.cuda.set_device(args.device)
    model = load_training_model(args.checkpoint, device=args.device)
    fresh_loss = average_loss(model, paths, args.device)
    state = json.loads((args.checkpoint / "trainer-state.json").read_text())
    expected = state["history"][-1]["validation_loss"]
    for name, value in fresh_loss.items():
        if not math.isclose(value, expected[name], abs_tol=0.02):
            raise RuntimeError(
                f"fresh full-SFT {name} differs: expected={expected[name]} actual={value}"
            )

    partial = args.export_output.with_name(args.export_output.name + ".partial")
    if args.export_output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite export: {args.export_output}")
    copy_inference_export(args.checkpoint, partial)
    shutil.copytree(args.model_root / "audio_tokenizer", partial / "audio_tokenizer")
    for name in ("LICENSE", "README.md"):
        source = args.model_root / name
        if source.is_file() and not (partial / name).exists():
            shutil.copy2(source, partial / name)
    partial.rename(args.export_output)

    receipt = {
        "schema_version": 1,
        "status": "fresh_full_sft_checkpoint_reload_and_export_complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_receipt_sha256": sha256_file(
            args.checkpoint / "checkpoint-receipt.json"
        ),
        "cache_receipt_sha256": sha256_file(args.cache_root / "cache-receipt.json"),
        "validation_examples": len(paths),
        "validation_loss": fresh_loss,
        "export": str(args.export_output),
        "export_model_index_sha256": sha256_file(
            args.export_output / "model.safetensors.index.json"
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_cuda_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(args.device)
        ),
        "interpretation_boundary": (
            "Fresh reload and export verify package mechanics and teacher-forced loss "
            "only. They do not establish perceptual quality or production fitness."
        ),
    }
    if args.verification_output.exists():
        raise FileExistsError(args.verification_output)
    args.verification_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
