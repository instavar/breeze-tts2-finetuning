from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.lora import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise matched Breeze LoRA variants"
    )
    parser.add_argument("--study-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.study_root / "study-receipt.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    variants = {}
    for name in VARIANTS:
        receipt = json.loads((args.study_root / name / "receipt.json").read_text())
        if receipt["status"] != "adapter_and_fresh_process_merge_verified":
            raise RuntimeError(f"variant is not fully verified: {name}")
        variants[name] = {
            "status": receipt["status"],
            "module_count": receipt["targets"]["module_count"],
            "trainable_parameters": receipt["targets"]["trainable_parameters"],
            "initial_loss": receipt["loss"]["initial"],
            "adapted_loss": receipt["loss"]["adapted"],
            "loss_change": receipt["loss"]["change"],
            "peak_cuda_memory_bytes": receipt["runtime"]["peak_cuda_memory_bytes"],
            "receipt": str(args.study_root / name / "receipt.json"),
        }

    backbone_change = variants["backbone_only"]["loss_change"]
    technically_fits = all(value < 0 for value in backbone_change.values())
    summary = {
        "schema_version": 1,
        "status": "matched_lora_variants_verified",
        "variants": variants,
        "bounded_findings": {
            "backbone_only_reduced_all_one_example_losses": technically_fits,
            "direct_depth_and_projection_gradients_exist": True,
            "quality_sufficiency_determined": False,
            "reason": (
                "A one-example objective-fit smoke cannot establish speaker identity, "
                "accent, prosody, or out-of-sample quality. Lower loss from broader "
                "targeting shows optimization capacity, not perceptual necessity."
            ),
        },
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
