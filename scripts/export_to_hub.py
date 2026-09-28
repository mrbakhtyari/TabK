"""Convert a TabK training output directory into the Hugging Face Hub layout.

Input:  <model-dir>/checkpoints/*.pth (as written by train_TabK.py)
Output: <out-dir>/config.json + fold_<i>.safetensors (weights only, no optimizer state)

Usage:
    uv run scripts/export_to_hub.py --model-dir models/TabK --out-dir hub/TabK
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def export(model_dir: Path, out_dir: Path) -> None:
    checkpoints = sorted((model_dir / "checkpoints").glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir / 'checkpoints'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = None
    for i, path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if config is None:
            config = checkpoint["config"]
        elif checkpoint["config"] != config:
            raise ValueError(f"{path.name} has a different config than the other folds")

        state_dict = {k: v.contiguous() for k, v in checkpoint["model_state_dict"].items()}
        save_file(state_dict, out_dir / f"fold_{i}.safetensors")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for extra in ("training_history.json", "loss_curves.png"):
        if (model_dir / extra).is_file():
            shutil.copy2(model_dir / extra, out_dir / extra)

    print(f"Exported {len(checkpoints)} folds to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.model_dir, args.out_dir)


if __name__ == "__main__":
    main()
