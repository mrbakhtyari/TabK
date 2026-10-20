import json
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file

from .config import AppConfig


def export_for_hub(model_dir: Path, out_dir: Path) -> int:
    """Convert checkpoints/*.pth from a training run into config.json + fold_<i>.safetensors."""
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
        json.dump(AppConfig.from_dict(cast(dict, config)).to_dict(), f, indent=2)

    return len(checkpoints)
