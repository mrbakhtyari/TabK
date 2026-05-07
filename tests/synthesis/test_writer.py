import json
from pathlib import Path

import numpy as np

from tabk.synthesis import NPZWriter


def test_npz_writer_saves_group_and_metadata(tmp_path: Path):
    n_configs = 10
    n_repeats = 5
    writer = NPZWriter(base_dir=tmp_path, n_configs=n_configs, n_repeats=n_repeats)

    strategy_name = "DummyStrategy"
    cfg_idx = 0
    cfg = {"k": 3, "n": 20, "d": 2}

    repeats = [
        {
            "seed": 123,
            "strategy_config": {"alpha": 0.1},
            "X": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            "y": np.array([0, 1], dtype=np.int8),
        },
        {
            "seed": 456,
            "strategy_config": {"alpha": 0.2},
            "X": np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float64),
            "y": np.array([1, 0], dtype=np.int8),
        },
    ]

    writer.save_group(strategy_name=strategy_name, cfg_idx=cfg_idx, cfg=cfg, repeats=repeats)

    cfg_width = len(str(n_configs - 1))
    rep_width = len(str(n_repeats - 1))

    # Check files exist
    target_dir = tmp_path / strategy_name / f"cfg{cfg_idx:0{cfg_width}d}"
    npz_path = target_dir / f"{strategy_name}_cfg{cfg_idx:0{cfg_width}d}.npz"
    meta_path = target_dir / f"{strategy_name}_cfg{cfg_idx:0{cfg_width}d}.json"
    assert npz_path.exists(), "NPZ dataset not saved"
    assert meta_path.exists(), "Metadata JSON not saved"

    # Validate npz contents
    with np.load(npz_path) as npz:
        # Rep 0 keys
        np.testing.assert_allclose(npz[f"rep{0:0{rep_width}d}_X"], repeats[0]["X"])
        np.testing.assert_array_equal(npz[f"rep{0:0{rep_width}d}_y"], repeats[0]["y"])
        # Rep 1 keys
        np.testing.assert_allclose(npz[f"rep{1:0{rep_width}d}_X"], repeats[1]["X"])
        np.testing.assert_array_equal(npz[f"rep{1:0{rep_width}d}_y"], repeats[1]["y"])

    # Validate metadata JSON
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["strategy_name"] == strategy_name
    assert meta["cfg_index"] == cfg_idx
    assert meta["n_repeats"] == len(repeats)
    assert isinstance(meta["seeds"], dict) and len(meta["seeds"]) == 2
    assert meta["seeds"][f"rep{0:0{rep_width}d}"] == 123
    assert meta["seeds"][f"rep{1:0{rep_width}d}"] == 456
    assert isinstance(meta["strategy_configs"], dict) and len(meta["strategy_configs"]) == 2
    assert meta["strategy_configs"][f"rep{0:0{rep_width}d}"]["alpha"] == 0.1
    assert meta["strategy_configs"][f"rep{1:0{rep_width}d}"]["alpha"] == 0.2
    assert meta["cluster_config"]  # is JSONable
