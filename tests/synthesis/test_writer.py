import json
from pathlib import Path

import h5py
import numpy as np

from tabk.synthesis import ClusterConfig, H5Writer
from tabk.utils import apply_standard_scaling


def test_h5_writer_publishes_normalized_samples_and_metadata(tmp_path: Path):
    path = tmp_path / "datalake.h5"
    writer = H5Writer(path, n_repeats=2, test_ratio=0.5, seed=7)
    cfg = ClusterConfig(num_clusters=3, num_samples=4, num_dimensions=2)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    repeats = [
        {"seed": seed, "strategy_config": {"alpha": np.float64(0.2)}, "X": X, "y": [0, 1, 1, 2]}
        for seed in (123, 456)
    ]

    writer.save_group("Example", 0, cfg, repeats)
    assert not path.exists()
    writer.finish()

    with h5py.File(path) as h5f:
        samples = h5f["datasets"]
        assert set(samples) == {"Example_cfg00000_rep0", "Example_cfg00000_rep1"}
        assert {samples[sid].attrs["split"] for sid in samples} == {"train", "test"}
        for sid in samples:
            sample = samples[sid]
            np.testing.assert_array_equal(
                sample["normalized_features"][:], apply_standard_scaling(X)
            )
            np.testing.assert_array_equal(sample["labels"][:], [0, 1, 1, 2])
            assert sample.attrs["k_value"] == 3
            assert sample.attrs["strategy"] == "Example"
            assert json.loads(sample.attrs["strategy_config_json"]) == {"alpha": 0.2}


def test_h5_writer_abort_preserves_previous_output(tmp_path: Path):
    path = tmp_path / "datalake.h5"
    path.write_bytes(b"previous output")

    writer = H5Writer(path, n_repeats=1, test_ratio=0.1, seed=42)
    writer.abort()

    assert path.read_bytes() == b"previous output"
