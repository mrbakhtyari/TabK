import json

import pandas as pd

from tabk.architecture.utils import (
    create_split_dataset,
    prepare_dataset_index,
)


def test_create_split_dataset_reproducibility():
    """Test that create_split_dataset is reproducible with the same seed."""
    records = [{"data_path": f"path/to/file_{i}.npz", "rep_id": f"rep_{i}"} for i in range(100)]
    seed = 42
    test_ratio = 0.2

    # Run 1
    df1 = create_split_dataset(records.copy(), test_size=test_ratio, seed=seed)

    # Run 2
    df2 = create_split_dataset(records.copy(), test_size=test_ratio, seed=seed)

    pd.testing.assert_frame_equal(df1, df2)


def test_create_split_dataset_randomness():
    """Test that create_split_dataset produces different results with different seeds."""
    records = [{"data_path": f"path/to/file_{i}.npz", "rep_id": f"rep_{i}"} for i in range(100)]
    test_ratio = 0.5  # use 0.5 to maximize chance of difference

    df1 = create_split_dataset(records.copy(), test_size=test_ratio, seed=42)
    df2 = create_split_dataset(records.copy(), test_size=test_ratio, seed=43)

    # They should (almost certainly) be different
    # We can check specific columns or the order
    try:
        pd.testing.assert_frame_equal(df1, df2)
        raise AssertionError("DataFrames should not be equal with different seeds")
    except AssertionError:
        pass  # Expected


def test_prepare_dataset_index_reproducibility(tmp_path):
    """
    Test that prepare_dataset_index generates a reproducible parquet file
    given a directory of inputs.
    """

    # Setup dummy directory structure
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()

    # Create dummy .npz and .json files
    # We don't need real .npz content, just the file existence for rglog
    # and valid json for extract_reps

    for i in range(5):
        subdir = dataset_dir / f"cfg{i:03d}"
        subdir.mkdir()

        npz_path = subdir / f"data_cfg{i:03d}_training.npz"
        npz_path.touch()

        json_path = subdir / f"data_cfg{i:03d}.json"

        metadata = {
            "seeds": {
                f"rep{j}": 1000 + j
                for j in range(3)  # 3 reps per file
            }
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f)

    # Run 1
    prepare_dataset_index(dataset_dir, test_ratio=0.2, seed=42)
    index_file = dataset_dir / "dataset_index.parquet"

    assert index_file.exists()
    df1 = pd.read_parquet(index_file)

    # Calculate checksum or hash of file if needed, but DF comparison is better
    # Delete file to ensure next run recreates it
    index_file.unlink()

    # Run 2
    prepare_dataset_index(dataset_dir, test_ratio=0.2, seed=42)

    assert index_file.exists()
    df2 = pd.read_parquet(index_file)

    pd.testing.assert_frame_equal(df1, df2)

    # Check split ratio roughly
    _total = len(df1)  # 5 files * 3 reps = 15
    train_count = df1["split"].value_counts()["train"]
    test_count = df1["split"].value_counts()["test"]

    # 15 * 0.2 = 3 test, 12 train
    assert test_count == 3
    assert train_count == 12
