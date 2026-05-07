from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tabk.synthesis import (
    ClusterConfig,
    GenerationSettings,
    StrategySpec,
    pipeline,
    run_generation,
)


# Top-level strategy class to be picklable for multiprocessing tests
@dataclass(frozen=True, slots=True)
class TopLevelDummyStrategy:
    cfg: Any

    def generate(self, config: dict):
        raise NotImplementedError


def _fake_generate_configs_single(**_: Any):
    return [ClusterConfig(num_clusters=3, num_samples=10, num_dimensions=2)]


class RecordingWriter:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def save_group(self, strategy_name: str, cfg_idx: int, cfg: Any, repeats: list[dict]):
        self.calls.append(
            {
                "strategy_name": strategy_name,
                "cfg_idx": cfg_idx,
                "cfg": cfg,
                "repeats": repeats,
            }
        )


def _sampler(rng: np.random.Generator, cfg: Any):
    return {"dummy": int(rng.integers(0, 100))}


def _fake_data_generator_class():
    class FakeDG:
        def __init__(self, strategy: Any):  # noqa: ARG002
            pass

        def generate_dataset(self, config: dict, seed: int | None = None):
            rng = np.random.default_rng(seed)
            X = rng.random((4, 2), dtype=np.float64)
            y = np.array([0, 1, 1, 0], dtype=np.int8)
            return X, y

    return FakeDG


def test_run_generation_inprocess_avoids_multiprocessing_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Avoid multiprocessing pickling issues by disabling timeout
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)
    monkeypatch.setattr(pipeline, "DataGenerator", _fake_data_generator_class())

    calls_writer = RecordingWriter()

    @dataclass(frozen=True, slots=True)
    class LocalDummyStrategy:
        cfg: Any

        def generate(self, config: dict):  # not used
            raise NotImplementedError

    spec = StrategySpec(
        name="DummyLocal",
        strategy_cls=LocalDummyStrategy,
        sampler=lambda rng, cfg: {"dummy": 1},
        supports_cfg=lambda cfg: True,
    )

    settings = GenerationSettings(
        n_repeats=3,
        master_seed=123,
        output_dir=tmp_path,
        n_configs=1,
        k_min=2,
        k_max=3,
        n_low=10,
        n_high=20,
        d_low=2,
        d_high=2,
        timeout=0.0,  # critical: run in-process
    )
    run_generation(settings, strategies=[spec], writer=calls_writer)

    assert len(calls_writer.calls) == 1
    call = calls_writer.calls[0]
    assert call["strategy_name"] == "DummyLocal"
    assert call["cfg_idx"] == 0
    assert len(call["repeats"]) == settings.n_repeats
    for rep in call["repeats"]:
        assert "X" in rep and "y" in rep and "seed" in rep and "strategy_config" in rep


def test_run_generation_supports_cfg_filtering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)
    monkeypatch.setattr(pipeline, "DataGenerator", _fake_data_generator_class())

    calls_writer = RecordingWriter()

    spec_yes = StrategySpec(
        name="Supports",
        strategy_cls=TopLevelDummyStrategy,
        sampler=_sampler,
        supports_cfg=lambda cfg: True,
    )
    spec_no = StrategySpec(
        name="NoSupports",
        strategy_cls=TopLevelDummyStrategy,
        sampler=_sampler,
        supports_cfg=lambda cfg: False,
    )

    settings = GenerationSettings(
        n_repeats=1,
        master_seed=7,
        output_dir=tmp_path,
        n_configs=1,
        k_min=2,
        k_max=3,
        n_low=10,
        n_high=20,
        d_low=2,
        d_high=2,
        timeout=0.0,
    )
    run_generation(settings, strategies=[spec_yes, spec_no], writer=calls_writer)

    assert len(calls_writer.calls) == 1
    assert calls_writer.calls[0]["strategy_name"] == "Supports"
    assert len(calls_writer.calls[0]["repeats"]) == 1
