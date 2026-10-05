import json
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np
import pytest

from tabk.synthesis import (
    ClusterConfig,
    GenerationSettings,
    SamplingConfig,
    StrategySpec,
    default_registry,
    pipeline,
    run_generation,
)
from tabk.synthesis.config import CesarCominConfig, DensiredConfig, LogUniformRange
from tabk.utils import apply_standard_scaling


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

    assert json.loads((tmp_path / "strategy_params.json").read_text()) == {"DummyLocal": None}
    assert len(calls_writer.calls) == 1
    call = calls_writer.calls[0]
    assert call["strategy_name"] == "DummyLocal"
    assert call["cfg_idx"] == 0
    assert len(call["repeats"]) == settings.n_repeats
    for rep in call["repeats"]:
        assert "X" in rep and "y" in rep and "seed" in rep and "strategy_config" in rep


def test_run_generation_uses_and_reports_selected_custom_priors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)
    monkeypatch.setattr(pipeline, "DataGenerator", _fake_data_generator_class())
    monkeypatch.setattr(pipeline, "config_report", lambda *args, **kwargs: None)

    priors = SamplingConfig(
        cesar_comin=CesarCominConfig(alpha=LogUniformRange(0.2, 0.3)),
        densired=DensiredConfig(use_connectors_prob=0, dens_factors_prob=1),
    )
    registry = default_registry(priors)
    selected = [spec for spec in registry if spec.name in {"CesarComin", "Densired"}]
    writer = RecordingWriter()
    run_generation(
        GenerationSettings(n_repeats=3, n_configs=1, output_dir=tmp_path, timeout=0),
        strategies=selected,
        writer=writer,
    )

    assert {call["strategy_name"] for call in writer.calls} == {"CesarComin", "Densired"}
    for call in writer.calls:
        for repeat in call["repeats"]:
            params = repeat["strategy_config"]
            if call["strategy_name"] == "CesarComin":
                assert 0.2 <= params["alpha"] <= 0.3
            else:
                assert params["connections"] == 0
                assert params["ratio_con"] == 0
                assert params["dens_factors"]

    report = json.loads((tmp_path / "strategy_params.json").read_text())
    assert set(report) == {"CesarComin", "Densired"}
    assert report["CesarComin"]["alpha"] == {"dist": "loguniform", "low": 0.2, "high": 0.3}
    assert report["Densired"]["use_connectors_prob"] == 0
    assert report["Densired"]["dens_factors_prob"] == 1


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


def test_run_generation_recovers_after_timeout_and_skips_failed_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)

    outcomes = [
        Empty(),
        ("error", ValueError("bad sample")),
        ("ok", (np.ones((4, 2)), np.zeros(4))),
    ]
    processes = []

    class FakeQueue:
        def __init__(self):
            self.tasks = []

        def put(self, task):
            self.tasks.append(task)

        def get(self, timeout):  # noqa: ARG002
            result = outcomes.pop(0)
            if isinstance(result, Empty):
                raise result
            return result

    class FakeProcess:
        def __init__(self, target, args):  # noqa: ARG002
            self.terminated = False
            processes.append(self)

        def start(self):
            pass

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self, timeout=None):  # noqa: ARG002
            pass

    class FakeContext:
        Queue = FakeQueue
        Process = FakeProcess

    def fake_get_context(method):
        assert method == "spawn"
        return FakeContext()

    monkeypatch.setattr(pipeline.multiprocessing, "get_context", fake_get_context)

    calls_writer = RecordingWriter()
    spec = StrategySpec(
        name="Recovering",
        strategy_cls=TopLevelDummyStrategy,
        sampler=_sampler,
        supports_cfg=lambda cfg: True,
    )
    settings = GenerationSettings(
        n_repeats=3,
        output_dir=tmp_path,
        n_configs=1,
        timeout=0.1,
    )

    run_generation(settings, strategies=[spec], writer=calls_writer)

    assert len(processes) == 2
    assert processes[0].terminated
    assert len(calls_writer.calls) == 1
    repeats = calls_writer.calls[0]["repeats"]
    assert len(repeats) == 1
    np.testing.assert_array_equal(repeats[0]["X"], np.ones((4, 2)))
    np.testing.assert_array_equal(repeats[0]["y"], np.zeros(4))
    assert outcomes == []


def test_inprocess_queue_empty_is_a_generation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)

    class FailingGenerator:
        def __init__(self, strategy):  # noqa: ARG002
            pass

        def generate_dataset(self, strategy_config, seed):  # noqa: ARG002
            raise Empty("from generator")

    monkeypatch.setattr(pipeline, "DataGenerator", FailingGenerator)
    calls_writer = RecordingWriter()
    spec = StrategySpec(
        name="Failing",
        strategy_cls=TopLevelDummyStrategy,
        sampler=_sampler,
        supports_cfg=lambda cfg: True,
    )

    run_generation(
        GenerationSettings(n_repeats=1, n_configs=1, output_dir=tmp_path, timeout=0),
        strategies=[spec],
        writer=calls_writer,
    )

    assert calls_writer.calls == []
    assert not (tmp_path / "timeouts.log").exists()
    assert "Failed to generate Failing" in caplog.text


def test_generation_writes_normalized_h5_with_deterministic_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pipeline, "generate_configs", _fake_generate_configs_single)
    monkeypatch.setattr(pipeline, "DataGenerator", _fake_data_generator_class())
    strategies = [
        StrategySpec(name, TopLevelDummyStrategy, _sampler, lambda cfg: True)
        for name in ("Zulu", "Alpha")
    ]
    run_generation(
        GenerationSettings(
            output_dir=tmp_path,
            n_configs=1,
            n_repeats=3,
            master_seed=19,
            timeout=0,
            test_ratio=0.25,
        ),
        strategies=strategies,
    )

    assert not list(tmp_path.rglob("*.npz"))
    assert not (tmp_path / "datalake.h5.tmp").exists()
    with h5py.File(tmp_path / "datalake.h5") as h5f:
        samples = h5f["datasets"]
        assert set(samples) == {
            f"{strategy}_cfg00000_rep{rep}" for strategy in ("Alpha", "Zulu") for rep in range(3)
        }
        assert {
            sample_id for sample_id in samples if samples[sample_id].attrs["split"] == "test"
        } == {"Alpha_cfg00000_rep1"}
        for sample_id in samples:
            sample = samples[sample_id]
            seed = int(sample.attrs["seed"])
            X = np.random.default_rng(seed).random((4, 2), dtype=np.float64)
            np.testing.assert_array_equal(
                sample["normalized_features"][:], apply_standard_scaling(X)
            )
            np.testing.assert_array_equal(sample["labels"][:], [0, 1, 1, 0])
            assert sample.attrs["k_value"] == 3
            assert sample.attrs["config_group_id"] == sample_id.rsplit("_", 1)[0]
