import dataclasses
import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy as np

from .config_generator import generate_configs
from .core import ClusterConfig, DataGenerator, StrategyConfig
from .registry import StrategySpec, default_registry
from .reporting import (
    config_report,
    save_run_metadata,
    save_strategy_params,
    save_timeout_log,
)
from .utils import suppress_output
from .writer import DatasetWriter, H5Writer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    n_repeats: int = 10
    master_seed: int = 42
    output_dir: Path = Path("datasets")
    test_ratio: float = 0.1

    # Safety mechanism
    timeout: float = 30.0  # Seconds before killing a hanging process. 0 to disable.

    # config bounds
    n_configs: int = 5
    k_min: int = 2
    k_max: int = 15
    n_low: int = 100
    n_high: int = 2500
    d_low: int = 2
    d_high: int = 200


def _worker_loop(input_q: Queue[Any], output_q: Queue[Any]):
    """Run queued generation tasks and send back results until a stop signal."""
    while True:
        try:
            task = input_q.get()
            if task is None:
                break  # Sentinel to exit

            # Unpack task
            strategy_cls, cfg, strategy_config, seed = task

            # Run Generation (with output suppression for libraries like Densired)
            try:
                with suppress_output():
                    gen = DataGenerator(strategy_cls(cfg))
                    X, y = gen.generate_dataset(strategy_config, seed=seed)
                output_q.put(("ok", (X, y)))
            except Exception as e:
                output_q.put(("error", e))

        except Exception as e:
            # Catastrophic worker failure
            try:
                output_q.put(("fatal", e))
            except Exception:
                pass
            break


class _GenerationRunner:
    """Run datasets in a worker when timeouts are enabled, otherwise inline."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.context = multiprocessing.get_context("spawn")
        self.worker: BaseProcess | None = None
        self.input_q: Queue[Any] | None = None
        self.output_q: Queue[Any] | None = None
        if timeout > 0:
            self.start()

    def start(self) -> None:
        self.input_q = self.context.Queue()
        self.output_q = self.context.Queue()
        self.worker = self.context.Process(target=_worker_loop, args=(self.input_q, self.output_q))
        self.worker.daemon = True
        self.worker.start()

    def stop(self) -> None:
        if self.worker and self.worker.is_alive():
            assert self.input_q is not None
            self.input_q.put(None)
            self.worker.join(timeout=1)
            if self.worker.is_alive():
                self.worker.terminate()

    def restart(self) -> None:
        assert self.worker is not None
        self.worker.terminate()
        self.worker.join()
        self.start()

    def generate(
        self,
        spec: StrategySpec,
        cfg: ClusterConfig,
        strategy_config: StrategyConfig,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.timeout > 0:
            assert self.input_q is not None and self.output_q is not None
            self.input_q.put((spec.strategy_cls, cfg, strategy_config, seed))
            res_type, res_payload = self.output_q.get(timeout=self.timeout)
            if res_type == "ok":
                return res_payload
            raise res_payload

        with suppress_output():
            gen = DataGenerator(spec.strategy_cls(cfg))
            return gen.generate_dataset(strategy_config, seed=seed)


def run_generation(
    settings: GenerationSettings,
    *,
    strategies: list[StrategySpec] | None = None,
    writer: DatasetWriter | None = None,
) -> None:
    pipeline_start_time = time.perf_counter()
    total_gen_time = 0.0
    timeout_events = []

    strategies = strategies or default_registry()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Datasets will be saved under: {settings.output_dir}")

    # Save strategy parameters
    save_strategy_params(settings.output_dir)

    cluster_configs = generate_configs(
        n_configs=settings.n_configs,
        k_min=settings.k_min,
        k_max=settings.k_max,
        n_low=settings.n_low,
        n_high=settings.n_high,
        d_low=settings.d_low,
        d_high=settings.d_high,
        seed=settings.master_seed,
    )

    master_ss = np.random.SeedSequence(settings.master_seed)
    cfg_seed_sequences = master_ss.spawn(len(cluster_configs))

    # Generate report for the configs
    config_report(cluster_configs, settings.output_dir, seed=settings.master_seed)

    h5_writer = None
    if writer is None:
        h5_writer = H5Writer(
            settings.output_dir / "datalake.h5",
            n_repeats=settings.n_repeats,
            test_ratio=settings.test_ratio,
            seed=settings.master_seed,
        )
        writer = h5_writer

    runner = _GenerationRunner(settings.timeout)

    completed = False
    try:
        for cfg_idx, (cfg, cfg_ss) in enumerate(zip(cluster_configs, cfg_seed_sequences)):
            try:
                cfg.validate()
            except ValueError as e:
                logger.warning(f"Skipping invalid config #{cfg_idx}: {e}")
                continue

            logger.info(f"=== ClusterConfig #{cfg_idx} -> {cfg}")

            strat_seed_sequences = cfg_ss.spawn(len(strategies))
            for spec, strat_ss in zip(strategies, strat_seed_sequences):
                if not spec.supports_cfg(cfg):
                    continue

                repeat_ss_list = strat_ss.spawn(settings.n_repeats)
                repeats_payload: list[dict] = []

                for rep_i, rep_ss in enumerate(repeat_ss_list):
                    gen_start = time.perf_counter()
                    try:
                        rep_rng = np.random.default_rng(rep_ss)
                        strategy_config = spec.sampler(rep_rng, cfg)
                        dataset_seed = int(rep_ss.generate_state(1)[0])

                        try:
                            X, y = runner.generate(spec, cfg, strategy_config, dataset_seed)
                        except queue.Empty:
                            if settings.timeout <= 0:
                                raise
                            timeout_message = (
                                f"TIMEOUT ({settings.timeout}s): Killing {spec.name} "
                                f"cfg #{cfg_idx} rep #{rep_i} "
                                f"(k={cfg.num_clusters}, n={cfg.num_samples}, "
                                f"d={cfg.num_dimensions})"
                            )
                            logger.warning(timeout_message)
                            timeout_events.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                                f"{timeout_message}"
                            )
                            runner.restart()
                            continue

                        gen_duration = time.perf_counter() - gen_start
                        total_gen_time += gen_duration
                        logger.info(f"Generated {spec.name} cfg #{cfg_idx} rep #{rep_i}")

                        repeats_payload.append(
                            {
                                "seed": dataset_seed,
                                "strategy_config": strategy_config,
                                "X": X,
                                "y": y,
                            }
                        )
                    except Exception as e:
                        total_gen_time += time.perf_counter() - gen_start
                        logger.error(
                            f"Failed to generate {spec.name} cfg #{cfg_idx} rep #{rep_i}: {e}"
                        )

                if repeats_payload:
                    writer.save_group(
                        strategy_name=spec.name,
                        cfg_idx=cfg_idx,
                        cfg=cfg,
                        repeats=repeats_payload,
                    )
        completed = True
    finally:
        runner.stop()
        if h5_writer is not None:
            if completed:
                h5_writer.finish()
            else:
                h5_writer.abort()

    pipeline_duration = time.perf_counter() - pipeline_start_time
    logger.info(f"Pipeline completed in {pipeline_duration:.2f}s")

    logger.info(f"Total time spent in generation routines: {total_gen_time:.2f}s")

    # Save timeout log if any
    save_timeout_log(timeout_events, settings.output_dir)

    # Save run metadata
    save_run_metadata(
        dataclasses.asdict(settings),
        pipeline_duration,
        total_gen_time,
        settings.output_dir,
    )
