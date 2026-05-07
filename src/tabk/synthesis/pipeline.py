import dataclasses
import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config_generator import generate_configs
from .core import DataGenerator
from .registry import StrategySpec, default_registry
from .reporting import (
    config_report,
    save_dataset_manifest,
    save_run_metadata,
    save_strategy_params,
    save_timeout_log,
)
from .utils import suppress_output
from .writer import DatasetWriter, NPZWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    n_repeats: int = 10
    master_seed: int = 42
    output_dir: Path = Path("datasets")

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


def _worker_loop(input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):
    """
    Persistent worker loop.
    Waits for tasks, executes them, and sends results back.
    """
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
    writer = writer or NPZWriter(
        settings.output_dir,
        n_configs=settings.n_configs,
        n_repeats=settings.n_repeats,
    )

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

    # --- Worker Management ---
    worker = None
    input_q = None
    output_q = None

    def start_worker():
        nonlocal worker, input_q, output_q
        input_q = multiprocessing.Queue()
        output_q = multiprocessing.Queue()
        worker = multiprocessing.Process(target=_worker_loop, args=(input_q, output_q))
        worker.daemon = True
        worker.start()

    def stop_worker():
        nonlocal worker
        if worker and worker.is_alive():
            input_q.put(None)
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()

    if settings.timeout > 0:
        start_worker()

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

                        X, y = None, None

                        if settings.timeout > 0:
                            # --- IPC Execution ---
                            task = (spec.strategy_cls, cfg, strategy_config, dataset_seed)
                            input_q.put(task)

                            try:
                                res_type, res_payload = output_q.get(timeout=settings.timeout)
                                if res_type == "ok":
                                    X, y = res_payload
                                else:
                                    raise res_payload  # Re-raise exception from worker
                            except queue.Empty:
                                logger.warning(
                                    f"TIMEOUT ({settings.timeout}s): Killing {spec.name} "
                                    f"cfg #{cfg_idx} rep #{rep_i} "
                                    f"(k={cfg.num_clusters}, n={cfg.num_samples}, "
                                    f"d={cfg.num_dimensions})"
                                )
                                timeout_events.append(
                                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                                    f"TIMEOUT ({settings.timeout}s): Killing {spec.name} "
                                    f"cfg #{cfg_idx} rep #{rep_i} "
                                    f"(k={cfg.num_clusters}, n={cfg.num_samples}, "
                                    f"d={cfg.num_dimensions})"
                                )
                                # Kill and restart worker
                                worker.terminate()
                                worker.join()
                                start_worker()
                                continue
                        else:
                            # --- In-Process Execution (No safety) ---
                            with suppress_output():
                                gen = DataGenerator(spec.strategy_cls(cfg))
                                X, y = gen.generate_dataset(strategy_config, seed=dataset_seed)

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
    finally:
        if settings.timeout > 0:
            stop_worker()

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

    # Save dataset manifest
    save_dataset_manifest(settings.output_dir)
