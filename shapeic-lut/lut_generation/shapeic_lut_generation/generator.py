from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import GenerationConfig
from .ngspice import simulate_block
from .writer import LutStorage


@dataclass(frozen=True)
class Job:
    indices: tuple[int, int, int]
    length: float
    vbs: float
    finger_width: float


def generate(config: GenerationConfig, force: bool = False) -> None:
    if config.output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {config.output_path}; use --force to replace it"
        )
    jobs = _jobs(config)
    print(f"Preflight: {config.device.name}")
    first = jobs[0]
    first_result = _run_job(config, first)

    with tempfile.TemporaryDirectory(prefix="shapeic-lut-generation-") as temporary:
        storage = LutStorage(config, Path(temporary))
        storage.put(first.indices, first_result)
        print(f"Progress: 1/{len(jobs)} jobs completed")
        remaining = jobs[1:]
        if config.simulator.workers == 1:
            for completed, job in enumerate(remaining, start=2):
                storage.put(job.indices, _run_job(config, job))
                print(f"Progress: {completed}/{len(jobs)} jobs completed")
        else:
            with ProcessPoolExecutor(max_workers=config.simulator.workers) as executor:
                futures = {executor.submit(_run_job, config, job): job for job in remaining}
                for completed, future in enumerate(as_completed(futures), start=2):
                    job = futures[future]
                    storage.put(job.indices, future.result())
                    print(f"Progress: {completed}/{len(jobs)} jobs completed")

        storage.validate_and_flush()
        storage.write_archive(force=force)
    print(f"Wrote {config.output_path}")


def _run_job(config: GenerationConfig, job: Job):
    return simulate_block(config, job.length, job.vbs, job.finger_width)


def _jobs(config: GenerationConfig) -> list[Job]:
    axes = config.sweep.axes()
    return [
        Job(
            indices=(length_index, vbs_index, width_index),
            length=float(length),
            vbs=float(vbs),
            finger_width=float(width),
        )
        for length_index, length in enumerate(axes["length"])
        for vbs_index, vbs in enumerate(axes["vbs"])
        for width_index, width in enumerate(axes["finger_width"])
    ]
