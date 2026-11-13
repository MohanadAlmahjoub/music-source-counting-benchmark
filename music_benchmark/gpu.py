"""Measure a covariance and eigensolver subpipeline on an optional CUDA GPU."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from .pilot import complex_normal
from .robustness import calibrate, noise_factor


def transform_and_eigenvalues(array, frames, whitener):
    white = frames @ whitener.T
    covariance = white.swapaxes(-1, -2) @ white.conj() / frames.shape[-2]
    return array.linalg.eigvalsh(covariance)


def measure(operation, warmup, repetitions):
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        operation()
        samples.append(perf_counter_ns() - start)
    return samples


def benchmark(output, repetitions, warmup):
    import cupy as cp

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    factor = noise_factor(0.2)
    noise = complex_normal(rng, (2048, 8)) @ factor.T
    whitener = np.linalg.inv(calibrate(noise)).astype(np.complex64)
    device_whitener = cp.asarray(whitener)
    stream = cp.cuda.get_current_stream()
    results = []
    raw = []
    for batch in (1, 4, 16, 64, 256):
        frames = np.ascontiguousarray(
            complex_normal(rng, (batch, 128, 8)) @ factor.T, dtype=np.complex64
        )
        oracle = transform_and_eigenvalues(np, frames.astype(np.complex128), whitener.astype(np.complex128))
        host_input_memory = cp.cuda.alloc_pinned_memory(frames.nbytes)
        host_input = np.ndarray(frames.shape, dtype=frames.dtype, buffer=host_input_memory)
        host_input[:] = frames
        host_output_memory = cp.cuda.alloc_pinned_memory(batch * 8 * np.dtype(np.float32).itemsize)
        host_output = np.ndarray((batch, 8), dtype=np.float32, buffer=host_output_memory)
        device_frames = cp.asarray(frames)
        upload_buffer = cp.empty_like(device_frames)

        def cpu():
            return transform_and_eigenvalues(np, frames, whitener)

        def gpu_resident():
            values = transform_and_eigenvalues(cp, device_frames, device_whitener)
            stream.synchronize()
            return values

        def gpu_transfer():
            upload_buffer.set(host_input, stream=stream)
            values = transform_and_eigenvalues(cp, upload_buffer, device_whitener)
            values.get(out=host_output, stream=stream, blocking=False)
            stream.synchronize()
            return host_output

        for mode, run in (("cpu", cpu), ("gpu_resident", gpu_resident), ("gpu_transfer", gpu_transfer)):
            values = run()
            if isinstance(values, cp.ndarray):
                values = values.get()
            max_error = float(np.max(np.abs(values - oracle)))
            samples = measure(run, warmup, repetitions)
            median, p95, p99 = np.percentile(samples, [50, 95, 99])
            results.append({
                "batch": batch, "mode": mode, "median_us": median / 1e3,
                "p95_us": p95 / 1e3, "p99_us": p99 / 1e3,
                "frames_per_second": batch * repetitions * 1e9 / sum(samples),
                "max_eigenvalue_abs_error_vs_fp64": max_error,
            })
            raw.extend((batch, mode, iteration, ns) for iteration, ns in enumerate(samples))

    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("batch", "mode", "iteration", "elapsed_ns"))
        writer.writerows(raw)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    (output / "environment.json").write_text(json.dumps({
        "gpu": properties["name"].decode(),
        "cupy": cp.__version__,
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "repetitions": repetitions, "warmup": warmup, "blas_threads": 1,
        "input": "128x8 complex64 per frame, white transform resident",
        "output": "eight float32 eigenvalues per frame",
        "scope": "whitening, covariance and eigenvalues only, no eigenvectors, MDL or spectrum",
        "timing": "synchronized host wall clock, preallocated pinned transfer buffers; includes library temporaries",
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/gpu"))
    parser.add_argument("--repetitions", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=24)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("use at least one repetition and a nonnegative warm-up")
    with threadpool_limits(limits=1, user_api="blas"):
        benchmark(args.out, args.repetitions, args.warmup)


if __name__ == "__main__":
    main()
