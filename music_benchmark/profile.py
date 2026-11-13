"""Measure CPU MUSIC stages on fixed inputs and check an independent SVD reference."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .pipeline import (
    METHODS,
    STAGES,
    estimate_noise_covariance,
    generate_snapshot,
    make_steering,
    reference_music,
    run_music,
    whitening_matrix,
)


PATHS = (
    ("raw_dense", "raw", "dense"),
    ("raw_projected", "raw", "projected"),
    ("whitened_dense", "whitened", "dense"),
    ("whitened_projected", "whitened", "projected"),
)
FIXTURES = (
    ((), 0.0, 0.0),
    ((-17.5,), 0.0, 0.0),
    ((-24.5, 18.5), 5.0, 12.0),
    ((-40.0, 5.0, 42.0), 10.0, 12.0),
)
CALIBRATION_SAMPLES = 2048
DIAGONAL_LOADING = 0.01
NOISE_RHO = 0.2


def _noise_covariance():
    distance = np.abs(np.arange(8)[:, None] - np.arange(8))
    return NOISE_RHO**distance


def _fixtures(seed, frames):
    rng = np.random.default_rng(seed)
    noise_covariance = _noise_covariance()
    calibration = generate_snapshot(
        rng, (), 0.0, noise_covariance=noise_covariance, snapshot_count=CALIBRATION_SAMPLES
    )
    snapshots = np.stack(
        [generate_snapshot(rng, *FIXTURES[index % len(FIXTURES)], noise_covariance) for index in range(frames)]
    )
    return calibration, snapshots


def _generation_timings(seed, frames):
    rng = np.random.default_rng(seed)
    covariance = _noise_covariance()
    samples = {"generate_calibration_noise": [], "generate_frame": []}
    for index in range(frames):
        start = perf_counter_ns()
        generate_snapshot(rng, (), 0.0, noise_covariance=covariance, snapshot_count=CALIBRATION_SAMPLES)
        samples["generate_calibration_noise"].append(perf_counter_ns() - start)
        start = perf_counter_ns()
        generate_snapshot(rng, *FIXTURES[index % len(FIXTURES)], covariance)
        samples["generate_frame"].append(perf_counter_ns() - start)
    return samples


def _setup(calibration):
    start = perf_counter_ns()
    covariance = estimate_noise_covariance(calibration, diagonal_loading=DIAGONAL_LOADING)
    after_covariance = perf_counter_ns()
    whitener = whitening_matrix(covariance)
    after_whitener = perf_counter_ns()
    prepared = make_steering(whitener=whitener)
    after_steering = perf_counter_ns()
    return covariance, prepared, {
        "estimate_noise_covariance": after_covariance - start,
        "make_whitener": after_whitener - after_covariance,
        "prepare_steering": after_steering - after_whitener,
        "total": after_steering - start,
    }


def _percentiles(samples):
    median, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {"median_us": median / 1000.0, "p95_us": p95 / 1000.0, "p99_us": p99 / 1000.0}


def _add_setup_rows(raw, summary, path, timings):
    for stage, samples in timings.items():
        keys = {"path": path, "method": "shared", "stage": stage}
        for frame_index, elapsed in enumerate(samples):
            raw.append(
                {
                    **keys,
                    "frame": frame_index,
                    "true_sources": None,
                    "estimated_sources": None,
                    "elapsed_ns": elapsed,
                }
            )
        summary.append({**keys, "subset": "all", "samples": len(samples), **_percentiles(samples)})


def _csv(path, rows):
    def write(output):
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if path.suffix == ".gz":
        with path.open("wb") as file, gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                write(output)
    else:
        with path.open("w", encoding="utf-8", newline="") as output:
            write(output)


def _measure_timings(snapshots, cases, warmup):
    for frame in snapshots[:warmup]:
        for _, method, prepared, scan in cases:
            run_music(frame, prepared, method, scan)
    records = {case[:2]: {stage: [] for stage in (*STAGES, "total")} for case in cases}
    for frame_index, frame in enumerate(snapshots):
        # Rotate case position within each frame to limit first/last-order effects.
        offset = frame_index % len(cases)
        for path, method, prepared, scan in cases[offset:] + cases[:offset]:
            samples = {}
            result = run_music(frame, prepared, method, scan, stage_ns=samples)
            start = perf_counter_ns()
            run_music(frame, prepared, method, scan)
            samples["total"] = perf_counter_ns() - start
            for stage, duration in samples.items():
                records[(path, method)][stage].append((duration, result.source_count))
    raw, summary = [], []
    for (path, method), stages in records.items():
        for stage, samples in stages.items():
            keys = {"path": path, "method": method, "stage": stage}
            for frame_index, (value, count) in enumerate(samples):
                true_count = len(FIXTURES[frame_index % len(FIXTURES)][0])
                raw.append({**keys, "frame": frame_index, "true_sources": true_count, "estimated_sources": count, "elapsed_ns": value})
            subsets = {
                "all": [value for value, _ in samples],
                "sources_present": [value for index, (value, _) in enumerate(samples) if index % len(FIXTURES)],
                "scan_active": [value for value, count in samples if count > 0],
            }
            for subset, values in subsets.items():
                if values:
                    summary.append({**keys, "subset": subset, "samples": len(values), **_percentiles(values)})
    return raw, summary


def _measure_throughput(snapshots, cases, repeats=5):
    rows = []
    for repeat in range(repeats):
        offset = repeat % len(cases)
        for path, method, prepared, scan in cases[offset:] + cases[:offset]:
            start = perf_counter_ns()
            for frame in snapshots:
                run_music(frame, prepared, method, scan)
            elapsed = perf_counter_ns() - start
            rows.append(
                {
                    "path": path,
                    "method": method,
                    "repeat": repeat,
                    "frames": len(snapshots),
                    "elapsed_ns": elapsed,
                    "frames_per_second": len(snapshots) * 1e9 / elapsed,
                }
            )
    return rows


def _measure_memory(snapshots, cases):
    rows = []
    for path, method, prepared, scan in cases:
        tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            for frame in snapshots:
                run_music(frame, prepared, method, scan)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        rows.append(
            {
                "path": path,
                "method": method,
                "frames": len(snapshots),
                "traced_peak_above_baseline_bytes": peak - baseline,
                "traced_live_above_baseline_bytes": current - baseline,
            }
        )
    return rows


def _check_reference(snapshots, cases, noise_covariance):
    rows = []
    # Reuse reference results across scan implementations, with one fixed subset.
    cache = {}
    for path, method, prepared, scan in cases:
        for frame_index, frame in enumerate(snapshots):
            whitened = prepared.whitener is not None
            key = (frame_index, method, whitened)
            if key not in cache:
                cache[key] = reference_music(
                    frame, prepared.angles_deg, noise_covariance if whitened else None, method
                )
            expected = cache[key]
            actual = run_music(frame, prepared, method, scan)
            same_shape = actual.angles_deg.shape == expected.angles_deg.shape
            max_angle_error = (
                float(np.max(np.abs(actual.angles_deg - expected.angles_deg), initial=0.0))
                if same_shape
                else None
            )
            rows.append(
                {
                    "path": path,
                    "method": method,
                    "frame": frame_index,
                    "true_source_count": len(FIXTURES[frame_index % len(FIXTURES)][0]),
                    "actual_source_count": actual.source_count,
                    "reference_source_count": expected.source_count,
                    "angles_match": np.array_equal(actual.angles_deg, expected.angles_deg),
                    "angle_max_abs_error_deg": max_angle_error,
                    "eigenvalue_max_abs_error": float(np.max(np.abs(actual.eigenvalues - expected.eigenvalues))),
                    "denominator_max_abs_error": float(np.max(np.abs(actual.denominator - expected.denominator))),
                }
            )
    return rows


def _cpu_name():
    if platform.system() == "Windows":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            return platform.processor()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
        except OSError:
            pass
    return platform.processor()


def _process_peak_rss_bytes():
    system = platform.system()
    if system == "Windows":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    if system in {"Linux", "Darwin"}:
        import resource

        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maximum * 1024 if system == "Linux" else maximum
    return None


def profile(output_dir, frames=400, warmup=32, threads=1, seed=20261116, reference_frames=16):
    """Write fixed-input timing, memory and reference data to a selected directory."""
    if frames < 1 or warmup < 0 or threads < 1 or seed < 0 or reference_frames < 1:
        raise ValueError("provide positive frame, thread and reference counts and nonnegative warmup and seed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=threads, user_api="blas"):
        calibration, snapshots = _fixtures(seed, frames)
        noise_covariance, whitened, _ = _setup(calibration)
        preparations = {"raw": make_steering(), "whitened": whitened}
        cases = [(path, method, preparations[space], scan) for path, space, scan in PATHS for method in METHODS]
        stage_rows, stage_summary = _measure_timings(snapshots, cases, warmup)
        setup_timings = {}
        for _ in range(frames):
            _, _, stage_ns = _setup(calibration)
            for stage, elapsed in stage_ns.items():
                setup_timings.setdefault(stage, []).append(elapsed)
        _add_setup_rows(stage_rows, stage_summary, "calibration_setup", setup_timings)
        _add_setup_rows(stage_rows, stage_summary, "data_generation", _generation_timings(seed, frames))
        throughput = _measure_throughput(snapshots, cases)
        memory = _measure_memory(snapshots, cases)
        reference = _check_reference(snapshots[:reference_frames], cases, noise_covariance)
        blas = threadpool_info()

    _csv(output_dir / "stage_samples.csv.gz", stage_rows)
    _csv(output_dir / "stage_summary.csv", stage_summary)
    _csv(output_dir / "throughput.csv", throughput)
    _csv(output_dir / "memory.csv", memory)
    _csv(output_dir / "numerical.csv", reference)
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_threads_requested": threads,
        "threadpools_during_benchmark": [
            {key: value for key, value in pool.items() if key != "filepath"} for pool in blas
        ],
        "clock": "perf_counter_ns, wall clock; stage instrumentation is excluded from the separately measured total and throughput",
        "warmup_frames_per_case": min(warmup, frames),
        "frames_per_case": frames,
        "reference_frames_per_case": min(reference_frames, frames),
        "seed": seed,
        "snapshot_sha256": hashlib.sha256(snapshots.astype("<c16", copy=False).tobytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration.astype("<c16", copy=False).tobytes()).hexdigest(),
        "shape_per_frame": list(snapshots.shape[1:]),
        "noise_rho": NOISE_RHO,
        "calibration_samples": CALIBRATION_SAMPLES,
        "diagonal_loading": DIAGONAL_LOADING,
        "fixtures_cycled_in_order": [
            {"angles_deg": angles, "strongest_snr_db": snr, "power_spread_db": spread}
            for angles, snr, spread in FIXTURES
        ],
        "process_lifetime_peak_rss_bytes": _process_peak_rss_bytes(),
        "memory_scope": (
            "Per-case tracemalloc peaks track Python allocations and NumPy allocations exposed to tracemalloc; "
            "they exclude the preallocated inputs/steering and untraced native BLAS allocations. "
            "The separate OS process lifetime peak includes all benchmark cases, preallocated data and output assembly; "
            "it cannot be attributed to individual stages."
        ),
        "timing_scope": (
            "Fixed synthetic inputs and steering are prepared before each frame timing. "
            "Calibration is timed separately on a fixed noise block. Total includes allocation of per-frame outputs. "
            "Throughput is sequential processing, excludes calibration, has no I/O, and is not FPGA or real-time throughput."
        ),
        "comparison_scope": (
            "The exact-eigensolver AMD 5% proxy maps an all-eight count to zero. "
            "This program does not reproduce the finite-sweep AMD SVD or its hardware peak scanner. "
            "Dense and projected scans are checked against SVD/triangular solves and scalar dot products. "
            "MDL and mdl_scalar use the same criterion over orders 0 through 3; the latter uses scalar arithmetic and caches its fixed penalty."
        ),
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--reference-frames", type=int, default=16)
    args = parser.parse_args()
    try:
        environment = profile(args.output_dir, args.frames, args.warmup, args.threads, args.seed, args.reference_frames)
    except ValueError as error:
        parser.error(str(error))
    print(f"Profiled {environment['frames_per_case']} frames per case on {environment['cpu']} into {args.output_dir}")


if __name__ == "__main__":
    main()
