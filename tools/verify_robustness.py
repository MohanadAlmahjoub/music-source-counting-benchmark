"""Independently reproduce the synthetic source-count decisions and cell totals."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits


SENSORS = 8
SNAPSHOTS = 128
CALIBRATION_SAMPLES = 2048
CALIBRATIONS = 24
FRAMES = 24
SEED_PAIRS = ((20261115, 20261215), (20261116, 20261216), (20261117, 20261217))
SCENARIOS = (("white", 0.0, 0.0), ("correlated", 0.2, 0.2), ("drift", 0.2, 0.35))
CELLS = (
    (0, None, None), (1, -5, 0), (1, 0, 0), (1, 5, 0), (2, 0, 12),
    (2, 5, 0), (2, 5, 12), (3, 5, 0), (3, 10, 12),
)
METHODS = ("amd_original", "amd_capped", "mdl_original", "mdl_capped", "float", "q12")
BRANCHES = ("raw", "white")
INDEX = np.arange(SENSORS)
DISTANCE = np.abs(INDEX[:, None] - INDEX[None, :])


def normal(rng, shape):
    """Draw unit-power complex Gaussian samples in the fixture RNG order."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def rank_estimates(values):
    """Use geometric means and the quantized median as independent count formulas."""
    count = int(np.sum(values / values[0] > 0.05))
    count = count if count != SENSORS else 0
    mdl = []
    for order in range(SENSORS):
        tail = values[order:]
        geometric = np.exp(np.log(tail).mean())
        likelihood = -SNAPSHOTS * len(tail) * np.log(geometric / tail.mean())
        penalty = order * (2 * SENSORS - order) * np.log(SNAPSHOTS) / 2
        mdl.append(likelihood + penalty)
    float_count = np.count_nonzero(values[:3] / np.median(values[4:]) > 2.25)
    quantized = np.rint(values / values[0] * 4095)
    fixed_count = np.count_nonzero(quantized[:3] > 2.25 * np.median(quantized[4:]))
    return count, min(count, 3), int(np.argmin(mdl)), int(np.argmin(mdl[:4])), float_count, fixed_count


def sensor_data(rng, factor, cell):
    """Generate one synthetic array frame as sensor-by-time columns."""
    count, snr, spread = cell
    noise = normal(rng, (SNAPSHOTS, SENSORS)) @ factor.T
    if count == 0:
        return noise.T
    while True:
        angles = np.sort(rng.uniform(-60, 60, count))
        if np.all(np.diff(angles) >= 6):
            break
    steering = np.exp(1j * np.pi * INDEX[:, None] * np.sin(np.deg2rad(angles)))
    power_db = snr - np.linspace(0, spread, count)
    amplitude = np.sqrt(10 ** (power_db / 10.0))
    signal = (normal(rng, (SNAPSHOTS, count)) * amplitude) @ steering.T
    return (noise + signal).T


def expected_frames():
    """Rebuild covariance matrices and whiten the covariance directly."""
    for test_seed, cal_seed in SEED_PAIRS:
        test_streams = np.random.SeedSequence(test_seed).spawn(len(SCENARIOS))
        cal_streams = np.random.SeedSequence(cal_seed).spawn(len(SCENARIOS))
        for scenario, test_stream, cal_stream in zip(SCENARIOS, test_streams, cal_streams, strict=True):
            name, cal_rho, test_rho = scenario
            test_rng, cal_rng = np.random.default_rng(test_stream), np.random.default_rng(cal_stream)
            cal_factor = np.linalg.cholesky(cal_rho ** DISTANCE)
            test_factor = np.linalg.cholesky(test_rho ** DISTANCE)
            for cal_id in range(CALIBRATIONS):
                calibration = (normal(cal_rng, (CALIBRATION_SAMPLES, SENSORS)) @ cal_factor.T).T
                cov = calibration @ calibration.conj().T / CALIBRATION_SAMPLES
                loading = 0.001 * np.trace(cov).real / SENSORS
                factor = np.linalg.cholesky(cov + np.eye(SENSORS) * loading)
                for cell in CELLS:
                    for frame in range(FRAMES):
                        data = sensor_data(test_rng, test_factor, cell)
                        raw = data @ data.conj().T / SNAPSHOTS
                        left_solved = solve_triangular(factor, raw, lower=True)
                        white = solve_triangular(factor, left_solved.conj().T, lower=True).conj().T
                        decisions = []
                        for matrix in (raw, white):
                            symmetric = (matrix + matrix.conj().T) / 2
                            values = np.linalg.eigvalsh(symmetric)[::-1]
                            decisions.extend(rank_estimates(values))
                        yield (name, test_seed, cal_seed, cal_id, *cell, frame), tuple(decisions)


def text_key(fields):
    return tuple("" if value is None else str(value) for value in fields)


def collect_totals(totals, cell, decisions):
    correctness = {}
    for branch, predictions in zip(BRANCHES, (decisions[:len(METHODS)], decisions[len(METHODS):]), strict=True):
        source_count = int(cell[1])
        right = {method: int(pred == source_count) for method, pred in zip(METHODS, predictions, strict=True)}
        correctness[branch] = right["q12"]
        for method, pred in zip(METHODS, predictions, strict=True):
            totals[(*cell, "accuracy", method, branch)][right[method]] += 1
            if source_count == 0:
                totals[(*cell, "false_alarm", method, branch)][int(pred > 0)] += 1
        totals[(*cell, "agreement", "q12_float", branch)][int(predictions[-2] == predictions[-1])] += 1
        for comparator in ("amd_capped", "mdl_capped"):
            event = right["q12"] - right[comparator]
            totals[(*cell, "accuracy_delta", f"q12_minus_{comparator}", branch)][event] += 1
    difference = correctness["white"] - correctness["raw"]
    totals[(*cell, "accuracy_delta", "q12_white_minus_raw", "paired")][difference] += 1


def verify(fixtures):
    identifiers = ("scenario", "test_seed", "calibration_seed", "calibration", "sources",
                   "strongest_snr_db", "power_spread_db", "frame")
    decision_columns = [f"{branch}_{method}" for branch in BRANCHES for method in METHODS]
    totals = defaultdict(Counter)
    with gzip.open(fixtures / "trials.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for frame_total, (row, (key, inferred)) in enumerate(zip(reader, expected_frames(), strict=True), 1):
            actual_key = tuple(row[field] for field in identifiers)
            if actual_key != text_key(key):
                raise ValueError(f"frame {frame_total} has an unexpected identity {actual_key}")
            observed = tuple(int(row[column]) for column in decision_columns)
            if observed != inferred:
                raise ValueError(f"frame {key} has decisions {observed}, expected {inferred}")
            cell = text_key((key[0], *key[4:7]))
            collect_totals(totals, cell, inferred)

    columns = ("scenario", "sources", "strongest_snr_db", "power_spread_db", "metric", "method", "branch")
    with (fixtures / "cells.csv").open(newline="", encoding="utf-8") as stream:
        for cell_total, row in enumerate(csv.DictReader(stream), 1):
            key = tuple(row[field] for field in columns)
            events = totals.pop(key)
            frames = sum(events.values())
            net = events[1] - events[-1]
            observed = tuple(int(row[field]) for field in ("frames", "count", "positive", "negative"))
            if observed != (frames, net, events[1], events[-1]):
                raise ValueError(f"incorrect cell totals for {key}")
            if not np.isclose(float(row["estimate"]), net / frames, rtol=0, atol=1e-15):
                raise ValueError(f"incorrect cell estimate for {key}")
    if totals:
        raise ValueError(f"missing {len(totals)} cell rows")
    return frame_total, frame_total * len(decision_columns), cell_total


def main():
    default_fixtures = Path(__file__).resolve().parents[1] / "results" / "robustness"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=default_fixtures, help="directory containing trials.csv.gz and cells.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        frames, decisions, summaries = verify(args.fixtures)
    print(f"Verified {frames:,} paired frames, {decisions:,} count decisions, and {summaries:,} cell summaries")


if __name__ == "__main__":
    main()
