"""Run the preregistered paired test of independent noise whitening."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.linalg import solve_triangular

from threadpoolctl import threadpool_limits

from .pilot import (
    SENSORS,
    SNAPSHOTS,
    adaptive_counts,
    amd_threshold_proxy,
    complex_normal,
    draw_angles,
)


SEED_PAIRS = ((20261115, 20261215), (20261116, 20261216), (20261117, 20261217))
SCENARIOS = (("white", 0.0, 0.0), ("correlated", 0.2, 0.2), ("drift", 0.2, 0.35))
CELLS = (
    (0, None, None), (1, -5, 0), (1, 0, 0), (1, 5, 0), (2, 0, 12),
    (2, 5, 0), (2, 5, 12), (3, 5, 0), (3, 10, 12),
)
CALIBRATIONS = 24
FRAMES = 24
CALIBRATION_SAMPLES = 2048
METHODS = ("amd_original", "amd_capped", "mdl_original", "mdl_capped", "float", "q12")
BRANCHES = ("raw", "white")
ORDERS = np.arange(SENSORS)
TAIL_SIZES = SENSORS - ORDERS
PENALTIES = 0.5 * ORDERS * (2 * SENSORS - ORDERS) * np.log(SNAPSHOTS)


def noise_factor(rho):
    sensors = np.arange(SENSORS)
    return np.linalg.cholesky(rho ** np.abs(sensors[:, None] - sensors))


def calibrate(noise):
    covariance = noise.T @ noise.conj() / len(noise)
    loading = 0.001 * covariance.trace().real / SENSORS
    return np.linalg.cholesky(covariance + loading * np.eye(SENSORS))


def whiten(snapshots, cholesky):
    return solve_triangular(cholesky, snapshots.T, lower=True, check_finite=False).T


def test_frame(rng, factor, cell):
    count, strongest_snr, spread = cell
    noise = complex_normal(rng, (SNAPSHOTS, SENSORS)) @ factor.T
    if count == 0:
        return noise
    angles = draw_angles(rng, count)
    steering = np.exp(1j * np.pi * np.arange(SENSORS)[:, None] * np.sin(np.deg2rad(angles)))
    amplitude = 10 ** ((strongest_snr - np.linspace(0, spread, count)) / 20)
    return noise + (complex_normal(rng, (SNAPSHOTS, count)) * amplitude) @ steering.T


def decisions(snapshots):
    covariance = snapshots.T @ snapshots.conj() / SNAPSHOTS
    values = np.linalg.eigvalsh(covariance)[::-1]
    amd = amd_threshold_proxy(values)
    positive = np.maximum(values, np.finfo(values.dtype).tiny)
    tail_sums = np.cumsum(positive[::-1])[::-1]
    tail_log_sums = np.cumsum(np.log(positive)[::-1])[::-1]
    scores = SNAPSHOTS * (TAIL_SIZES * np.log(tail_sums / TAIL_SIZES) - tail_log_sums)
    scores[-1] = 0.0
    scores += PENALTIES
    floating, q12 = adaptive_counts(values)
    return amd, min(amd, 3), int(np.argmin(scores)), int(np.argmin(scores[:4])), floating, q12


def collect(path):
    shape = (len(SCENARIOS), len(SEED_PAIRS), CALIBRATIONS, len(CELLS), FRAMES, 2, len(METHODS))
    output = np.empty(shape, dtype=np.uint8)
    with path.open("wb") as file, gzip.GzipFile(fileobj=file, filename="", mode="wb", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["scenario", "test_seed", "calibration_seed", "calibration", "sources",
                 "strongest_snr_db", "power_spread_db", "frame"]
                + [f"{branch}_{method}" for branch in BRANCHES for method in METHODS]
            )
            for seed_index, (test_seed, cal_seed) in enumerate(SEED_PAIRS):
                test_streams = np.random.SeedSequence(test_seed).spawn(len(SCENARIOS))
                cal_streams = np.random.SeedSequence(cal_seed).spawn(len(SCENARIOS))
                for scenario, (name, cal_rho, test_rho) in enumerate(SCENARIOS):
                    test_rng = np.random.default_rng(test_streams[scenario])
                    cal_rng = np.random.default_rng(cal_streams[scenario])
                    cal_factor, test_factor = noise_factor(cal_rho), noise_factor(test_rho)
                    for calibration in range(CALIBRATIONS):
                        noise = complex_normal(cal_rng, (CALIBRATION_SAMPLES, SENSORS)) @ cal_factor.T
                        transform = calibrate(noise)
                        for cell_index, cell in enumerate(CELLS):
                            for frame in range(FRAMES):
                                raw = test_frame(test_rng, test_factor, cell)
                                before, after = decisions(raw), decisions(whiten(raw, transform))
                                output[scenario, seed_index, calibration, cell_index, frame] = before, after
                                writer.writerow((name, test_seed, cal_seed, calibration, *cell, frame, *before, *after))
    return output


def interval(clusters, draws, bounded_rate=False):
    """Clusters are seed by calibration, with within-cluster means."""
    estimate = clusters.mean()
    if bounded_rate and estimate in (0, 1):
        # The probability a cluster has any event bounds its mean event fraction.
        upper = 1 - 0.025 ** (1 / clusters.size)
        return (0.0, upper) if estimate == 0 else (1 - upper, 1.0)
    samples = clusters[np.arange(clusters.shape[0])[None, :, None], draws].mean(axis=(1, 2))
    return tuple(np.quantile(samples, (0.025, 0.975)))


def summarize(data, path):
    rng = np.random.default_rng(20261299)
    draws = rng.integers(CALIBRATIONS, size=(4000, len(SEED_PAIRS), CALIBRATIONS))
    rows = []
    total_frames = len(SEED_PAIRS) * CALIBRATIONS * FRAMES

    def record(scenario, cell, metric, method, branch, events, paired=False):
        lower, upper = interval(events.mean(axis=-1), draws, bounded_rate=not paired)
        if paired and np.all(events == events.flat[0]):
            # Bootstrap cannot represent unobserved discordances. Bound their
            # contribution using the chance of any change in a calibration.
            value = int(events.flat[0])
            rare_event_upper = 1 - 0.025 ** (1 / (len(SEED_PAIRS) * CALIBRATIONS))
            lower = value - (value + 1) * rare_event_upper
            upper = value + (1 - value) * rare_event_upper
        rows.append({
            "scenario": scenario, "sources": cell[0], "strongest_snr_db": cell[1],
            "power_spread_db": cell[2], "metric": metric, "method": method, "branch": branch,
            "frames": total_frames, "count": int(events.sum()),
            "positive": int((events > 0).sum()), "negative": int((events < 0).sum()),
            "estimate": events.mean(), "ci95_low": lower, "ci95_high": upper,
        })

    truth = np.array([cell[0] for cell in CELLS]).reshape(1, 1, len(CELLS), 1, 1, 1)
    for scenario, (name, _, _) in enumerate(SCENARIOS):
        correct = data[scenario] == truth
        for index, cell in enumerate(CELLS):
            for branch, label in enumerate(BRANCHES):
                values = data[scenario, :, :, index, :, branch]
                for method, method_name in enumerate(METHODS):
                    record(name, cell, "accuracy", method_name, label, values[..., method] == cell[0])
                    if cell[0] == 0:
                        record(name, cell, "false_alarm", method_name, label, values[..., method] > 0)
                record(name, cell, "agreement", "q12_float", label, values[..., 4] == values[..., 5])
                right = correct[:, :, index, :, branch].astype(np.int8)
                for comparator in (1, 3):
                    record(name, cell, "accuracy_delta", f"q12_minus_{METHODS[comparator]}",
                           label, right[..., 5] - right[..., comparator], paired=True)
            q12 = correct[:, :, index, :, :, 5].astype(np.int8)
            record(name, cell, "accuracy_delta", "q12_white_minus_raw", "paired",
                   q12[..., 1] - q12[..., 0], paired=True)

    agreement = (data[..., 1, 4] == data[..., 1, 5]).transpose(1, 0, 2, 3, 4)
    # Calibrations across scenarios are independent. Sample each scenario and
    # seed pair as its own stratum when calculating the two aggregate intervals.
    aggregate_draws = rng.integers(CALIBRATIONS, size=(4000, 9, CALIBRATIONS))
    truth_all = np.array([cell[0] for cell in CELLS[1:]]).reshape(1, 1, 1, -1, 1)
    correct_q12 = data[:, :, :, 1:, :, 1, 5] == truth_all
    correct_mdl = data[:, :, :, 1:, :, 1, 3] == truth_all
    paired = correct_q12.astype(np.int8) - correct_mdl
    overall = {}
    for name, values in (("agreement", agreement), ("q12_minus_mdl", paired)):
        cluster_values = values.mean(axis=(-2, -1)).reshape(9, CALIBRATIONS)
        low, high = interval(cluster_values, aggregate_draws, bounded_rate=name == "agreement")
        overall[name] = {
            "count": int(values.sum()), "frames": values.size, "estimate": values.mean(),
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum()),
            "ci95_low": low, "ci95_high": high,
        }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, overall


def benchmark():
    rng = np.random.default_rng(20261301)
    cal = complex_normal(rng, (CALIBRATION_SAMPLES, SENSORS)) @ noise_factor(0.2).T
    frame = complex_normal(rng, (SNAPSHOTS, SENSORS))
    factor = calibrate(cal)
    results = {}
    for name, operation in (("calibrate", lambda: calibrate(cal)), ("whiten", lambda: whiten(frame, factor))):
        for _ in range(30):
            operation()
        elapsed = []
        for _ in range(400):
            start = perf_counter_ns()
            operation()
            elapsed.append(perf_counter_ns() - start)
        results[name] = {
            "median_us": np.median(elapsed) / 1000, "p95_us": np.percentile(elapsed, 95) / 1000,
            "samples_ns": elapsed,
        }
    return results


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("Matched white", "white", "accuracy", (2, 5, 12)),
        ("Matched correlated", "correlated", "accuracy", (2, 5, 12)),
        ("Noise only", None, "false_alarm", (0, None, None)),
    )
    labels = {"white": "White", "correlated": "Correlated", "drift": "Drift"}
    methods = (("amd_capped", "AMD 0–3"), ("mdl_capped", "MDL 0–3"), ("q12", "Q12"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True, layout="constrained")
    for axis, (title, scenario, metric, cell) in zip(axes, panels):
        groups = methods if scenario else tuple((name, label) for name, label in labels.items())
        for index, branch in enumerate(BRANCHES):
            x = np.arange(len(groups)) + (index - 0.5) * 0.33
            values = []
            for item, _ in groups:
                group, method = (scenario, item) if scenario else (item, "q12")
                values.append(next(row for row in rows if
                    (row["scenario"], row["sources"], row["strongest_snr_db"], row["power_spread_db"],
                     row["metric"], row["method"], row["branch"]) == (group, *cell, metric, method, branch)))
            heights = np.array([row["estimate"] for row in values]) * 100
            low = np.array([row["ci95_low"] for row in values]) * 100
            high = np.array([row["ci95_high"] for row in values]) * 100
            axis.bar(x, heights, 0.31, yerr=np.maximum(0, (heights - low, high - heights)),
                     color=("#52667a", "#087e75")[index], capsize=2,
                     label=("Before whitening", "After whitening")[index])
        axis.set_xticks(np.arange(len(groups)), [label for _, label in groups])
        axis.set_title(title + ("\n2 sources, +5 dB, 12 dB spread" if scenario else "\nQ12 false alarms"))
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.24)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Exact count accuracy (%)")
    axes[2].set_ylabel("False alarm rate (%)")
    axes[2].axhline(5, color="#a73528", linestyle="--", linewidth=1)
    axes[2].legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def reproduce(output):
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api="blas"):
        data = collect(output / "trials.csv.gz")
        rows, overall = summarize(data, output / "cells.csv")
        timings = benchmark()
    plot(rows, output / "whitening.png")
    (output / "metrics.json").write_text(
        json.dumps({"overall": overall, "cpu_timing": timings}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {data[..., 0, 0].size:,} paired trials across {len(CELLS) * len(SCENARIOS)} cells")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    reproduce(parser.parse_args().out)


if __name__ == "__main__":
    main()
