"""Plot the measured stage latencies from the CPU profiling output."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "whiten_snapshots": "Whitening",
    "covariance": "Covariance",
    "eigendecomposition": "Eigendecomposition",
    "count_sources": "Source count (MDL)",
    "construct_noise_subspace": "Noise subspace",
    "scan_spectrum": "Spectrum",
    "pick_angles": "Peak selection",
    "total": "End-to-end",
}


def plot(output):
    with (output / "stage_summary.csv").open(encoding="utf-8", newline="") as file:
        rows = {
            (row["method"], row["stage"]): row
            for row in csv.DictReader(file)
            if row["path"] == "whitened_projected" and row["subset"] == "all"
        }
    positions = np.arange(len(LABELS))
    fig, axis = plt.subplots(figsize=(9, 5.1), layout="constrained")
    for offset, method, label, color in (
        (-0.16, "mdl", "Array-based MDL", "#52667a"),
        (0.16, "mdl_scalar", "Scalar MDL", "#087e75"),
    ):
        samples = [rows[method, stage] for stage in LABELS]
        medians = np.array([float(sample["median_us"]) for sample in samples])
        p95 = np.array([float(sample["p95_us"]) for sample in samples])
        axis.barh(positions + offset, medians, 0.29, color=color, label=label)
        axis.hlines(positions + offset, medians, p95, color=color, linewidth=1.3)
        axis.plot(p95, positions + offset, "|", color=color, markersize=6)
    axis.set_yticks(positions, list(LABELS.values()))
    axis.invert_yaxis()
    axis.set_xlabel("Wall-clock latency per frame (µs)")
    axis.set_title("CPU MUSIC with whitening and projected spectrum\nMedian bars and p95 endpoints", loc="left")
    axis.legend(loc="lower right", frameon=False)
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.savefig(output / "latency.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
