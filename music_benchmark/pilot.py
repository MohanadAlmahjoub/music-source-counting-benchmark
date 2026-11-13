"""Compare three source-count rules on synthetic uniform linear array snapshots."""

import argparse
import csv
from pathlib import Path

import numpy as np


SENSORS = 8
SNAPSHOTS = 128
QUANTIZATION_MAX = 4095


def complex_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def draw_angles(rng, count):
    while True:
        angles = np.sort(rng.uniform(-60, 60, count))
        if np.all(np.diff(angles) >= 6):
            return angles


def eigenvalues_for_trial(rng, count, strongest_snr_db, spread_db, noise_factor):
    snapshots = complex_normal(rng, (SNAPSHOTS, SENSORS)) @ noise_factor.T
    if count:
        angles = draw_angles(rng, count)
        steering = np.exp(
            1j * np.pi * np.arange(SENSORS)[:, None] * np.sin(np.deg2rad(angles))
        )
        offsets = np.linspace(0, spread_db, count)
        amplitudes = 10 ** ((strongest_snr_db - offsets) / 20)
        snapshots += (complex_normal(rng, (SNAPSHOTS, count)) * amplitudes) @ steering.T
    covariance = snapshots.conj().T @ snapshots / SNAPSHOTS
    return np.linalg.eigvalsh(covariance)[::-1]


def amd_threshold_proxy(values):
    raw_count = np.count_nonzero(values > 0.05 * values[0])
    return 0 if raw_count == SENSORS else raw_count


def mdl_count(values):
    values = np.maximum(values, np.finfo(values.dtype).tiny)
    scores = []
    for count in range(SENSORS):
        noise = values[count:]
        likelihood = SNAPSHOTS * len(noise) * (np.log(noise.mean()) - np.log(noise).mean())
        penalty = 0.5 * count * (2 * SENSORS - count) * np.log(SNAPSHOTS)
        scores.append(likelihood + penalty)
    return int(np.argmin(scores))


def adaptive_counts(values):
    float_count = np.count_nonzero(values[:3] > 2.25 * np.median(values[-4:]))
    codes = np.rint(QUANTIZATION_MAX * values / values[0]).astype(np.int32)
    integer_count = np.count_nonzero(8 * codes[:3] > 9 * (codes[5] + codes[6]))
    return float_count, integer_count


def run_grid(seed, trials):
    rng = np.random.default_rng(seed)
    sensor_distance = np.abs(np.arange(SENSORS)[:, None] - np.arange(SENSORS))
    for noise_rho in (0.0, 0.2):
        noise_factor = np.linalg.cholesky(noise_rho ** sensor_distance)
        for count in range(4):
            conditions = [(None, None)]
            if count:
                spreads = (0,) if count == 1 else (0, 12)
                conditions = [
                    (snr, spread) for snr in (-5, 0, 5, 10, 15) for spread in spreads
                ]
            for snr, spread in conditions:
                totals = np.zeros(6, dtype=np.int32)
                for _ in range(trials):
                    values = eigenvalues_for_trial(rng, count, snr, spread, noise_factor)
                    amd = amd_threshold_proxy(values)
                    adaptive_float, adaptive_q12 = adaptive_counts(values)
                    totals += (
                        amd == count,
                        min(amd, 3) == count,
                        mdl_count(values) == count,
                        adaptive_float == count,
                        adaptive_q12 == count,
                        adaptive_float == adaptive_q12,
                    )
                yield {
                    "noise_rho": noise_rho,
                    "true_sources": count,
                    "strongest_snr_db": snr,
                    "source_power_spread_db": spread,
                    "trials": trials,
                    "amd_proxy_correct": totals[0],
                    "amd_proxy_capped_correct": totals[1],
                    "mdl_correct": totals[2],
                    "adaptive_float_correct": totals[3],
                    "adaptive_q12_correct": totals[4],
                    "adaptive_agreement": totals[5],
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("pilot-results.csv"))
    args = parser.parse_args()
    if args.seed < 0 or args.trials < 1:
        parser.error("provide a nonnegative seed and at least one trial per condition")

    rows = list(run_grid(args.seed, args.trials))
    with args.out.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} conditions and {len(rows) * args.trials} trials to {args.out}")


if __name__ == "__main__":
    main()
