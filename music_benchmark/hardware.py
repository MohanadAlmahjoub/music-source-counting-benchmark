"""Measure fixed-point error and model two kernels in the MDL/MUSIC path."""

import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .pipeline import (
    SENSORS,
    SNAPSHOTS,
    count_sources,
    estimate_noise_covariance,
    generate_snapshot,
    make_steering,
    pick_angles,
    run_music,
    scan_spectrum,
    whitening_matrix,
)


WORD_LENGTHS = (4, 6, 8, 10, 12, 14, 16)
FULL_SCALES = (4, 8, 16)
SEEDS = (20261121, 20261122)
SCENARIOS = (("white", 0.0, 0.0), ("correlated", 0.2, 0.2), ("drift", 0.2, 0.35))
CELLS = (
    ((), None, 0),
    ((-17.5,), 0, 0),
    ((-24.5, 18.5), 5, 12),
    ((-40.0, 5.0, 42.0), 10, 12),
    ((-32.0, -24.0, 17.0), 15, 12),
)


def noise_covariance(rho):
    sensors = np.arange(SENSORS)
    return rho ** np.abs(sensors[:, None] - sensors)


def quantize_iq(values, bits, full_scale):
    """Round each component to a signed, symmetric code with nearest-even ties."""
    maximum = (1 << (bits - 1)) - 1
    scale = maximum / full_scale
    real = np.rint(np.clip(values.real, -full_scale, full_scale) * scale).astype(np.int64)
    imag = np.rint(np.clip(values.imag, -full_scale, full_scale) * scale).astype(np.int64)
    return real, imag, scale


def quantized_covariance(snapshot, bits, full_scale):
    """Accumulate unscaled complex covariance in integer codes, then convert."""
    real, imag, scale = quantize_iq(snapshot, bits, full_scale)
    real_products = real.T @ real + imag.T @ imag
    imag_products = imag.T @ real - real.T @ imag
    return (real_products + 1j * imag_products) / (len(snapshot) * scale**2)


def quantized_spectrum(vectors, steering_codes, count, bits):
    """Compute MUSIC denominators from unit-vector coefficients in integer codes."""
    noise_vectors = vectors[:, : SENSORS - count]
    real, imag, scale = quantize_iq(noise_vectors, bits, 1)
    steering_real, steering_imag = steering_codes
    projection_real = real.T @ steering_real + imag.T @ steering_imag
    projection_imag = real.T @ steering_imag - imag.T @ steering_real
    return (projection_real**2 + projection_imag**2).sum(axis=0) / scale**4


def add_trial(metrics, count, angles, reference, truth, covariance_error=0.0, clipped=0):
    count_matches = count == reference.source_count
    peaks_match = np.array_equal(angles, reference.angles_deg)
    metrics["count_agreement"] += count_matches
    metrics["exact_peak_agreement"] += peaks_match
    metrics["joint_count_peak_agreement"] += count_matches and peaks_match
    if count == len(truth) and angles.size == len(truth):
        metrics["joint_truth_within_2deg"] += np.all(np.abs(angles - truth) <= 2)
    metrics["clipped_components"] += clipped
    metrics["covariance_relative_errors"].append(covariance_error)
    if count_matches and angles.size == reference.angles_deg.size and angles.size:
        metrics["peak_displacements_deg"].extend(np.abs(angles - reference.angles_deg))


def finish_metrics(metrics):
    covariance_errors = metrics.pop("covariance_relative_errors")
    peak_displacements = metrics.pop("peak_displacements_deg")
    metrics["covariance_relative_error_p95"] = np.percentile(covariance_errors, 95)
    metrics["paired_peaks"] = len(peak_displacements)
    metrics["peak_displacement_p95_deg"] = (
        np.percentile(peak_displacements, 95) if peak_displacements else ""
    )


def measure_accuracy(trials):
    configs = [("float64", 64, None, None)]
    configs += [("iq_covariance", bits, scale, None) for bits in WORD_LENGTHS for scale in FULL_SCALES]
    configs += [("projection", None, None, bits) for bits in WORD_LENGTHS]
    configs += [("iq_and_projection", 12, 16, 14)]
    for scenario, calibration_rho, test_rho in SCENARIOS:
        calibration_covariance = noise_covariance(calibration_rho)
        evaluation_covariance = noise_covariance(test_rho)
        for cell_index, (truth, snr, spread) in enumerate(CELLS):
            metrics = {
                config: dict(
                    count_agreement=0,
                    exact_peak_agreement=0,
                    joint_count_peak_agreement=0,
                    joint_truth_within_2deg=0,
                    clipped_components=0,
                    covariance_relative_errors=[],
                    peak_displacements_deg=[],
                )
                for config in configs
            }
            for seed in SEEDS:
                calibration_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
                samples = generate_snapshot(
                    calibration_rng, (), None, noise_covariance=calibration_covariance,
                    snapshot_count=2048,
                )
                whitening = whitening_matrix(estimate_noise_covariance(samples, 0.001))
                prepared = make_steering(whitener=whitening)
                steering_codes = {
                    bits: quantize_iq(prepared.steering, bits, 1)[:2] for bits in WORD_LENGTHS
                }
                test_rng = np.random.default_rng(np.random.SeedSequence([seed, cell_index, 2]))
                for _ in range(trials):
                    snapshot = generate_snapshot(test_rng, truth, snr, spread, evaluation_covariance)
                    reference = run_music(snapshot, prepared)
                    add_trial(metrics["float64", 64, None, None], reference.source_count,
                              reference.angles_deg, reference, truth)
                    whitened = snapshot @ whitening.T
                    covariance = whitened.T @ whitened.conj() / len(whitened)
                    covariance_norm = np.linalg.norm(covariance)
                    _, vectors = np.linalg.eigh(covariance)
                    for bits in WORD_LENGTHS:
                        denominator = quantized_spectrum(
                            vectors, steering_codes[bits], reference.source_count, bits,
                        )
                        angles = pick_angles(denominator, prepared.angles_deg, reference.source_count)
                        add_trial(metrics["projection", None, None, bits], reference.source_count,
                                  angles, reference, truth)
                    for full_scale in FULL_SCALES:
                        clipped = (
                            np.count_nonzero(np.abs(snapshot.real) > full_scale)
                            + np.count_nonzero(np.abs(snapshot.imag) > full_scale)
                        )
                        for bits in WORD_LENGTHS:
                            fixed = quantized_covariance(snapshot, bits, full_scale)
                            fixed = whitening @ fixed @ whitening.conj().T
                            values, vectors = np.linalg.eigh(fixed)
                            count = count_sources(values[::-1])
                            denominator = scan_spectrum(vectors, prepared, count)
                            angles = pick_angles(denominator, prepared.angles_deg, count)
                            error = np.linalg.norm(fixed - covariance) / covariance_norm
                            add_trial(metrics["iq_covariance", bits, full_scale, None], count, angles,
                                      reference, truth, error, clipped)
                            if bits == 12 and full_scale == 16:
                                fixed_denominator = quantized_spectrum(
                                    vectors, steering_codes[14], count, 14,
                                )
                                fixed_angles = pick_angles(
                                    fixed_denominator, prepared.angles_deg, count,
                                )
                                add_trial(metrics["iq_and_projection", bits, full_scale, 14],
                                          count, fixed_angles, reference, truth, error, clipped)
            for (kernel, iq_bits, scale, projection_bits), row in metrics.items():
                finish_metrics(row)
                yield {
                    "scenario": scenario, "sources": len(truth), "strongest_snr_db": snr,
                    "power_spread_db": spread, "kernel": kernel, "iq_bits": iq_bits,
                    "iq_full_scale": scale, "projection_bits": projection_bits,
                    "frames": len(SEEDS) * trials,
                    "input_components": len(SEEDS) * trials * SNAPSHOTS * SENSORS * 2,
                    **row,
                }


def model_architectures():
    clock_mhz = 200
    grid_bins = make_steering().angles_deg.size
    covariance_products = SENSORS * 2 + (SENSORS * (SENSORS - 1) // 2) * 4
    cases = (
        ("covariance_8x128", 12, SNAPSHOTS * covariance_products, 3072, 256, 0,
         (8, 32, 64, 128)),
        ("projection_1src_240bins", 14, grid_bins * 7 * (4 * SENSORS + 2), 196, 16, 6720,
         (32, 64, 128, 256, 288)),
        ("projection_2src_240bins", 14, grid_bins * 6 * (4 * SENSORS + 2), 168, 16, 6720,
         (32, 64, 128, 256)),
    )
    for name, bits, real_products, input_bytes, output_bytes, coefficient_bytes, lane_counts in cases:
        for lanes in lane_counts:
            clock_ns = 1000 / clock_mhz
            compute_cycles = (real_products + lanes - 1) // lanes
            ingress_cycles = (input_bytes + 7) // 8
            egress_cycles = (output_bytes + 7) // 8
            yield {
                "kernel": name, "iq_bits": bits, "assumed_clock_mhz": clock_mhz,
                "assumed_stream_bits": 64, "real_multiply_lanes": lanes,
                "real_multiplications_per_frame": real_products, "input_bytes": input_bytes,
                "output_bytes": output_bytes, "resident_coefficient_bytes": coefficient_bytes,
                "compute_cycles_floor": compute_cycles,
                "ingress_cycles_floor": ingress_cycles, "egress_cycles_floor": egress_cycles,
                "overlapped_interval_floor_us": max(compute_cycles, ingress_cycles, egress_cycles)
                * clock_ns / 1000,
                "serialized_stage_budget_floor_us": (compute_cycles + ingress_cycles + egress_cycles)
                * clock_ns / 1000,
            }


def model_transfers():
    for scenario, input_bytes, output_bytes in (
        ("covariance_round_trip_packed_iq12", 3072, 256),
        ("covariance_round_trip_complex64", 8192, 256),
        ("covariance_sensor_resident_to_host", 0, 256),
        ("projection_2src_round_trip_with_pl_peak_pick", 168, 16),
        ("projection_2src_round_trip_without_pl_peak_pick", 168, 1920),
    ):
        for rate in (2, 8, 16):
            for fixed_us in (1, 5, 20):
                payload_us = (input_bytes + output_bytes) / (1000 * rate)
                yield {
                    "boundary": scenario, "input_bytes": input_bytes, "output_bytes": output_bytes,
                    "assumed_bandwidth_gbyte_per_second": rate,
                    "assumed_total_fixed_overhead_us": fixed_us, "payload_time_us": payload_us,
                    "transfer_floor_us": fixed_us + payload_us,
                }


def write_rows(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=128, help="frames per seed and case")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hardware"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        write_rows(args.output_dir / "fixed-point.csv", measure_accuracy(args.trials))
    write_rows(args.output_dir / "architecture-model.csv", model_architectures())
    write_rows(args.output_dir / "transfer-model.csv", model_transfers())
    print(f"Wrote fixed-point measurements and analytical models to {args.output_dir}")


if __name__ == "__main__":
    main()
