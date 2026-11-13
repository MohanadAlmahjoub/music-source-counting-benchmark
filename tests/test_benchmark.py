"""Verify estimator semantics, whitening geometry, and published data invariants."""

import csv
import gzip
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from music_benchmark.pilot import adaptive_counts, amd_threshold_proxy, mdl_count
from music_benchmark.pipeline import (
    generate_snapshot,
    make_steering,
    reference_music,
    run_music,
    whitening_matrix,
)
from music_benchmark.robustness import calibrate, decisions, noise_factor, whiten


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        limit = threadpool_limits(limits=1, user_api="blas")
        self.addCleanup(limit.restore_original_limits)

    def test_count_rules_and_sentinel(self):
        noise = np.ones(8)
        one = np.array([9.0, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(amd_threshold_proxy(noise), 0)
        self.assertEqual(amd_threshold_proxy(one), 0)
        self.assertEqual(mdl_count(one), 1)
        self.assertEqual(adaptive_counts(one), (1, 1))

    def test_whitening_matches_covariance_congruence(self):
        rng = np.random.default_rng(114)
        mix = noise_factor(0.3).astype(complex)
        mix[2, 0] += 0.15j
        rows = rng.standard_normal((128, 8)) + 1j * rng.standard_normal((128, 8))
        mixed = rows @ mix.T
        calibration = rng.standard_normal((2048, 8)) + 1j * rng.standard_normal((2048, 8))
        factor = calibrate(calibration @ mix.T)
        white = whiten(mixed, factor)
        inverse = np.linalg.inv(factor)
        np.testing.assert_allclose(
            white.T @ white.conj(), inverse @ (mixed.T @ mixed.conj()) @ inverse.conj().T
        )
        self.assertEqual(len(decisions(white)), 6)

    def test_whitened_music_uses_consistent_steering(self):
        rho = 0.3
        sensors = np.arange(8)
        covariance = (rho ** np.abs(sensors[:, None] - sensors)).astype(complex)
        phase = np.exp(0.2j * sensors)
        covariance *= phase[:, None] * phase.conj()
        frame = generate_snapshot(np.random.default_rng(81), [-28, 32], 15, 2, covariance)
        prepared = make_steering(whitener=whitening_matrix(covariance))
        actual = run_music(frame, prepared)
        reference = reference_music(frame, noise_covariance=covariance)
        self.assertEqual(actual.source_count, 2)
        np.testing.assert_allclose(actual.eigenvalues, reference.eigenvalues, atol=1e-10)
        np.testing.assert_allclose(actual.denominator, reference.denominator, atol=1e-12)
        np.testing.assert_allclose(actual.angles_deg, [-28, 32], atol=1.5)

    def test_public_raw_decisions_recover_reported_rates(self):
        archive = Path(__file__).resolve().parents[1] / "results/robustness/trials.csv.gz"
        totals = Counter()
        alarms = Counter()
        agreements = 0
        with gzip.open(archive, "rt", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                totals["all"] += 1
                agreements += row["white_float"] == row["white_q12"]
                if row["sources"] == "0":
                    scenario = row["scenario"]
                    totals[scenario] += 1
                    for method in ("white_q12", "white_mdl_capped"):
                        alarms[scenario, method] += int(row[method]) > 0
        self.assertEqual(totals, {"all": 46656, "white": 1728, "correlated": 1728, "drift": 1728})
        self.assertEqual(agreements, 46627)
        self.assertEqual([alarms[s, "white_q12"] for s in ("white", "correlated", "drift")], [4, 1, 804])
        self.assertEqual([alarms[s, "white_mdl_capped"] for s in ("white", "correlated", "drift")], [0, 0, 10])


if __name__ == "__main__":
    unittest.main()
