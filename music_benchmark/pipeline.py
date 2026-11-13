"""CPU MUSIC and an independent SVD reference for complex array snapshots."""

from dataclasses import dataclass
from functools import lru_cache
from math import log
from time import perf_counter_ns

import numpy as np


SENSORS = 8
SNAPSHOTS = 128
ANGLE_GRID_DEG = np.linspace(-90.0, 90.0, 256)[9:249]
METHODS = ("amd_5pct", "mdl", "mdl_scalar", "tail_float", "tail_q12")
FLOAT64_TINY = np.finfo(np.float64).tiny
STAGES = (
    "whiten_snapshots",
    "covariance",
    "eigendecomposition",
    "count_sources",
    "construct_noise_subspace",
    "scan_spectrum",
    "pick_angles",
)


@dataclass(frozen=True)
class PreparedSteering:
    """Column steering vectors in the same space as the whitened snapshots."""

    angles_deg: np.ndarray
    steering: np.ndarray
    whitener: np.ndarray | None


@dataclass(frozen=True)
class MusicResult:
    source_count: int
    angles_deg: np.ndarray
    eigenvalues: np.ndarray
    denominator: np.ndarray


def complex_normal(rng, shape):
    """Draw circular complex Gaussian noise with unit expected squared magnitude."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def steering_vectors(angles_deg, sensors=SENSORS):
    """Return the half-wavelength ULA steering vectors as columns."""
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.exp(1j * np.pi * np.arange(sensors)[:, None] * np.sin(angles_rad))


def generate_snapshot(
    rng,
    angles_deg,
    strongest_snr_db,
    power_spread_db=0.0,
    noise_covariance=None,
    snapshot_count=SNAPSHOTS,
    sensors=SENSORS,
):
    """Generate independent narrowband emitters and return one time sample per row."""
    data = complex_normal(rng, (snapshot_count, sensors))
    if noise_covariance is not None:
        data = data @ np.linalg.cholesky(noise_covariance).T
    if len(angles_deg):
        offsets_db = np.linspace(0.0, power_spread_db, len(angles_deg))
        amplitudes = 10.0 ** ((strongest_snr_db - offsets_db) / 20.0)
        samples = complex_normal(rng, (snapshot_count, len(angles_deg))) * amplitudes
        data += samples @ steering_vectors(angles_deg, sensors).T
    return data


def sample_covariance(snapshots):
    """Return E[x xᴴ] for known zero-mean snapshots stored as plain xᵀ rows."""
    return snapshots.T @ snapshots.conj() / len(snapshots)


def estimate_noise_covariance(noise_snapshots, diagonal_loading=0.01):
    """Estimate a Hermitian noise covariance from independent noise-only data."""
    if not 0.0 <= diagonal_loading <= 1.0:
        raise ValueError("diagonal_loading must be between zero and one")
    covariance = sample_covariance(noise_snapshots)
    covariance = (covariance + covariance.conj().T) / 2.0
    mean_power = covariance.trace().real / covariance.shape[0]
    return (1.0 - diagonal_loading) * covariance + diagonal_loading * mean_power * np.eye(
        covariance.shape[0]
    )


def whitening_matrix(noise_covariance):
    """Return W=L⁻¹ for a Hermitian positive-definite covariance LLᴴ."""
    factor = np.linalg.cholesky(noise_covariance)
    return np.linalg.solve(factor, np.eye(factor.shape[0], dtype=factor.dtype))


def make_steering(angles_deg=ANGLE_GRID_DEG, sensors=SENSORS, whitener=None):
    """Precompute steering atoms and normalize columns after any whitening."""
    angles = np.array(angles_deg, dtype=np.float64, copy=True)
    if angles.ndim != 1 or angles.size < 3 or np.any(np.diff(angles) <= 0):
        raise ValueError("provide at least three angles in strictly increasing order")
    steering = steering_vectors(angles, sensors)
    if whitener is not None:
        whitener = np.array(whitener, copy=True)
        steering = whitener @ steering
    steering /= np.linalg.norm(steering, axis=0)
    return PreparedSteering(angles, steering, whitener)


def quantize_eigenvalues(eigenvalues, bits=12):
    """Quantize descending eigenvalues relative to the first, rounding ties to even."""
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values[0] <= 0.0:
        return np.zeros(values.shape, dtype=np.int32)
    limit = (1 << bits) - 1
    return np.rint(np.clip(values / values[0], 0.0, 1.0) * limit).astype(np.int32)


@lru_cache(maxsize=64)
def _mdl_penalties(snapshot_count):
    penalty_scale = 0.5 * log(snapshot_count)
    return tuple(penalty_scale * order * (2 * SENSORS - order) for order in range(SENSORS))


def _scalar_mdl(values, snapshot_count, max_sources):
    penalties = _mdl_penalties(snapshot_count)
    noise_sum = 0.0
    noise_log_sum = 0.0
    best_score = float("inf")
    best_order = 0
    for order in range(SENSORS - 1, -1, -1):
        value = max(float(values[order]), FLOAT64_TINY)
        noise_sum += value
        noise_log_sum += log(value)
        if order <= max_sources:
            noise_dim = SENSORS - order
            likelihood = snapshot_count * (noise_dim * log(noise_sum / noise_dim) - noise_log_sum)
            score = likelihood + penalties[order]
            if score <= best_score:
                best_score, best_order = score, order
    return best_order


def count_sources(eigenvalues, method="mdl", snapshot_count=SNAPSHOTS, max_sources=3):
    """Estimate effective count from descending eigenvalues of an 8-element array."""
    values = np.asarray(eigenvalues)
    if values.shape != (SENSORS,):
        raise ValueError("the count rules expect eight descending eigenvalues")
    if not 0 <= max_sources < SENSORS or snapshot_count < 1:
        raise ValueError("invalid maximum source count or sample count")
    if method not in METHODS:
        raise ValueError(f"unknown source count method: {method}")
    if values[0] <= 0.0:
        return 0
    if method == "amd_5pct":
        count = int(np.count_nonzero(values > 0.05 * values[0]))
        return 0 if count == SENSORS else count
    if method == "mdl":
        positive = np.maximum(values, FLOAT64_TINY)
        suffix_sum = np.cumsum(positive[::-1])[::-1]
        suffix_log_sum = np.cumsum(np.log(positive[::-1]))[::-1]
        order = np.arange(max_sources + 1)
        noise_dim = SENSORS - order
        log_ratio = np.log(suffix_sum[order] / noise_dim) - suffix_log_sum[order] / noise_dim
        penalty = 0.5 * order * (2 * SENSORS - order) * np.log(snapshot_count)
        return int(np.argmin(snapshot_count * noise_dim * log_ratio + penalty))
    if method == "mdl_scalar":
        return _scalar_mdl(values, snapshot_count, max_sources)
    if method == "tail_float":
        cutoff = 1.125 * (values[5] + values[6])
        return int(np.count_nonzero(values[:max_sources] > cutoff))
    codes = quantize_eigenvalues(values)
    return int(np.count_nonzero(8 * codes[:max_sources] > 9 * (codes[5] + codes[6])))


def pick_angles(denominator, angles_deg, count):
    """Select the deepest interior grid minima and return angles in ascending order."""
    if count == 0:
        return np.empty(0, dtype=np.float64)
    # Collapse roundoff-scale plateaus before comparing neighboring bins.
    denominator = np.round(denominator, decimals=12)
    local_minima = np.flatnonzero(
        (denominator[1:-1] < denominator[:-2]) & (denominator[1:-1] <= denominator[2:])
    ) + 1
    minima = local_minima[np.argsort(denominator[local_minima], kind="stable")[:count]]
    return np.sort(angles_deg[minima])


def _noise_subspace(vectors, count, scan):
    if count == 0:
        return None
    sensors = vectors.shape[1]
    noise_vectors = vectors[:, : sensors - count]
    if scan == "dense":
        return noise_vectors @ noise_vectors.conj().T
    return noise_vectors.conj().T


def _scan(subspace, steering, scan):
    if subspace is None:
        return np.ones(steering.shape[1])
    projection = subspace @ steering
    if scan == "dense":
        return np.einsum("ij,ij->j", steering.conj(), projection).real
    return np.einsum("ij,ij->j", projection.conj(), projection).real


def scan_spectrum(vectors, prepared, count, scan="projected"):
    """Return MUSIC's denominator for ascending eigenvector columns and a fixed grid."""
    if scan not in ("dense", "projected"):
        raise ValueError(f"unknown spectrum scan: {scan}")
    return _scan(_noise_subspace(vectors, count, scan), prepared.steering, scan)


def run_music(snapshots, prepared, method="mdl", scan="projected", stage_ns=None):
    """Run MUSIC with a dense or thin noise projector and optionally collect stage timings."""
    if scan not in ("dense", "projected"):
        raise ValueError(f"unknown spectrum scan: {scan}")
    instrumented = stage_ns is not None
    if instrumented:
        start = perf_counter_ns()
    data = snapshots if prepared.whitener is None else snapshots @ prepared.whitener.T
    if instrumented:
        after_whitening = perf_counter_ns()
    covariance = sample_covariance(data)
    if instrumented:
        after_covariance = perf_counter_ns()
    values, vectors = np.linalg.eigh(covariance)
    eigenvalues = values[::-1]
    if instrumented:
        after_eigendecomposition = perf_counter_ns()
    count = count_sources(eigenvalues, method, snapshot_count=len(data))
    if instrumented:
        after_count = perf_counter_ns()
    subspace = _noise_subspace(vectors, count, scan)
    if instrumented:
        after_subspace = perf_counter_ns()
    denominator = _scan(subspace, prepared.steering, scan)
    if instrumented:
        after_scan = perf_counter_ns()
    angles = pick_angles(denominator, prepared.angles_deg, count)
    if instrumented:
        after_angles = perf_counter_ns()
        stage_ns.update(
            whiten_snapshots=after_whitening - start,
            covariance=after_covariance - after_whitening,
            eigendecomposition=after_eigendecomposition - after_covariance,
            count_sources=after_count - after_eigendecomposition,
            construct_noise_subspace=after_subspace - after_count,
            scan_spectrum=after_scan - after_subspace,
            pick_angles=after_angles - after_scan,
        )
    return MusicResult(count, angles, eigenvalues, denominator)


def _reference_count(values, method, samples, max_sources):
    if method == "amd_5pct":
        count = sum(float(value) > 0.05 * values[0] for value in values)
        return 0 if count == SENSORS else count
    if method in ("mdl", "mdl_scalar"):
        safe = np.maximum(values, FLOAT64_TINY)
        scores = []
        for count in range(max_sources + 1):
            noise = safe[count:]
            likelihood = samples * len(noise) * (np.log(noise.mean()) - np.log(noise).mean())
            scores.append(likelihood + 0.5 * count * (2 * SENSORS - count) * np.log(samples))
        return int(np.argmin(scores))
    if method == "tail_float":
        cutoff = 2.25 * np.median(values[-4:])
        return sum(float(value) > cutoff for value in values[:max_sources])
    if method == "tail_q12":
        if values[0] <= 0:
            return 0
        codes = [round(4095 * max(0.0, float(value)) / values[0]) for value in values]
        cutoff = 2.25 * (codes[5] + codes[6]) / 2
        return sum(code > cutoff for code in codes[:max_sources])
    raise ValueError(f"unknown source count method: {method}")


def reference_music(snapshots, angles_deg=ANGLE_GRID_DEG, noise_covariance=None, method="mdl"):
    """Reference based on snapshot SVD and scalar steering/eigenvector dot products."""
    channels = snapshots.T
    vectors = steering_vectors(angles_deg, channels.shape[0])
    if noise_covariance is not None:
        factor = np.linalg.cholesky(noise_covariance)
        channels = np.linalg.solve(factor, channels)
        vectors = np.linalg.solve(factor, vectors)
    left_vectors, singular_values, _ = np.linalg.svd(channels, full_matrices=False)
    eigenvalues = singular_values**2 / channels.shape[1]
    count = _reference_count(eigenvalues, method, channels.shape[1], max_sources=3)
    noise_vectors = left_vectors[:, count:]
    denominator = np.empty(vectors.shape[1])
    for index, vector in enumerate(vectors.T):
        unit_vector = vector / np.linalg.norm(vector)
        denominator[index] = sum(abs(np.vdot(noise_vector, unit_vector)) ** 2 for noise_vector in noise_vectors.T)
    rounded = np.round(denominator, decimals=12)
    local_minima = [
        index
        for index in range(1, len(denominator) - 1)
        if rounded[index] < rounded[index - 1] and rounded[index] <= rounded[index + 1]
    ]
    selected = sorted(local_minima, key=rounded.__getitem__)[:count]
    angles = np.sort(np.asarray(angles_deg)[selected])
    return MusicResult(count, angles, eigenvalues, denominator)
