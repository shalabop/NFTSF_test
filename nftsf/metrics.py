"""
Forecast evaluation metrics: CRPS, MAE, RMSE, CI50/CI90 coverage.

Ported from two internally-consistent implementations in the source repo
(nftsf_ssh):

- CRPS: the closed-form, dependency-free order-statistic estimator from
  root `test_model.py::ensemble_crps` / `get_metrics_per_step`, generalized
  here to batches of trajectories. This is mathematically equivalent to the
  `properscoring.crps_ensemble` estimator used by
  `unified_trajectory_forecasting/eval/evaluate.py`, but avoids adding the
  `properscoring` dependency to this minimal repo's requirements.txt.
- MAE / RMSE (of the ensemble median) and CI50/CI90 coverage: ported from
  `unified_trajectory_forecasting/eval/evaluate.py::_mae_per_step`,
  `_rmse_per_step`, `_ci_coverage_per_step`. The CI90 = [5th, 95th]
  percentile band and CI50 = [25th, 75th] percentile band convention
  matches the reference repo's own
  eval/NF/forecast_nf_encoder.py percentile choices.
- ISCE (Integrated Squared Calibration Error): ported directly from the
  reference repo's `eval/compare.py::_isce_central_per_step`. For each of
  101 central-interval widths (0%, 1%, ..., 100%), compares empirical
  coverage of that width's percentile band against its nominal value, and
  integrates the squared error over all widths. Lower is better calibrated.

Shapes:
    ground_truth : (N, T) float32       — N test trajectories, T future steps
    samples      : (N, T, S) float32    — S forecast samples per step
"""

import numpy as np


def crps_per_step(ground_truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS per forecast step, averaged over test trajectories.

    For an ensemble of M sorted samples x_(1) <= ... <= x_(M):
        CRPS = mean_i |x_i - y|  -  (2 / [M*(M-1)]) * sum_i w_i * (x_(i+1) - x_(i))
    where w_i = i * (M - i) counts the pairs straddling gap i
    (Gneiting & Raftery, 2007 order-statistic representation).
    """
    mae_term = np.mean(np.abs(samples - ground_truth[:, :, None]), axis=2)  # (N, T)

    n_samples = samples.shape[2]
    sorted_samples = np.sort(samples, axis=2)
    diff = sorted_samples[:, :, 1:] - sorted_samples[:, :, :-1]
    w = np.arange(1, n_samples) * np.arange(n_samples - 1, 0, -1)
    energy_term = np.sum(diff * w[None, None, :], axis=2) / (n_samples * (n_samples - 1))

    crps = mae_term - energy_term  # (N, T)
    return crps.mean(axis=0)  # (T,)


def mae_per_step(ground_truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """MAE of the median forecast per step, averaged over trajectories."""
    median = np.median(samples, axis=-1)  # (N, T)
    return np.abs(ground_truth - median).mean(axis=0)  # (T,)


def rmse_per_step(ground_truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """RMSE of the median forecast per step."""
    median = np.median(samples, axis=-1)
    return np.sqrt(((ground_truth - median) ** 2).mean(axis=0))


def ci_coverage_per_step(ground_truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Fraction of ground-truth values inside [lower, upper] per step."""
    inside = (ground_truth >= lower) & (ground_truth <= upper)
    return inside.mean(axis=0)  # (T,)


def isce_per_step(ground_truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Integrated Squared Calibration Error per forecast step.

    Ported from compare.py::_isce_central_per_step (1% steps over central
    coverage widths 0-100%), generalized here over the batch axis.
    """
    T = ground_truth.shape[1]
    pcts = np.arange(0, 101, 1)
    isce = np.zeros(T)
    for t in range(T):
        gt_t = ground_truth[:, t]
        samp_t = samples[:, t, :]
        err = 0.0
        for pct in pcts:
            if pct == 0:
                emp = 0.0
            elif pct == 100:
                emp = 1.0
            else:
                lo = (100 - pct) / 2.0
                hi = 100 - lo
                lo_q = np.percentile(samp_t, lo, axis=1)
                hi_q = np.percentile(samp_t, hi, axis=1)
                emp = np.mean((gt_t >= lo_q) & (gt_t <= hi_q))
            err += (emp - pct / 100.0) ** 2
        isce[t] = err / len(pcts)
    return isce


def compute_all_metrics(ground_truth: np.ndarray, samples: np.ndarray) -> dict:
    """Compute CRPS, MAE, RMSE, CI50 and CI90 coverage, per step and mean."""
    ci90_lower = np.percentile(samples, 5, axis=-1)
    ci90_upper = np.percentile(samples, 95, axis=-1)
    ci50_lower = np.percentile(samples, 25, axis=-1)
    ci50_upper = np.percentile(samples, 75, axis=-1)

    crps_t = crps_per_step(ground_truth, samples)
    mae_t = mae_per_step(ground_truth, samples)
    rmse_t = rmse_per_step(ground_truth, samples)
    ci90_t = ci_coverage_per_step(ground_truth, ci90_lower, ci90_upper)
    ci50_t = ci_coverage_per_step(ground_truth, ci50_lower, ci50_upper)
    isce_t = isce_per_step(ground_truth, samples)

    return {
        "crps_per_step": crps_t,
        "mae_per_step": mae_t,
        "rmse_per_step": rmse_t,
        "ci90_coverage_per_step": ci90_t,
        "ci50_coverage_per_step": ci50_t,
        "isce_per_step": isce_t,
        "crps_mean": float(crps_t.mean()),
        "mae_mean": float(mae_t.mean()),
        "rmse_mean": float(rmse_t.mean()),
        "ci90_coverage_mean": float(ci90_t.mean()),
        "ci50_coverage_mean": float(ci50_t.mean()),
        "isce_mean": float(isce_t.mean()),
    }
