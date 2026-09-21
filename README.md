# NFTSF minimal repo
Minimal NFTSF (Normalizing Flow Time Series
Forecasting) pipeline: one training example, one test/eval example, and a
single unified codepath that automatically picks the encoder or
non-encoder architecture based on context length. No sweep scripts, no
multi-model comparison scripts, no unrelated preprocessing — the test
example does include single-model trajectory-band and forecast-density
plots (see "Plots and timing" below), simplified from the reference
repo's multi-model `compare.py`.

Ported from the [`nftsf_ssh`](https://github.com/shalabop/nftsf_ssh)
source repo, following the training/testing method and metrics of
[Time_series_forecasting_for_stochastic_dynamics](https://github.com/12Mahmud99/Time_series_forecasting_for_stochastic_dynamics)
(`architectures/` and `eval/` folders — see "Metrics" below for the exact
cross-check).

## Automatic architecture selection

`nftsf/model_selector.py` defines:

```python
ENCODER_THRESHOLD = 64

if n_past > ENCODER_THRESHOLD:
    # GRU-encoder architecture (architecture_encoder.py)
else:
    # plain, non-encoder architecture (architecture.py)
```

This is not a manual flag — both `nftsf/train.py` and `nftsf/test.py` call
`model_selector.build_model(device, n_past, n_future)`, which prints which
architecture it chose and why, e.g.:

```
n_past=100 > 64 -> using GRU-encoder architecture
```

Note: `n_past == 64` uses the **non-encoder** path (the comparison is a
strict `>`). `nftsf/test.py` reads `n_past` from the training run's
`config.json` rather than a CLI override, so a checkpoint always reloads
into the architecture it was trained with.

## Layout

```
nftsf/
  architecture.py          # non-encoder conditional A-RQS flow (create_nfm)
  architecture_encoder.py  # GRU-encoder + flow (create_nfm_encoder) — trimmed
                            # from the source repo to the GRU variant only,
                            # dropping MLP/CNN/Transformer encoders and the
                            # numbered preset_stage1..16 ablations
  model_selector.py        # ENCODER_THRESHOLD + build_model()
  train.py                 # data load -> segment extraction -> training loop
                            # (+ training-time / sampling-time benchmark)
  test.py                  # checkpoint load -> forecast samples -> metrics -> plots
  metrics.py                # CRPS / MAE / RMSE / CI50 / CI90 coverage / ISCE
  plots.py                 # trajectory-band plot + histogram2d density plot
examples/
  data/alanine_phi_train.npy
  data/alanine_phi_test.npy
  run_train_example.sh
  run_test_example.sh
```

## Quickstart

```bash
pip install -r requirements.txt
./examples/run_train_example.sh   # trains a smoke-test checkpoint (n_past=100 -> encoder)
./examples/run_test_example.sh    # evaluates it: CRPS/MAE/RMSE/CI50/CI90/ISCE + plots
```

`run_test_example.sh` prints CRPS, MAE, RMSE, CI50/CI90 coverage, ISCE, and
a sampling-time benchmark, then saves to `examples/test_results/`:
- `trajectory_bands.png` — ground truth + predicted median + 50%/90% bands
- `histogram2d_light.png` / `histogram2d_dark.png` — forecast sample density
- `results.npz` — all per-step metrics, the raw samples/ground truth, and
  the sampling-time benchmark

The training example uses `--n_past 100 --n_future 100` against
`alanine_phi_train.npy`, which demonstrates the encoder path
auto-activating (100 > 64). **The paper's real runs use `--epochs 1000
--stride 1`** (full sliding-window data augmentation); **this example uses
`--epochs 2 --stride 200`** (fewer, non-overlapping segments) purely to
verify the pipeline runs end-to-end quickly — it will not produce a
well-trained model. Similarly, the test example passes `--max_segments 5`
to evaluate only a handful of test windows instead of the full held-out
set.

> This repo's example scripts were written and statically checked
> (`python -m py_compile`) in an environment without network access to
> install `torch`/`normflows`/`matplotlib`, so end-to-end execution
> (including the plots) has not been verified live. Please run
> `run_train_example.sh` and `run_test_example.sh` yourself after
> `pip install -r requirements.txt` to confirm the pipeline runs
> end-to-end, and open an issue if anything doesn't match this README.

## Metrics

`nftsf/metrics.py` computes CRPS, MAE, RMSE (of the ensemble median), and
CI50/CI90 coverage, per forecast step and averaged. These are ported from
two internally-consistent implementations already present in the source
repo:

- **CRPS**: the closed-form, dependency-free order-statistic estimator
  from `nftsf_ssh/test_model.py::ensemble_crps` /
  `get_metrics_per_step`. This is mathematically equivalent to the
  `properscoring.crps_ensemble` estimator used by
  `nftsf_ssh/unified_trajectory_forecasting/eval/evaluate.py`, but was
  chosen here specifically to avoid adding the `properscoring` dependency
  to this minimal repo.
- **MAE / RMSE / CI50 / CI90 coverage**: ported from
  `unified_trajectory_forecasting/eval/evaluate.py::_mae_per_step`,
  `_rmse_per_step`, `_ci_coverage_per_step`.
- **CI90 = [5th, 95th] percentile band, CI50 = [25th, 75th] percentile
  band**: this convention was cross-checked directly against the
  reference repo's own
  `eval/NF/forecast_nf_encoder.py`, which computes the identical
  percentile bounds (`np.percentile(samples, 5/95/25/75, axis=...)`).

- **ISCE (Integrated Squared Calibration Error)**: ported directly from
  the reference repo's `eval/compare.py::_isce_central_per_step`. For each
  of 101 central-interval widths (0%, 1%, ..., 100%), it compares the
  empirical coverage of that width's percentile band against its nominal
  value and integrates the squared error over all widths — a single
  overall calibration score (lower is better), rather than checking just
  the two fixed CI50/CI90 widths. Printed and saved as `isce_mean * 1000`
  to match `compare.py`'s own table/plot scaling.

The reference repo's non-encoder training entry point is
`train/NFTSF/train_nftsf.py`; its argument defaults and model-variant
scheme match `nftsf_ssh/train_model.py` closely, confirming `nftsf_ssh`
already follows this reference method — this minimal repo inherits that
same lineage.

## Plots and timing

`nftsf/train.py` measures and saves into `config.json` (ported from
`train/NFTSF/train_nftsf.py::main()`):
- `training_time_seconds` — wall-clock time for the full training loop
- a **sampling benchmark**: time to draw 100 forecast samples from one
  training segment's context (`sampling_benchmark_total_seconds`,
  `sampling_time_per_sample_seconds`), with `torch.cuda.synchronize()`
  guards on GPU so the timing isn't measuring an async kernel launch

`nftsf/test.py` runs the same sampling benchmark against the loaded
checkpoint, and generates two plot types via `nftsf/plots.py` for the
first `--n_traj_show` test segments (default 3), simplified to a single
trained model from the reference repo's multi-model
`eval/compare.py`:
- **`plot_trajectory_bands`** (from `plot_trajectory_comparison_grid`):
  ground truth, predicted median, a 50% band (25th/75th percentile) and a
  90% band (5th/95th percentile)
- **`plot_histogram2d`** (from `plot_histogram2d_comparison_grid`): a 2D
  log-density histogram of forecast samples over the forecast horizon,
  with ground truth overlaid, in both light- and dark-background variants
