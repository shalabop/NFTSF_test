# NFTSF minimal repo
Minimal NFTSF (Normalizing Flow Time Series
Forecasting) pipeline: one training example, one test/eval example, and a
single unified codepath that automatically picks the encoder or
non-encoder architecture based on context length. No sweep scripts, no
comparison/plotting scripts, no unrelated preprocessing.

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
  test.py                  # checkpoint load -> forecast samples -> metrics
  metrics.py                # CRPS / MAE / RMSE / CI50 / CI90 coverage
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
./examples/run_test_example.sh    # evaluates it, prints CRPS/MAE/RMSE/CI50/CI90
```

The training example uses `--n_past 100 --n_future 100` against
`alanine_phi_train.npy`, which demonstrates the encoder path
auto-activating (100 > 64). **The paper's real runs use `--epochs 1000
--stride 1`** (full sliding-window data augmentation); **this example uses
`--epochs 2 --stride 200`** (fewer, non-overlapping segments) purely to
verify the pipeline runs end-to-end quickly — it will not produce a
well-trained model. Similarly, the test example passes `--max_segments 5`
to evaluate only a handful of test windows instead of the full held-out
set.

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

The reference repo's non-encoder training entry point is
`train/NFTSF/train_nftsf.py`; its argument defaults and model-variant
scheme match `nftsf_ssh/train_model.py` closely, confirming `nftsf_ssh`
already follows this reference method — this minimal repo inherits that
same lineage.
