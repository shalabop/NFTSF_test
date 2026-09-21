"""
Minimal test/eval entry point for NFTSF.

Trimmed from the source repo's test_model.py: keeps only the
load-checkpoint + generate-forecast-samples + compute-metrics path.

Architecture is never chosen manually here: n_past is read from the
training run's saved config.json (not a CLI override), and the same
automatic selector in nftsf/model_selector.py picks the matching
architecture — so a checkpoint always reloads into the architecture it
was trained with.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from nftsf.model_selector import build_model
from nftsf.metrics import compute_all_metrics
from nftsf.plots import plot_trajectory_bands, plot_histogram2d
from nftsf.train import load_data, extract_segments, setup_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test/evaluate a trained NFTSF checkpoint.")
    parser.add_argument("--config_path", type=str, required=True,
                         help="config_*.json produced by nftsf/train.py.")
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to the held-out test .npy file (same format as training data).")
    parser.add_argument("--n_samples", type=int, default=500, help="Forecast samples per test segment.")
    parser.add_argument("--max_segments", type=int, default=None,
                         help="Cap the number of evaluated test segments (for quick smoke tests). "
                              "Default: evaluate every non-overlapping segment in the test set.")
    parser.add_argument("--n_traj_show", type=int, default=3,
                         help="Number of test segments to render in the trajectory-band and "
                              "histogram2d plots.")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=29182)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config_path) as f:
        config = json.load(f)

    n_past, n_future = config["n_past"], config["n_future"]
    mean, std = config["norm_mean"], config["norm_std"]
    print(f"Loaded config: n_past={n_past} n_future={n_future} variant={config['variant']}")

    model, variant = build_model(device, n_past, n_future)
    assert variant == config["variant"], (
        f"Config was trained as {config['variant']!r} but n_past={n_past} now "
        f"auto-selects {variant!r}; the checkpoint would not load correctly."
    )
    state_dict = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    data = load_data(args.data_path)
    data = (data - mean) / std

    # Non-overlapping test segments: one independent forecast window per segment.
    segments = extract_segments(data, n_past, n_future, stride=n_past + n_future)
    if args.max_segments is not None:
        segments = segments[:args.max_segments]
    segments = segments.to(device)
    context = segments[:, :n_past]
    ground_truth = segments[:, n_past:].cpu().numpy()  # (N, n_future), normalized
    full_trajectories = segments.cpu().numpy() * std + mean  # (N, n_past+n_future), real units

    print(f"Generating {args.n_samples} forecast samples for {context.shape[0]} test segments...")
    all_samples = []
    with torch.no_grad():
        for i in range(context.shape[0]):
            past_repeat = context[i:i + 1].repeat(args.n_samples, 1)
            samples = model.sample(args.n_samples, past_repeat)[0]  # (n_samples, n_future)
            all_samples.append(samples.cpu().numpy())
    samples = np.stack(all_samples, axis=0)  # (N, n_samples, n_future)
    samples = np.transpose(samples, (0, 2, 1))  # (N, n_future, n_samples)

    # Denormalize back to original units before computing metrics.
    ground_truth = ground_truth * std + mean
    samples = samples * std + mean

    metrics = compute_all_metrics(ground_truth, samples)

    print(f"\nMean CRPS : {metrics['crps_mean']:.4f}")
    print(f"Mean MAE  : {metrics['mae_mean']:.4f}")
    print(f"Mean RMSE : {metrics['rmse_mean']:.4f}")
    print(f"CI50 coverage (target 0.50): {metrics['ci50_coverage_mean']:.4f}")
    print(f"CI90 coverage (target 0.90): {metrics['ci90_coverage_mean']:.4f}")
    print(f"ISCE*1000 (lower=better calibrated): {metrics['isce_mean'] * 1000:.4f}")

    # Sampling benchmark: same style/measurement as train.py's, run here against
    # the loaded checkpoint (mirrors train/NFTSF/train_nftsf.py::main()).
    _samp_n = 100
    _samp_ctx = context[:1]
    _samp_rep = _samp_ctx.expand(_samp_n, -1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    _t0 = time.time()
    with torch.no_grad():
        model.sample(_samp_n, _samp_rep)
    if device.type == "cuda":
        torch.cuda.synchronize()
    _sampling_time = time.time() - _t0
    _time_per_sample = _sampling_time / _samp_n
    print(f"Sampling benchmark ({_samp_n} samples): "
          f"{_sampling_time * 1000:.1f}ms total  "
          f"({_time_per_sample * 1000:.3f}ms/sample)")

    print("\nGenerating trajectory-band and histogram2d plots...")
    plot_trajectory_bands(
        full_trajectories, samples, n_past, n_future,
        os.path.join(args.output_dir, "trajectory_bands.png"),
        n_show=args.n_traj_show,
    )
    plot_histogram2d(
        full_trajectories, samples, n_past, n_future,
        os.path.join(args.output_dir, "histogram2d_light.png"),
        n_show=args.n_traj_show, dark=False,
    )
    plot_histogram2d(
        full_trajectories, samples, n_past, n_future,
        os.path.join(args.output_dir, "histogram2d_dark.png"),
        n_show=args.n_traj_show, dark=True,
    )
    print(f"Saved plots to: {args.output_dir}")

    results_path = os.path.join(args.output_dir, "results.npz")
    np.savez(
        results_path,
        ground_truth=ground_truth.astype(np.float32),
        samples=samples.astype(np.float32),
        crps_per_step=metrics["crps_per_step"],
        mae_per_step=metrics["mae_per_step"],
        rmse_per_step=metrics["rmse_per_step"],
        ci50_coverage_per_step=metrics["ci50_coverage_per_step"],
        ci90_coverage_per_step=metrics["ci90_coverage_per_step"],
        isce_per_step=metrics["isce_per_step"],
        sampling_benchmark_n_samples=_samp_n,
        sampling_benchmark_total_seconds=_sampling_time,
        sampling_time_per_sample_seconds=_time_per_sample,
    )
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
