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

import numpy as np
import torch

from nftsf.model_selector import build_model
from nftsf.metrics import compute_all_metrics
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
    )
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
