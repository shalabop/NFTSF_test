"""
Minimal training entry point for NFTSF.

Trimmed from the source repo's train_model.py: keeps only the data-loading
path needed for the example dataset (multi_sim .npy format), segment
extraction, and the training loop. Drops SDE-generation flags, sweep
support, unused data formats, LR scheduling, resume/early-stopping, and
plotting.

Architecture is never chosen manually here — see nftsf/model_selector.py.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from nftsf.model_selector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train an NFTSF conditional flow model.")
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to a .npy file, shape (Time, 1+NumSims): column 0 is time.")
    parser.add_argument("--n_past", type=int, default=100, help="Context length (steps).")
    parser.add_argument("--n_future", type=int, default=100, help="Forecast horizon (steps).")
    parser.add_argument("--stride", type=int, default=1, help="Stride between extracted segments.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_device(device_arg):
    device = torch.device("cuda" if (device_arg == "auto" and torch.cuda.is_available()) else
                           (device_arg if device_arg != "auto" else "cpu"))
    print(f"Using device: {device}")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path):
    """Load a multi_sim .npy file: shape (Time, 1+NumSims), column 0 is time.

    Returns:
        torch.Tensor of shape (NumSims, Time).
    """
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    positions_only = tensor_data[:, 1:]  # drop time column
    reshaped = positions_only.T  # (NumSims, Time)
    print(f"  Reshaped to: {tuple(reshaped.shape)} (Batch, Time)")
    return reshaped


def extract_segments(tracks, n_past, n_future, stride=1):
    """Sliding-window segment extraction. See docstring in the source repo's
    train_model.py::extract_segments for the full derivation.

    Returns:
        segments: tensor of shape (N_segments, n_past + n_future)
    """
    n_extrp = n_past + n_future
    segments = []
    num_tracks, length_track = tracks.shape
    for i in range(num_tracks):
        for start in range(0, length_track - n_extrp + 1, stride):
            segments.append(tracks[i, start:start + n_extrp])
    if not segments:
        raise ValueError(f"No segments extracted! Check data length ({length_track}) vs n_extrp ({n_extrp})")
    return torch.stack(segments)


def normalize(data):
    """Z-score normalize using global mean/std. Returns (normalized, mean, std)."""
    mean = data.mean().item()
    std = data.std().item()
    return (data - mean) / std, mean, std


def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_path)
    data, mean, std = normalize(data)

    segments = extract_segments(data, args.n_past, args.n_future, args.stride)
    n_total = segments.shape[0]
    n_val = max(1, int(n_total * args.val_fraction))
    perm = torch.randperm(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    n_extrp = args.n_past + args.n_future
    train_segments = segments[train_idx].to(device)
    val_segments = segments[val_idx].to(device)

    train_context, train_samples = train_segments[:, :args.n_past], train_segments[:, args.n_past:n_extrp]
    val_context, val_samples = val_segments[:, :args.n_past], val_segments[:, args.n_past:n_extrp]

    print(f"Train segments: {train_context.shape[0]}  Val segments: {val_context.shape[0]}")

    model, variant = build_model(device, args.n_past, args.n_future)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    n_segments = train_context.shape[0]
    batch_size = args.batch_size if 0 < args.batch_size < n_segments else n_segments

    loss_history, val_loss_history = [], []
    _train_start = time.time()
    for epoch in tqdm(range(args.epochs), desc="Training"):
        perm = torch.randperm(n_segments, device=device)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_segments, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = -model.log_prob(train_samples[idx], train_context[idx]).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Training crashed at epoch {epoch} (loss={loss.item()})")
                break
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / max(n_batches, 1))

        with torch.no_grad():
            val_loss = -model.log_prob(val_samples, val_context).mean().item()
        val_loss_history.append(val_loss)

    _training_time_seconds = time.time() - _train_start
    print(f"\nTotal training time: {_training_time_seconds:.1f} seconds "
          f"({_training_time_seconds / 60:.1f} min)")

    # Sampling benchmark: time n_samp_n samples from the first segment's context.
    # Ported from the reference repo's train/NFTSF/train_nftsf.py::main().
    _samp_n = 100
    _samp_ctx = train_context[:1]  # (1, n_past)
    _samp_rep = _samp_ctx.expand(_samp_n, -1)  # (n_samples, n_past)
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    config = {
        "model_path": model_path,
        "variant": variant,
        "n_past": args.n_past,
        "n_future": args.n_future,
        "past_dim": 1,
        "norm_mean": mean,
        "norm_std": std,
        "final_loss": loss_history[-1],
        "final_val_loss": val_loss_history[-1],
        "timestamp": timestamp,
        "training_time_seconds": _training_time_seconds,
        "sampling_benchmark_n_samples": _samp_n,
        "sampling_benchmark_total_seconds": _sampling_time,
        "sampling_time_per_sample_seconds": _time_per_sample,
    }
    config_path = os.path.join(args.output_dir, f"config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")
    print(f"\nFinal train loss: {loss_history[-1]:.4f}  Final val loss: {val_loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
