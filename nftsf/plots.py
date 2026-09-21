"""
Trajectory-band and forecast-density plots.

Ported and simplified (single trained model, not the multi-model comparison
grid) from the reference repo's eval/compare.py:
`plot_trajectory_comparison_grid` -> plot_trajectory_bands
`plot_histogram2d_comparison_grid` -> plot_histogram2d

Both take (N, n_past + n_future) ground-truth trajectories and
(N, n_future, S) forecast samples, all in real (denormalized) units, for a
handful of test segments (n_show), and save one PNG each.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np


def plot_trajectory_bands(full_trajectories, samples, n_past, n_future, output_path, n_show=3):
    """Ground truth, predicted median, 50% band (25/75 pct) and 90% band
    (5/95 pct) for the first n_show test segments. Mirrors compare.py's
    plot_trajectory_comparison_grid styling (median in red, 50% band in
    blue, the outer 90% wings in orange).
    """
    n_show = min(n_show, full_trajectories.shape[0])
    future_steps = np.arange(n_past, n_past + n_future)
    all_steps = np.arange(0, n_past + n_future)

    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show / 3, 3), squeeze=False)
    axes = axes[0]

    MEDIAN_COLOR = "#df1111"
    legend_handles, legend_labels = [], []

    for col in range(n_show):
        ax = axes[col]
        real_traj = full_trajectories[col]
        samp = samples[col]  # (n_future, S)
        median = np.median(samp, axis=1)
        lo90 = np.percentile(samp, 5, axis=1)
        hi90 = np.percentile(samp, 95, axis=1)
        lo50 = np.percentile(samp, 25, axis=1)
        hi50 = np.percentile(samp, 75, axis=1)

        line_truth, = ax.plot(all_steps, real_traj, "k-", linewidth=1.0, label="Truth")
        line_median, = ax.plot(future_steps, median, color=MEDIAN_COLOR, linewidth=1.2, label="Median")
        ax.fill_between(future_steps, lo50, hi50, color="tab:blue", alpha=0.40)
        ax.fill_between(future_steps, lo90, lo50, color="tab:orange", alpha=0.40)
        ax.fill_between(future_steps, hi50, hi90, color="tab:orange", alpha=0.40)
        ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
        ax.set_title(f"Test segment {col}", fontsize=10)
        ax.set_xlabel("Forecast step")

        if col == 0:
            ax.set_ylabel("Value")
            legend_handles = [line_truth, line_median,
                               Patch(facecolor="tab:blue", alpha=0.40),
                               Patch(facecolor="tab:orange", alpha=0.40)]
            legend_labels = ["Truth", "Median", "50% band", "90% band"]

    fig.legend(legend_handles, legend_labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=9, framealpha=0.8)
    fig.suptitle("Trajectory forecast: median + 50%/90% bands", fontsize=11, y=1.18)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_histogram2d(full_trajectories, samples, n_past, n_future, output_path, n_show=3, dark=False):
    """2D density histogram of forecast samples over forecast steps, with
    ground truth overlaid. Mirrors compare.py's
    plot_histogram2d_comparison_grid (dark=True reproduces its dark-
    background variant, dark=False its light-background variant).
    """
    n_show = min(n_show, full_trajectories.shape[0])
    style = "dark_background" if dark else "default"
    future_steps = np.arange(n_past, n_past + n_future)
    all_steps = np.arange(0, n_past + n_future)

    with plt.style.context(style):
        fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show / 3, 3), squeeze=False)
        axes = axes[0]

        n_x_bins = max(12, n_future // 5)
        n_y_bins = 24
        y_lo = min(full_trajectories[:n_show].min(), samples[:n_show].min())
        y_hi = max(full_trajectories[:n_show].max(), samples[:n_show].max())
        x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
        y_edges = np.linspace(y_lo, y_hi, n_y_bins + 1)

        truth_color = "lime" if dark else "black"
        facecolor = "black" if dark else "white"

        for col in range(n_show):
            ax = axes[col]
            real_traj = full_trajectories[col]
            samp = samples[col]  # (n_future, S)
            samp_for_hist = samp.T  # (S, n_future)
            time_rep = np.tile(future_steps, (samp_for_hist.shape[0], 1))
            ax.hist2d(time_rep.flatten(), samp_for_hist.flatten(),
                      bins=[x_edges, y_edges], cmap="magma", density=True,
                      norm=LogNorm(vmin=1e-6), rasterized=True)
            ax.plot(all_steps, real_traj, color=truth_color, linewidth=1.0, label="Truth")
            ax.axvline(x=n_past, color="gray", linestyle="--", alpha=0.5)
            ax.set_facecolor(facecolor)
            ax.set_title(f"Test segment {col}", fontsize=10)
            ax.set_xlabel("Forecast step")
            if col == 0:
                ax.set_ylabel("Value")
                ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

        suffix = "dark" if dark else "light"
        fig.suptitle(f"Prediction density ({suffix} background)", fontsize=11, y=1.05)
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
