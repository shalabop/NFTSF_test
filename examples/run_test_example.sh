#!/usr/bin/env bash
# Evaluates the checkpoint produced by run_train_example.sh against the
# held-out test trajectories, printing CRPS/MAE/RMSE/CI50/CI90 coverage.
#
# n_past is read from the training run's config.json, not passed here, so
# the same automatic encoder/non-encoder selection applies on reload.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_PATH="$(ls -t examples/results/config_*.json | head -n 1)"
if [ -z "${CONFIG_PATH}" ]; then
    echo "No config_*.json found in examples/results/ — run run_train_example.sh first." >&2
    exit 1
fi

python3 -m nftsf.test \
    --config_path "${CONFIG_PATH}" \
    --data_path examples/data/alanine_phi_test.npy \
    --n_samples 50 \
    --max_segments 5 \
    --output_dir examples/test_results
