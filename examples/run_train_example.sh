#!/usr/bin/env bash
# Smoke-test training example. n_past=100 > ENCODER_THRESHOLD (64), so this
# automatically activates the GRU-encoder architecture.
#
# NOTE: the paper's real runs use --epochs 1000 --stride 1 (full sliding-
# window data augmentation). This example uses --epochs 2 and a much larger
# --stride (fewer, non-overlapping segments) purely to verify the pipeline
# runs end-to-end quickly; it will not produce a well-trained model.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m nftsf.train \
    --data_path examples/data/alanine_phi_train.npy \
    --n_past 100 \
    --n_future 100 \
    --stride 200 \
    --epochs 2 \
    --batch_size 256 \
    --output_dir examples/results \
    --model_name alanine_phi_example
