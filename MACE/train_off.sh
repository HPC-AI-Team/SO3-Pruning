#!/usr/bin/env bash
set -euo pipefail

mace_run_train \
    --config="MACE/configs/off.yaml" \
    --foundation_model="/path/to/off_pruned.model" \
    --train_file="/path/to/train.xyz" \
    --test_file="/path/to/test.xyz" \
    --work_dir="/path/to/off_training"
