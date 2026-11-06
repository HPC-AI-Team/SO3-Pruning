#!/usr/bin/env bash
set -euo pipefail

mace_run_train \
    --config="MACE/configs/mp.yaml" \
    --foundation_model="/path/to/mp_pruned.model" \
    --train_file="/path/to/train.xyz" \
    --work_dir="/path/to/mp_training"
