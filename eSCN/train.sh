#!/usr/bin/env bash
set -euo pipefail

fairchem --mode train \
    --config-yml="eSCN/configs/train_1m.yml" \
    --checkpoint="/path/to/escn_pruned.pt" \
    --run-dir="/path/to/escn_training" \
    --num-gpus=8 \
    --seed=0
