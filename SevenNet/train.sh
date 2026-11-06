#!/usr/bin/env bash
set -euo pipefail

sevenn train SevenNet/configs/SevenNet-0.yaml \
    --working_dir="/path/to/sevennet_training"
