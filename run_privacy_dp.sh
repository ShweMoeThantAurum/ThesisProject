#!/bin/bash

# Run AEFL + DP experiments for multiple datasets and sigma values.

DATASETS=("sz" "pems08" "los")
SIGMAS=("0.0" "0.01" "0.05" "0.10" "0.20")

export FL_MODE="aefl"
export COMPRESSION_ENABLED=false
export DP_ENABLED=true

for DATASET in "${DATASETS[@]}"; do
  for SIGMA in "${SIGMAS[@]}"; do

    echo "============================================"
    echo " RUNNING DP EXPERIMENT | DATASET=$DATASET | SIGMA=$SIGMA"
    echo "============================================"

    export DATASET="$DATASET"
    export DP_SIGMA="$SIGMA"
    export VARIANT_ID="dp_sigma_${SIGMA}"

    # Clean logs for a fresh run of this (dataset, sigma)
    rm -f run_logs/*.log

    # Restart client containers
    docker compose down -v
    docker compose up -d

    # Run server orchestration (will call generate_cloud_summary,
    # which now writes correct per-variant energy summaries)
    python -m src.fl.server.main

    # Shutdown clients cleanly
    docker compose down -v

  done
done

