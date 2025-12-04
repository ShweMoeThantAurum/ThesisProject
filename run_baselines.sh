#!/bin/bash

DATASETS=("sz" "pems08" "los")
MODES=("FedAvg" "FedProx" "AEFL")

for DATASET in "${DATASETS[@]}"; do
  for MODE in "${MODES[@]}"; do
  
    echo "==============================="
    echo " Running BASELINE: DATASET=$DATASET MODE=$MODE"
    echo "==============================="

    export DATASET=$DATASET
    export FL_MODE=$MODE

    # Baseline → no variants
    export VARIANT_ID=""
    export DP_ENABLED=false
    export COMPRESSION_ENABLED=false

    # Optional: clean logs before each run
    rm -f run_logs/*.log

    # Start clients
    docker compose up -d

    # Run server
    python -m src.fl.server.main

    # Stop clients
    docker compose down

    echo "Finished run: $DATASET-$MODE"
    echo ""
  done
done

