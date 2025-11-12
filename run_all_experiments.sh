#!/bin/bash
set -e

echo "=============================================================="
echo " AEFL CLOUD EXPERIMENTS – FULL PIPELINE EXECUTION"
echo "=============================================================="
echo "Started at: $(date)"
echo

ROOT_DIR=$(pwd)
LOG_DIR="$ROOT_DIR/run_logs"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
RUN_LOG="$LOG_DIR/run_$TS.log"

echo "Logs will be stored at: $RUN_LOG"
echo

# ---------------------------------------
# Helper function
# ---------------------------------------
run_exp () {
    CMD=$1
    NAME=$2

    echo "----------------------------------------------------------"
    echo "Running: $NAME"
    echo "Command: $CMD"
    echo "----------------------------------------------------------"

    {
        echo "[START] $NAME - $(date)"
        eval "$CMD"
        RET=$?
        echo "[END] $NAME - exit code $RET - $(date)"
        echo
    } >> "$RUN_LOG" 2>&1

    if [ $RET -ne 0 ]; then
        echo "⚠️  WARNING: $NAME FAILED (exit code $RET)"
    else
        echo "✔ Completed: $NAME"
    fi

    echo
}

# ---------------------------------------
# PREPROCESSING (ensure data prepared)
# ---------------------------------------

echo "=== STEP 1: PREPROCESSING DATA (SZ, LOS, PeMS08) ==="
run_exp "python -m data.preprocess_sz"   "Preprocess SZ"
run_exp "python -m data.preprocess_los"  "Preprocess LosLoop"
run_exp "python -m data.preprocess_pems08" "Preprocess PeMSD8"


# ---------------------------------------
# AEFL Experiments
# ---------------------------------------
echo "=== STEP 2: AEFL EXPERIMENTS ==="
run_exp "python -m src.experiments.run_aefl --config configs/aefl_sz.yaml"      "AEFL SZ"
run_exp "python -m src.experiments.run_aefl --config configs/aefl_los.yaml"     "AEFL LosLoop"
run_exp "python -m src.experiments.run_aefl --config configs/aefl_pems08.yaml"  "AEFL PeMSD8"


# ---------------------------------------
# Centralized Baselines
# ---------------------------------------
echo "=== STEP 3: CENTRALIZED BASELINES ==="
run_exp "python -m src.experiments.run_centralized --config configs/centralized_sz.yaml"     "Centralized SZ"
run_exp "python -m src.experiments.run_centralized --config configs/centralized_los.yaml"    "Centralized LosLoop"
run_exp "python -m src.experiments.run_centralized --config configs/centralized_pems08.yaml" "Centralized PeMSD8"


# ---------------------------------------
# FedAvg Baselines
# ---------------------------------------
echo "=== STEP 4: FEDAVG BASELINES ==="
run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_sz.yaml"      "FedAvg SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_los.yaml"     "FedAvg LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_pems08.yaml"  "FedAvg PeMSD8"


# ---------------------------------------
# FedProx Baselines
# ---------------------------------------
echo "=== STEP 5: FEDPROX BASELINES ==="
run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_sz.yaml"      "FedProx SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_los.yaml"     "FedProx LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_pems08.yaml"  "FedProx PeMSD8"


# ---------------------------------------
# Local-Only Baselines
# ---------------------------------------
echo "=== STEP 6: LOCAL-ONLY BASELINES ==="
run_exp "python -m src.experiments.run_localonly --config configs/localonly_sz.yaml"      "LocalOnly SZ"
run_exp "python -m src.experiments.run_localonly --config configs/localonly_los.yaml"     "LocalOnly LosLoop"
run_exp "python -m src.experiments.run_localonly --config configs/localonly_pems08.yaml"  "LocalOnly PeMSD8"


# ---------------------------------------
# Periodic k=2 Baselines
# ---------------------------------------
echo "=== STEP 7: PERIODIC (k=2) BASELINES ==="
run_exp "python -m src.experiments.run_fedavg --config configs/periodic2_sz.yaml"      "Periodic SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/periodic2_los.yaml"     "Periodic LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/periodic2_pems08.yaml"  "Periodic PeMSD8"


# ---------------------------------------
# Top-K Compression Baselines
# ---------------------------------------
echo "=== STEP 8: TOP-K COMPRESSION BASELINES ==="
run_exp "python -m src.experiments.run_fedavg --config configs/topk_sz.yaml"      "TopK SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/topk_los.yaml"     "TopK LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/topk_pems08.yaml"  "TopK PeMSD8"


# ---------------------------------------
# Q8 Quantization Baselines
# ---------------------------------------
echo "=== STEP 9: Q8 QUANTIZATION BASELINES ==="
run_exp "python -m src.experiments.run_fedavg --config configs/q8_sz.yaml"      "Q8 SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/q8_los.yaml"     "Q8 LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/q8_pems08.yaml"  "Q8 PeMSD8"


# ---------------------------------------
# FINAL STEP — SUMMARY & PLOTS
# ---------------------------------------
echo "=== STEP 10: ANALYSIS / PLOTS ==="
run_exp "python -m src.analysis.make_plots" "Generate Plots"


echo "=============================================================="
echo " ALL EXPERIMENTS COMPLETED"
echo " Results uploaded to S3 bucket: aefl-results"
echo " Full run log stored at: $RUN_LOG"
echo " Finished at: $(date)"
echo "=============================================================="
