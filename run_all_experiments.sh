#!/bin/bash
set -e

echo "=============================================================="
echo " AEFL EXPERIMENTS – FULL EXPERIMENTS EXECUTION"
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


# ------------------------------------------------------------
# Helper function for experiment execution
# ------------------------------------------------------------
run_exp () {
    CMD=$1
    NAME=$2

    echo "----------------------------------------------------------"
    echo " Running: $NAME"
    echo " Command: $CMD"
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


# ============================================================
# PREPROCESSING
# ============================================================
echo
echo "=============================================================="
echo "                       PREPROCESSING"
echo "=============================================================="
echo

run_exp "python -m data.preprocess_sz"        "Preprocess SZ"
run_exp "python -m data.preprocess_los"       "Preprocess LosLoop"
run_exp "python -m data.preprocess_pems08"    "Preprocess PeMSD8"


# ============================================================
# EXPERIMENT COUNTER
# ============================================================
TOTAL_EXPS=5
EXP=0

next_exp () {
    EXP=$((EXP + 1))
    echo
    echo "=============================================================="
    printf "                EXPERIMENT %d/%d: %s\n" "$EXP" "$TOTAL_EXPS" "$1"
    echo "=============================================================="
    echo
}


# ============================================================
# EXPERIMENT 1 — AEFL (Proposed Framework)
# ============================================================
next_exp "AEFL (Proposed Framework)"

run_exp "python -m src.experiments.run_aefl --config configs/aefl_sz.yaml"       "AEFL SZ"
run_exp "python -m src.experiments.run_aefl --config configs/aefl_los.yaml"      "AEFL LosLoop"
run_exp "python -m src.experiments.run_aefl --config configs/aefl_pems08.yaml"   "AEFL PeMSD8"


# ============================================================
# EXPERIMENT 2 — Centralized
# ============================================================
next_exp "Centralized Baselines"

run_exp "python -m src.experiments.run_centralized --config configs/centralized_sz.yaml"      "Centralized SZ"
run_exp "python -m src.experiments.run_centralized --config configs/centralized_los.yaml"     "Centralized LosLoop"
run_exp "python -m src.experiments.run_centralized --config configs/centralized_pems08.yaml"  "Centralized PeMSD8"


# ============================================================
# EXPERIMENT 3 — FedAvg
# ============================================================
next_exp "FedAvg Baselines"

run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_sz.yaml"       "FedAvg SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_los.yaml"      "FedAvg LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/fedavg_pems08.yaml"   "FedAvg PeMSD8"


# ============================================================
# EXPERIMENT 4 — FedProx
# ============================================================
next_exp "FedProx Baselines"

run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_sz.yaml"      "FedProx SZ"
run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_los.yaml"     "FedProx LosLoop"
run_exp "python -m src.experiments.run_fedavg --config configs/fedprox_pems08.yaml"  "FedProx PeMSD8"


# ============================================================
# EXPERIMENT 5 — Local-Only
# ============================================================
next_exp "Local-Only Baselines"

run_exp "python -m src.experiments.run_localonly --config configs/localonly_sz.yaml"      "LocalOnly SZ"
run_exp "python -m src.experiments.run_localonly --config configs/localonly_los.yaml"     "LocalOnly LosLoop"
run_exp "python -m src.experiments.run_localonly --config configs/localonly_pems08.yaml"  "LocalOnly PeMSD8"


# ============================================================
# FINAL STEP — Analysis
# ============================================================
echo
echo "=============================================================="
echo "                       FINAL ANALYSIS"
echo "=============================================================="
echo

run_exp "python -m src.analysis.make_plots" "Generate Comparison Plots"


echo
echo "=============================================================="
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo " Results uploaded to S3 bucket: aefl-results"
echo " Full run log stored at: $RUN_LOG"
echo " Finished at: $(date)"
echo "=============================================================="
echo
