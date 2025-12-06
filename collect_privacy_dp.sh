#!/bin/bash

# Collect accuracy + energy results for DP experiments into a single CSV:
#   privacy_dp_results.csv
#
# It expects:
#   outputs/<dataset>/aefl/final_metrics_aefl_dp_sigma_<SIGMA>.json
#   outputs/<dataset>/aefl/energy_summary_dp_sigma_<SIGMA>.json
#
# After the Python changes, energy_summary_* now contains correct
# non-zero total_energy_j for each DP sigma.

OUT="privacy_dp_results.csv"
echo "dataset,sigma,MAE,RMSE,MAPE,total_energy_j" > "$OUT"

SIGMAS=("0.0" "0.01" "0.05" "0.10" "0.20")

for DATASET in sz pems08 los; do
  for SIGMA in "${SIGMAS[@]}"; do

    VARIANT="dp_sigma_${SIGMA}"
    METRICS="outputs/${DATASET}/aefl/final_metrics_aefl_${VARIANT}.json"
    ENERGY="outputs/${DATASET}/aefl/energy_summary_${VARIANT}.json"

    if [ ! -f "$METRICS" ]; then
      echo "Missing $METRICS"
      continue
    fi

    MAE=$(jq .MAE "$METRICS")
    RMSE=$(jq .RMSE "$METRICS")
    MAPE=$(jq .MAPE "$METRICS")

    if [ -f "$ENERGY" ]; then
      TOTAL=$(jq .total_energy_j "$ENERGY")
    else
      TOTAL="0"
    fi

    echo "$DATASET,$SIGMA,$MAE,$RMSE,$MAPE,$TOTAL" >> "$OUT"
  done
done

echo "[OK] DP CSV saved → $OUT"

