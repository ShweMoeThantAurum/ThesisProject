#!/bin/bash

OUT_CSV="accuracy_all_datasets.csv"
echo "dataset,mode,variant,MAE,RMSE,MAPE" > $OUT_CSV

for DATASET in sz pems08 los; do
  for MODE in fedavg fedprox aefl; do

    METRICS_FILE="outputs/${DATASET}/${MODE}/final_metrics_${MODE}.json"

    if [ -f "$METRICS_FILE" ]; then
      MAE=$(jq .MAE $METRICS_FILE)
      RMSE=$(jq .RMSE $METRICS_FILE)
      MAPE=$(jq .MAPE $METRICS_FILE)
      VARIANT=$(jq -r .variant $METRICS_FILE)

      echo "${DATASET},${MODE},${VARIANT},${MAE},${RMSE},${MAPE}" >> $OUT_CSV
    else
      echo "WARNING: Missing $METRICS_FILE"
    fi

  done
done

echo "Accuracy CSV generated: $OUT_CSV"

