#!/bin/bash

OUT_CSV="energy_all_datasets.csv"
echo "dataset,mode,variant,total_energy_j" > $OUT_CSV

for DATASET in sz pems08 los; do
  for MODE in aefl fedavg fedprox; do
    FILE="outputs/$DATASET/$MODE/energy_summary.json"
    if [ -f "$FILE" ]; then
      TOTAL=$(jq .total_energy_j $FILE)
      VARIANT=$(jq -r .variant $FILE)
      echo "$DATASET,$MODE,$VARIANT,$TOTAL" >> $OUT_CSV
    fi
  done
done

echo "Energy CSV generated: $OUT_CSV"