#!/bin/bash

trap 'echo "Interrupted!"; kill 0; exit 130' SIGINT SIGTERM

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 '<out_name>|<benchmark>|<arg_json>'"
    exit 1
fi

IFS='|' read -r out_name benchmark arg_json <<< "$1"

echo "=== name=$out_name benchmark=$benchmark $arg_json"

if [[ "$out_name" = "" ]]; then
  echo NOTHING.
  exit 0
fi

if [[ "$out_name" =~ ^# ]]; then
  echo SKIP.
  exit 0
fi

TRAJ_ANALYSIS="$arg_json" \
python3 swebench_main.py \
  --benchmark "$benchmark" \
  --log_path "../out/$out_name/log" \
  --patches_path "../out/$out_name/patch" \
  --output_path "../out/$out_name/output"

echo "=== FINISHED"
date
