#!/usr/bin/env bash

echo "Running tests..."

sizes=(250 500 750 1000 1250 1500)
dsets=(AMZN CRM FB GOOG IBM KO PFE UPS)

for dset in ${dsets[*]}; do
    for size in ${sizes[*]}; do
        python3 umi.py -d $dset -t $size -v > "results/${dset}_${size}.txt" &
    done
done

wait

echo "All tests complete."

