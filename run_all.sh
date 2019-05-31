#!/usr/bin/env bash

echo "Running tests..."

# Run all tests in separate processes
python3 umi.py -d AMZN -t 1500 -v > "results/AMZN_1500.txt" &
python3 umi.py -d CRM  -t 1500 -v > "results/CRM_1500.txt"  &
python3 umi.py -d FB   -t 1500 -v > "results/FB_1500.txt"   &
python3 umi.py -d GOOG -t 1500 -v > "results/GOOG_1500.txt" &
python3 umi.py -d IBM  -t 1500 -v > "results/IBM_1500.txt"  &
python3 umi.py -d KO   -t 1500 -v > "results/KO_1500.txt"   &
python3 umi.py -d PFE  -t 1500 -v > "results/PFE_1500.txt"  &
python3 umi.py -d UPS  -t 1500 -v > "results/UPS_1500.txt"  &

wait

echo "All tests complete."

