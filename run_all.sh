#!/usr/bin/sh 

# Run all tests in separate processes

python3 run.py AMZN 1500 2 &
python3 run.py CRM 1500 2 &
python3 run.py FB 1500 2 &
python3 run.py GOOG 1500 2 &
python3 run.py IBM 1500 2 &
python3 run.py KO 1500 2 &
python3 run.py PFE 1500 2 &
python3 run.py UPS 1500 2 &

