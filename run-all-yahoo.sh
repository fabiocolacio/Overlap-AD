#!/usr/bin/env bash

echo "Running tests..."

sizes=(70 140 210 280)
threshs=(0.005 0.006 0.007 0.008 0.009 0.01)

threads=$1
thread_count=0

echo "Running with $threads threads"

outfile="yahoo-results.txt"

echo "filename   true_pos   true_neg   false_pos  false_neg" > $outfile

for dset in {1..67..1}; do
    file="data/yahoo/yahoo/A1Benchmark/real_$dset.csv"
    for size in ${sizes[*]}; do
        for thresh in ${threshs[*]}; do
            if [ $thread_count -lt $threads ]; then
                ./yahoo-test.py -if $file -t $size -s $thresh -c >> $outfile &
                thread_count=$((thread_count+1))
            else
                wait
                thread_count=0
            fi

            echo "$thread_count running tests"
        done
        wait
    done
done

echo "All tests complete."

