#!/bin/sh

n_threads=4
outfile="results.csv"

active_threads=0
dsets=(AAPL GOOG FB AMZN)
tsizes=(0 10 100 50 100 150 200 250 300 350 400 450 500)

echo "Beginning test."

echo "dataset,trainingsize,tp,tn,fp,fn" > "$outfile"
for dset in ${dsets[@]}
do
    for tsize in ${tsizes[@]}
    do
        (printf "%s,%d," "$dset" "$tsize" | \
            glue ./svm.py -t "$tsize" --twitter "$dset") \
            >> "$outfile" &

        active_threads=$((active_threads + 1))

        if [ $active_threads -ge $n_threads ];
        then
            wait -n
            active_threads=$((active_threads - 1))
        fi
    done
done

wait

echo "Done testing."
