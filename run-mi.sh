#!/bin/sh

max_jobs=4
cur_jobs=0

if [ -n "$1" ]
then
    max_jobs="$1"
fi

threshs=$(seq 0 0.003125 0.05)
wins=$(echo 5; echo 10; seq 50 50 500)

mkdir -p results

manage_jobs() {
    cur_jobs=$((cur_jobs+1))
    if [ "$cur_jobs" -gt "$max_jobs" ]
    then
	wait -n
	cur_jobs=$((cur_jobs-1))
    fi
}

twitter() {
    dsets=(AAPL GOOG AMZN FB)

    for dset in ${dsets[@]}
    do
	outfile="results/mi-$dset.csv"
	
	./newumi.py --win-size "$win" \
		    --thresh "$thresh" \
		    --twitter "$dset" \
		    >> "$outfile" &

	manage_jobs
    done
}

yahoo() {
    dsets=(real_7 real_19)

    for dset in ${dsets[@]}
    do
	outfile="results/mi-$dset.csv"

	./newumi.py --win-size "$win" \
		    --thresh "$thresh" \
		    --yahoo \
		    -if "A1Benchmark/$dset.csv" \
		    >> "$outfile" &

	manage_jobs
    done
}

for win in ${wins[@]}
do
    for thresh in ${threshs[@]}
    do
	twitter
	yahoo
    done
done

wait
