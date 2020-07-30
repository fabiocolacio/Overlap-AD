#!/bin/sh

max_jobs=4
cur_jobs=0

if [ -n "$1" ]
then
    max_jobs="$1"
fi

algos=(lof moving-average random-forest knn svm naive-bayes decision-tree)
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
	outfile="results/$algo-$dset.csv"

	if [ "$algo" = "knn" ]
	then
	    knn_max=20

	    if [ "$knn_max" -gt "$win" ]
	    then
		knn_max="$win"
	    fi
	    	    
	    knn_ks=$(seq 5 5 $kmax)
	    knn_threshs=$(seq 10 10 100)

	    for k in ${knn_ks[@]}
	    do
		for thresh in ${knn_threshs[@]}
		do
		    ./anomaly-detection.py \
			--algorithm "$algo" \
			--train "$win" \
			--threshold "$thresh" \
			-k "$k" \
			--twitter "$dset" \
			>> "$outfile" &

		    manage_jobs
		done
	    done
	elif [ "$algo" = "lof" ]
	then
	    lof_k_max=20

	    if [ "$lof_k_max" -gt "$win" ]
	    then
		lof_k_max="$win"
	    fi
	    
	    lof_ks=$(seq 5 5 $lof_k_max)

	    for lof_k in ${lof_ks[@]}
	    do
		./anomaly-detection.py \
		    --algorithm "$algo" \
		    --train "$win" \
		    -k "$lof_k" \
		    --twitter "$dset" \
		    >> "$outfile" &
	    done
	elif [ "$algo" = "moving-average" ]
	then
	    moving_average_threshs=$(seq 1 5)

	    for moving_average_thresh in ${moving_average_threshs[@]}
	    do
		./anomaly-detection.py \
		    --algorithm "$algo" \
		    --train "$win" \
		    --threshold "$moving_average_thresh" \
		    --twitter "$dset" \
		    >> "$outfile" &

		manage_jobs
	    done
	else
	    ./anomaly-detection.py \
		--algorithm "$algo" \
		--train "$win" \
		--twitter "$dset" \
		>> "$outfile" &

	    manage_jobs
	fi
    done
}

yahoo() {
    dsets=(real_7 real_19)

    for dset in ${dsets[@]}
    do
	outfile="results/$algo-$dset.csv"

	if [ "$algo" = "knn" ]
	then
	    knn_max=20

	    if [ "$knn_max" -gt "$win" ]
	    then
		knn_max="$win"
	    fi
	    
	    knn_ks=$(seq 5 5 $kmax)
	    knn_threshs=$(seq 10 10 100)

	    for k in ${knn_ks[@]}
	    do
		for thresh in ${knn_threshs[@]}
		do
		    ./anomaly-detection.py \
			--algorithm "$algo" \
			--train "$win" \
			--threshold "$thresh" \
			-k "$k" \
			--yahoo \
			-if "A1Benchmark/$dset.csv" \
			>> "$outfile" &

		    manage_jobs
		done
	    done
	elif [ "$algo" = "lof" ]
	then
	    lof_k_max=20

	    if [ "$lof_k_max" -gt "$win" ]
	    then
		lof_k_max="$win"
	    fi
	    
	    lof_ks=$(seq 5 5 $lof_k_max)

	    for lof_k in ${lof_ks[@]}
	    do
		./anomaly-detection.py \
		    --algorithm "$algo" \
		    --train "$win" \
		    -k "$lof_k" \
		    -if "A1Benchmark/$dset.csv" \
		    --yahoo \
		    >> "$outfile" &
	    done
	elif [ "$algo" = "moving-average" ]
	then
	    moving_average_threshs=$(seq 1 5)

	    for moving_average_thresh in ${moving_average_threshs[@]}
	    do
		./anomaly-detection.py \
		    --algorithm "$algo" \
		    --train "$win" \
		    --threshold "$moving_average_thresh" \
		    --yahoo \
		    -if "A1Benchmark/$dset.csv" \
		    >> "$outfile" &

		manage_jobs
	    done
	else
	    ./anomaly-detection.py \
		--algorithm "$algo" \
		--train "$win" \
		--yahoo \
		-if "A1Benchmark/$dset.csv" \
		>> "$outfile" &

	    manage_jobs
	fi
    done
}

for algo in ${algos[@]}
do
    for win in ${wins[@]}
    do
	twitter
	yahoo
    done
done

wait
