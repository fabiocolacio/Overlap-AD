#!/usr/bin/env python3

import umi
import numpy as np
import os
import argparse
from collections import deque

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--infile', dest='infile', type=str, help='Path to the dataset to test')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    parser.add_argument('-s', '--threshold', dest='threshold', type=float)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-c', '--csv', dest='csv', action='store_true')
    parser.set_defaults(verbose=False)
    parser.set_defaults(csv=False)

    args = parser.parse_args()
    infile = args.infile
    trusted_size = args.trusted_size
    threshold = args.threshold
    verbose = args.verbose
    csv = args.csv

    all_data = np.loadtxt(infile, delimiter=',', skiprows=1, usecols=1, dtype=float)
    labels = np.loadtxt(infile, delimiter=',', skiprows=1, usecols=2, dtype=int)

    discarded_anomalies, i, j = 0, 0, trusted_size
    while i < j:
        if labels[i] == 1:
            labels[i] = -1
            discarded_anomalies += 1
            j += 1
        i += 1

    data_size = len(all_data)

    results = np.zeros(data_size, dtype=int)
    results[:trusted_size] = np.full(trusted_size, umi.NORMAL, dtype=int)
    
    trusted_data = deque(maxlen=trusted_size + 2)
    for i in range(0, j):
        if labels[i] == 0:
            trusted_data.append((all_data[i],))

    anomalies = 0

    for i in range(j, data_size):
        datapoint = np.array((all_data[i],))

        last_classification = results[i - 1]

        revision, fresh = umi.classify(datapoint, trusted_data, last_classification, threshold)

        results[i - 1] = revision
        results[i] = fresh

        if verbose:
            print("Point %d classified as %d" % (i - 1, revision))

        if revision == umi.ANOMALY:
            anomalies += 1

    if verbose:
        print("Classified %d anomalies out of %d points" % (anomalies, data_size - trusted_size))
   
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(results)):
        if labels[i] == 0 and results[i] == umi.NORMAL:
            true_neg += 1
        elif labels[i] == 0 and results[i] == umi.ANOMALY:
            false_pos += 1
        elif labels[i] == 1 and results[i] == umi.NORMAL:
            false_neg += 1
        elif labels[i] == 1 and results[i] == umi.ANOMALY:
            true_pos += 1

    if csv:
        print("%10s %10d %10f %10s %10s %10s %10s %10s" % (os.path.basename(infile), trusted_size, threshold, true_pos, true_neg, false_pos, false_neg, discarded_anomalies))
    else:
        print("Discarded Anomalies:", discarded_anomalies)
        print("True Positives:", true_pos)
        print("True Negatives:", true_neg)
        print("False Positives:", false_pos)
        print("False Negatives:", false_neg)

