#!/usr/bin/env python3

import umi
import numpy as np
import argparse
import json
from collections import deque

if __name__ == '__main__':
    DATA_PATH = './nab/realTweets/realTweets/Twitter_volume_%s.csv'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sample of LCE method for detecting anomalies in twitter dataset')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    parser.add_argument('-s', '--threshold', dest='threshold', type=float)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    dataset = args.dataset
    trusted_size = args.trusted_size
    threshold = args.threshold
    verbose = args.verbose
    
    # Load specified dataset into memory.
    all_data = np.loadtxt(open(DATA_PATH % (dataset), 'rb'), delimiter=",", skiprows=1, usecols=1, dtype=int)
    timestamps = np.loadtxt(open(DATA_PATH % (dataset), 'rb'), delimiter=",", skiprows=1, usecols=0, dtype=str)
    data_size = len(all_data)
    
    # Create results buffer.
    # The trusted data are marked as normal.
    # Remaining data is still unclassified.
    results = np.zeros(data_size, dtype=int)
    results[:trusted_size] = np.full(trusted_size, umi.NORMAL, dtype=int)
    
    # Create the trusted buffer
    # This buffer contains the current set of data that are known to be trustworthy.
    # At the moment, it just contains the first `trusted_size` datapoints.
    # We add space for two additional indices for pending data which is temporarily trusted.
    trusted_data = deque(maxlen=trusted_size + 2)
    for i in range(0, trusted_size): trusted_data.append((all_data[i],))

    anomalies = 0

    # Classify the remaining data
    for i in range(trusted_size, data_size):
        # The new, unclassified data
        datapoint = np.array((all_data[i],))

        # The result of the last classification. For the first result, this is NORMAL
        last_classification = results[i - 1]

        # Get a revision for the last classification if it was pending, and a classification for the new data.
        revision, fresh = umi.classify(datapoint, trusted_data, last_classification, threshold)

        # Store our results
        results[i - 1] = revision
        results[i] = fresh

        if verbose:
            print("Point %d classified as %d" % (i-1, revision))

        if revision == umi.ANOMALY:
            anomalies += 1

    print("Classified %d anomalies out of %d points" % (anomalies, data_size - trusted_size))
    
    fh = open("labels/combined_labels.json")
    labels_raw = fh.read()
    fh.close()
    anomalies = json.loads(labels_raw)["realTweets/Twitter_volume_%s.csv" % (dataset)]

    true_pos, true_neg, false_pos, false_neg, = 0, 0, 0, 0
    for i in range(len(results)):
        if timestamps[i] in anomalies:
            if results[i] == umi.ANOMALY:
                results[i] = umi.TRUE_POSITIVE
                true_pos += 1
            else:
                results[i] = umi.FALSE_NEGATIVE
                false_neg += 1
        else:
            if results[i] == umi.NORMAL:
                results[i] = umi.TRUE_NEGATIVE
                true_neg += 1
            else:
                results[i] = umi.FALSE_POSITIVE
                false_pos += 1

    print("True Positives:", true_pos)
    print("True Negatives:", true_neg)
    print("False Positives:", false_pos)
    print("False Negatives:", false_neg)

