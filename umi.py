#!/usr/bin/env python3
#
# umi.py
# 
# This file contains a reference implementation of our method of intelligently classifying
# data using morisita indices. There is also a sample program that makes use of our
# classification function. Although our sample uses a fixed, known-size dataset, our
# classification function is also suitable for use in streams of unknown length.

from collections import deque

UNCLASSIFIED = 0
NORMAL = 1
ANOMALY = 2
PENDING = 3
FALSE_POSITIVE = 4
FALSE_NEGATIVE = 5

def lce(data, mincluster=2, numbenchmarks=15):
    # Number of dimensions per datapoint
    feature_count = len(data[0])

    # Size of a quadrat (halves itself each trial)
    delta = 1

    lce = np.zeros(numbenchmarks)

    for benchmark in range(numbenchmarks):
        #  Number of quadrats per dimension. Same as 1 / delta
        q = np.power(2, benchmark)

        # Total number of quadrats
        Q = np.power(q, feature_count)

        # The domains of each cell
        # Eg: benchmark = 2, delta = 0.25
        # [0, 0.25, 0.5, 0.75, 1.0]

        # quadrat_sums contains number of datapoints in each quadrat.
        # Recall that Q = q ^ feature_count
        quadrat_sums = np.zeros((q,) * feature_count)

        for i in range(len(data)):
            quadrat = tuple(int(np.floor(data[i][f] / delta)) for f in range(feature_count))
            quadrat_sums[quadrat] += 1

        for i in np.ndenumerate(quadrat_sums):
            quadrat_sum = quadrat_sums[i[0]]

            if quadrat_sum >= m:
                continue

            product = 1
            for j in range(m):
                product *= quadrat_sum - j
            lce[benchmark] += product
        lce[benchmark] *= Q
        
        delta /= 2

    return lce

def classify(datapoint, trusted_data, last_classification):
    """Classifies data as NORMAL, ANOMALY, or PENDING.

    Args:
        datapoint: The current datapoint to classify. May be array-like for multi-dimensional data.
        trusted_data: A double-ended queque of datapoints that are trusted
        last_classification: The classification of the previous datapoint
    Returns:
        A tuple containing the revised classification for the last datapoint, and the classification for the new datapoint.
    """
    
    revision = last_classification
    classification = UNCLASSIFIED

    # Number of dimensions in the data.
    feature_count = len(datapoint)

    # Check if the datapoint is an exact duplicate of a trusted datapoint
    if datapoint in trusted_data:
        if last_classification == PENDING:
            revision = NORMAL
        classification = NORMAL

    # Check if the data lands in the same cluster as a trusted datapoint
    else:
        trusted_min = min(trusted_data)
        trusted_max = max(trusted_data)

        all_min = min(datapoint, trusted_min)
        all_max = max(datapoint, trusted_max)

        scale_min = 0.0
        scale_max = 1.0

        # Scale data linearly to range [0.0, 1.0]
        all_data_scaled = np.interp((*trusted_data, datapoint), (all_min, all_max), (scale_min, scale_max))
        trusted_data_scaled = np.interp(trusted_data, (all_min, all_max), (scale_min, scale_max))

        # Minimum cluster size
        min_cluster = 2

        # Number of times to divide quadrat size
        num_benchmarks = 15

        # Find LCE for trusted set and trusted set + new data
        all_lce = lce(all_data_scaled, min_cluster, num_benchmarks)
        trusted_lce = lce(trusted_data_scaled, min_cluster, num_benchmarks)

        # Find the smallest quadrat size with positive LCE
        lce_index = 0
        for benchmark in range(num_benchmarks):
            if trusted_lce[benchmark] > 0:
                lce_index = benchmark
        all_lce_val = all_lce[lce_index]
        trusted_lce_val = trusted_lce[lce_index]

        # If all_lce_val > trusted_lce_val, the new data resides in a pre-existing cluster
        clustered = all_lce_val > trusted_lce_val

        if clustered:
            classification = NORMAL

            if last_classification == PENDING:
                revision = NORMAL

        else:
            classification = PENDING

            if last_classification == PENDING:
                revision = ANOMALY

    # If the new data is NORMAL, the replace the oldest trusted data with the new data
    if classification == NORMAL:
        trusted_data.popleft()

    # If pending data was normal, remove the oldest datapoint that we didn't remove last time
    if last_value == PENDING and revision == NORMAL:
        trusted_data.popleft()

    # If the pending data was anomalous, remove it from the trusted dataset
    if revision == ANOMALY:
        trusted_data.pop()

    # Keep track of current datapoint for future reference
    trusted_data.append(datapoint)

    return revision, classification


if __name__ == '__main__':
    import numpy as np
    import argparse

    DATA_PATH = './nab/realTweets/realTweets/Twitter_volume_%s.csv'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sample of LCE method for detecting anomalies in twitter dataset')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    args = parser.parse_args()
    dataset = args.dataset
    trusted_size = args.trusted_size
    
    # Load specified dataset into memory.
    all_data = np.loadtxt(open(DATA_PATH % (dataset), 'rb'), delimiter=",", skiprows=1, usecols=1, dtype=int)
    data_size = len(all_data)
    
    # Create results buffer.
    # The trusted data are marked as normal.
    # Remaining data is still unclassified.
    results = np.zeros(data_size, dtype=int)
    results[:trusted_size] = np.full(trusted_size, NORMAL, dtype=int)
    
    # Create the trusted buffer
    # This buffer contains the current set of data that are known to be trustworthy.
    # At the moment, it just contains the first `trusted_size` datapoints.
    # We add space for two additional indices for pending data which is temporarily trusted.
    trusted_data = deque(maxlen=trusted_size + 2)
    for i in range(0, trusted_size): trusted_data.append(all_data[i])

    # Classify the remaining data
    for i in range(trusted_size, data_size):
        # The new, unclassified data
        datapoint = all_data[i]

        # The result of the last classification. For the first result, this is NORMAL
        last_classification = results[i - 1]

        # Get a revision for the last classification if it was pending, and a classification for the new data.
        revision, fresh = classify(datapoint, trusted_data, last_classification)

        # Store our results
        results[i - 1] = revision
        results[i] = fresh
    
