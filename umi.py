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

def classify(datapoint, trusted_data, last_classification):
    """Classifies data as NORMAL, ANOMALY, or PENDING.

    Args:
        datapoint: The current datapoint to classify
        trusted_data: A double-ended queque of datapoints that are trusted
        last_classification: The classification of the previous datapoint
    Returns:
        A tuple containing the revised classification for the last datapoint, and the classification for the new datapoint.
    """
    
    revision = last_classification
    classification = UNCLASSIFIED

    # Check if the datapoint is an exact duplicate of a trusted datapoint
    if datapoint in trusted_data:
        if last_classification == PENDING:
            revision = NORMAL
        classification = NORMAL

    # Check if the data lands in the same cluster as a trusted datapoint
    else:
        elif clustered:
            classification = NORMAL

            if last_classification == PENDING:
                revision = NORMAL

        else:
            classification = PENDING

            if last_classification == PENDING:
                revision = ANOMALY

    # Keep track of current datapoint for future reference
    trusted_data.append(datapoint)

    # If the new data is NORMAL, the replace the oldest trusted data with the new data
    if classification == NORMAL:
        trusted_data.popleft()

    # If pending data was normal, remove the oldest datapoint that we didn't remove last time
    if last_value == PENDING and revision == NORMAL:
        trusted_data.popleft()

    # If the pending data was anomalous, remove it from the trusted dataset
    if revision == ANOMALY:
        trusted_data.pop()

    return revision, classification


if __name__ == '__main__':
    import numpy as np
    import argparse

    DATA_PATH = './nab/realTweets/realTweets/Twitter_volume_%s.csv'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sample of LCE method for detecting anomalies in twitter dataset')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    parser.add_argument('-m', '--min-cluster', dest='mincluster', type=int, help='The minimum cluster number')
    args = parser.parse_args()
    dataset = args.dataset
    trusted_size = args.trusted_size
    mincluster = args.mincluster
    
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
    
