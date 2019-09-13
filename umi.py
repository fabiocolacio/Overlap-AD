#!/usr/bin/env python3
#
# umi.py
# 
# This file contains a reference implementation of our method of intelligently classifying
# data using morisita indices. There is also a sample program that makes use of our
# classification function. Although our sample uses a fixed, known-size dataset, our
# classification function is also suitable for use in streams of unknown length.

from collections import deque
import numpy as np

UNCLASSIFIED = 0
NORMAL = 1
ANOMALY = 2
PENDING = 3
FALSE_POSITIVE = 4
FALSE_NEGATIVE = 5
TRUE_POSITIVE = 6
TRUE_NEGATIVE = 7

def euclidean_dist(a, b):
    """Euclidean distance formula for two n-dimensional points.

    Args:
        a: The first point
        b: The second point
    Returns:
        The euclidean distance of the two points.
    """

    return np.linalg.norm(a - b)

def in_threshold(point, dset, thresh):
    """Checks if the given point is within the distance threshold of any point in dset.
    
    Args:
        point: The point to check
        dset: The dataset to check
        thresh: The euclidean distance threshold to check for point.
    Returns: 
        True if point is within the distance threshold for one or more points within dset.
        False otherwise.
    """

    for other in dset:
        if euclidean_dist(point, other) <= thresh:
            return True
    return False

def lce(data, min_cluster=2, num_benchmarks=15):
    """Calculates the LCE of a given dataset.
    
    Args:
        data: The data to find the LCE of
        min_cluster: The minimum amount of datapoints needed to form a cluster in one quadrat.
        num_benchmarks: The maximum number of times to halve the quadrat size.
    Returns:
        An array of size num_benchmarks containing the lce for each quadrat size.
    """

    # Number of dimensions per datapoint
    feature_count = len(data[0])

    # Size of a quadrat (halves itself each trial)
    delta = 1

    lce = np.zeros(num_benchmarks, dtype=int)

    for benchmark in range(num_benchmarks):
        # Contains the number of points in each quadrat
        quadrat_sums = {}

        # Find sum of points in each quadrat
        for point in data:
            quadrat = tuple(int(np.ceil(point[f] / delta)) - 1 if point[f] != 0 else 0 for f in range(feature_count))
            if quadrat in quadrat_sums:
                quadrat_sums[quadrat] += 1
            else:
                quadrat_sums[quadrat] = 1

        # Solve n!/(n - m)! for each quadrat, and add to LCE
        for key, value in quadrat_sums.items():
            if value < min_cluster:
                continue

            product = 1
            for n in range(min_cluster):
                 product *= (value - n)
            lce[benchmark] += product
        
        delta /= 2

    return lce

def classify(datapoint, trusted_data, last_classification, threshold=0):
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

    # Number of dimensions per datapoint
    feature_count = len(datapoint)

    all_data = np.array([*np.unique(trusted_data, axis=0), datapoint], dtype=float, copy=True)

    # Minimum and maximum value for each dimension
    feature_mins = np.amin(all_data, axis=0)
    feature_maxs = np.amax(all_data, axis=0)

    # Normalize the data to the range [0.0, 1.0]
    for feature in range(feature_count):
        for i in range(len(all_data)):
            if all_data[i][feature] == feature_mins[feature] and feature_mins[feature] == feature_maxs[feature]:
                all_data[i][feature] = 0.0
            else:
                all_data[i][feature] = (all_data[i][feature] - feature_mins[feature]) / (feature_maxs[feature] - feature_mins[feature])

    # Check if the datapoint is an exact duplicate of a trusted datapoint
    if in_threshold(all_data[-1],  all_data[:-1], threshold):
        if last_classification == PENDING:
            revision = NORMAL
        classification = NORMAL

    # Check if the data lands in the same cluster as a trusted datapoint
    else:

        # Minimum cluster size
        min_cluster = 2

        # Number of times to divide quadrat size
        num_benchmarks = 15

        # Find LCE for trusted set and trusted set + new data
        all_lce = lce(all_data, min_cluster, num_benchmarks)
        trusted_lce = lce(all_data[:-1], min_cluster, num_benchmarks)

        all_lce_val = 0
        trusted_lce_val = 0
        # If all_lce_val > trusted_lce_val, the new data resides in a pre-existing cluster
        for lce_val in all_lce:
            if lce_val > 0:
                all_lce_val = lce_val
        for lce_val in trusted_lce:
            if lce_val > 0:
                trusted_lce_val = lce_val

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
    if last_classification == PENDING and revision == NORMAL:
        trusted_data.popleft()

    # If the pending data was anomalous, remove it from the trusted dataset
    if revision == ANOMALY:
        trusted_data.pop()

    # Keep track of current datapoint for future reference
    trusted_data.append(datapoint)

    return revision, classification

