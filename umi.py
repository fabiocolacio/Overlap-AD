#!/usr/bin/env python3
#
# umi.py
# 
# This file contains a reference implementation of our method of intelligently classifying
# data using morisita indices. There is also a sample program that makes use of our
# classification function. Although our sample uses a fixed, known-size dataset, our
# classification function is also suitable for use in streams of unknown length.

from itertools import count
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

def classify(datapoint, trusted_data, last_classification, threshold):
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
            if np.array_equiv(all_data[i][feature], feature_mins[feature]) and np.array_equiv(all_data[i][feature], feature_maxs[feature]):
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

class classifier:
    """classifier is a lazy iterator that calls classify repeatedly. It is sugar that can be used as an alternative to using classify() in a loop.

    Args:
        data: The data to test/classify. This can be a sequence such as a list or numpy array,
        or it can be a lazy iterator that returns the next datapoint in a streaming context.
        trusted_data: A deque containing the trusted_data for the system.
        of list or array, but will internally copied into a deque, as required by classify().
        threshold: The euclidean distance threshold. See classify() for details.
    Returns:
        An iterator that returns a tuple containing (rev, fresh) where rev is the revised classification
        for the last point, and fresh is the prediction for the new point.
    """

    def __init__(self, data, trusted_data, threshold):
        self.data = data
        self.trusted_data = trusted_data
        self.threshold = threshold
        self.data_iter = iter(data)
        self.last = NORMAL

    def __iter__(self):
        return self

    def __next__(self):
        datapoint = next(self.data_iter)
        rev, fresh = classify(datapoint, self.trusted_data, self.last, self.threshold)
        self.last = fresh
        return (rev, fresh)

def test(data, labels, window_size, threshold):
    """test tests this algorithm against a dataset

    Args:
        data: A list containing the data to test.
        labels: A list containing labels for data (must be formatted as ANOMALY or NORMAL as defined in this module.
        window_size: The size of the trusted window to use.
        threshold: The size of the euclidean distance threshold to use.
    Returns:
        A tuple containing number of false positives, false negatives, and skipped anomalies
    """
    data_iter = iter(data)

    fp, fn = 0, 0

    # Extract trusted data from data.
    # Skips data with anomalous label.
    trusted_data = deque(maxlen=window_size)
    i, taken_data, skipped_anomalies = 0, 0, 0
    for value in data_iter:
        i = taken_data + skipped_anomalies
        if taken_data >= window_size:
            break
        if labels[i] == ANOMALY:
            skipped_anomalies += 1
        else:
            trusted_data.append(value)
            taken_data += 1

    for r, _ in classifier(data_iter, trusted_data, threshold):
        print("point classified as", "anomaly" if r == ANOMALY else "normal")
        if r == ANOMALY != labels[i - 1]:
            fp += 1
        elif r == NORMAL != labels[i - 1]:
            fn += 1
        i += 1

    return (fp, fn, skipped_anomalies)

if __name__ == "__main__":
    import os

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--infile', dest='infile', type=str, help='Path to the dataset to test')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int, help='The size of the subset of trusted data')
    parser.add_argument('-s', '--threshold', dest='threshold', type=float, help='The euclidean distance threshold')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Show detailed output while test is running')
    parser.add_argument('-c', '--csv', dest='csv', action='store_true', help='Format the program output into csv fomat: (filename,trusted size,threshold,false positives, false negatives,discarded anomalies')
    parser.add_argument('--twitter', dest='twitter_set',type=str, help='The dataset to use if using twitter (eg. AAPL, GOOG, etc)')
    parser.add_argument('--yahoo', dest='yahoo', action='store_true', help='Set this flag to parse the csv as a yahoo A1Benchmark')
    parser.add_argument('--SWaT', dest='SWaT', action='store_true', help='Set this flag to parse the csv as the SWaT data set')
    parser.set_defaults(twitter_set=False)
    parser.set_defaults(yahoo=False)
    parser.set_defaults(SWaT=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(csv=False)

    args = parser.parse_args()
    infile = args.infile
    trusted_size = args.trusted_size
    threshold = args.threshold
    verbose = args.verbose
    csv = args.csv

    data, labels = None, None
    if args.twitter_set:
        DATA_PATH = './nab/realTweets/realTweets/Twitter_volume_%s.csv'
        data = list(map(lambda x: np.array([x]),
                        np.loadtxt(open(DATA_PATH % (args.twitter_set), 'rb'), delimiter=",", skiprows=1, usecols=1, dtype=int)))

        with open("labels/combined_labels.json") as fh:
            timestamps = np.loadtxt(open(DATA_PATH % (args.twitter_set), 'rb'), delimiter=",", skiprows=1, usecols=0, dtype=str)
            labels_raw = fh.read()
            anomalies = json.loads(labels_raw)["realTweets/Twitter_volume_%s.csv" % (args.twitter_set)]
            labels = list(map(lambda x: ANOMALY if x in anomalies else NORMAL, timestamps))
    elif args.SWaT:
        data_raw = np.loadtxt(infile, delimiter=',', usecols=range(1,52), skiprows=2, dtype=float)
        rows, cols = data_raw.shape
        labels = list(map(lambda x: NORMAL if x == 1.0 else ANOMALY, data_raw[:, cols-1]))
        data = data_raw[:, :cols-1]

        # Scale the data as in Li Dan's implementation
        for i in range(cols - 1):
            data[:, i] /= max(data[:, i])
            data[:, i] = 2 * data[:, i] - 1

        # Run PCA algorithm to reduce to 5 dimensions
        from sklearn.decomposition import PCA
        num_signals = 5
        pca = PCA(n_components=num_signals,svd_solver='full')
        pca.fit(data)
        pc = pca.components_
        data = np.matmul(data, pc.transpose(1, 0))
    elif args.yahoo:
        data = list(map(lambda x: np.array([x]),
                   np.loadtxt(infile, delimiter=',', skiprows=1, usecols=1, dtype=float)))
        labels = list(map(lambda x: NORMAL if x == 0 else ANOMALY,
                          np.loadtxt(infile, delimiter=',', skiprows=1, usecols=2, dtype=int)))
    else:
        print("No format specified! Run with --help for more info.")
        os.exit(1)

    false_pos, false_neg, discarded_anomalies = test(data, labels, trusted_size, threshold)

    accu = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    pre = true_pos / (true_pos + false_pos)
    req = true_pos / (true_pos _ false_neg)
    f1 = 2 * ((pre * req) / (pre + req))
    fpr = false_pos / (false_pos + true_neg)

    if csv:
        print("%10s %10d %10f %10s %10s %10s" % (os.path.basename(infile), trusted_size, threshold, false_pos, false_neg, discarded_anomalies))
    else:
        print("Discarded Anomalies:", discarded_anomalies)
        print("False Positives:", false_pos)
        print("False Negatives:", false_neg)
        print("Accu:", accu)
        print("Pre:", pre)
        print("Req:", req)
        print("fq:", f1)
        print("FPR:", fpr)

