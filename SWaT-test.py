#!/usr/bin/env python3

import umi
import numpy
import argparse
from sklearn.decomposition import PCA
from collections import deque

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample of the LCE method for detecting anomalies in SWaT dataset')
    parser.add_argument('-t', '--trusted-size', dest='trusted_size', type=int)
    parser.add_argument('-if', '--infile', dest='infile', type=str)
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

    all_data = numpy.loadtxt(infile, delimiter=',', usecols=range(1,52), skiprows=2, dtype=float)
    rows, cols = all_data.shape
    all_samples = all_data[:, :cols-1]
    all_labels = all_data[:, cols-1]
    print(all_samples.shape)

    # Scale the data.
    # Done to match conditions used in Li Dan's GAN-AD
    # implementation for SWaT so that we may compare results.
    for i in range(cols - 1):
        all_samples[:, i] /= max(all_samples[:, i])
        all_samples[:, i] = 2 * all_samples[:, i] - 1

    # Run PCA on dataset
    # All parameters here are made to match Li Dan's GAN-AD
    # implementation for SWaT, so that we may compare results.
    X = all_samples
    num_signals = 5
    pca = PCA(n_components=num_signals, svd_solver='full')
    pca.fit(X)
    pc = pca.components_
    all_samples = numpy.matmul(X, pc.transpose(1, 0))
    rows, cols = all_data.shape
    print(all_samples.shape)
    discarded_anomalies, i, j = 0, 0, trusted_size

    while i < j:
        if all_labels[i] != 1.0:
            all_labels[i] = -1
            discarded_anomalies += 1
            j += 1
        i += 1

    results = numpy.zeros(rows, dtype=int)
    results[:trusted_size] = numpy.full(trusted_size, umi.NORMAL, dtype=int)

    trusted_data = deque(maxlen=trusted_size + 2)
    for i in range(j):
        if all_labels[i] == 1.0:
            trusted_data.append(all_samples[i])
    anomalies = 0

    for i in range(j, rows):
        datapoint = all_samples[i]

        last_classification = results[i -  1]

        revision, fresh = umi.classify(datapoint, trusted_data, last_classification, threshold)

        results[i - 1] = revision
        results[i] = fresh

        if verbose:
            print("Point %d classified as %d" % (i - 1, revision))

        if revision == umi.ANOMALY:
            anomalies += 1
    
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(results)):
        if all_labels[i] == 1.0 and results[i] == umi.NORMAL:
            true_neg += 1
        elif all_labels[i] == 1.0 and results[i] == umi.ANOMALY:
            false_pos += 1
        elif all_labels[i] != 1.0 and results[i] == umi.NORMAL:
            false_neg += 1
        elif all_labels[i] != 1.0 and results[i] == umi.ANOMALY:
            true_pos += 1

    if csv:
        print("%10s %10d %10f %10s %10s %10s %10s %10s" % (os.path.basename(infile), trusted_size, threshold, true_pos, true_neg, false_pos, false_neg, discarded_anomalies))
    else:
        print("Discarded Anomalies:", discarded_anomalies)
        print("True Positives:", true_pos)
        print("True Negatives:", true_neg)
        print("False Positives:", false_pos)
        print("False Negatives:", false_neg)


