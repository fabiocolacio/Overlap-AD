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
    parser.add_argument('-v', '--verbose', dest='verbose', type=float)
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

    

