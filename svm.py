#!/usr/bin/env python3

import json
import numpy
import argparse
import functools
from sklearn import svm

Normal = 1
Anomaly = -1

argparser = argparse.ArgumentParser()

argparser.add_argument('--twitter',
        type=str, dest='twitter', default=None,
        help='The Twitter dataset to run the test on (AAPL, GOOG, etc).')

argparser.add_argument('-t', '--train', 
        type=int, dest='train_size', default=0,
        help='The size of the training data set')

args = argparser.parse_args()

train, data, labels = None, None, None

if args.twitter:
    datapath = './nab/realTweets/realTweets/Twitter_volume_{}.csv'.format(args.twitter)
    labelspath = './labels/combined_labels.json'

    datafile = open(datapath, 'rb')
    labelsfile = open(labelspath, 'rb')

    data = list(map(lambda sample: (sample,), numpy.loadtxt(datafile, delimiter=",", skiprows=1, usecols=1, dtype=int)))
    datafile.seek(0) 
    timestamps = numpy.loadtxt(datapath, delimiter=",", skiprows=1, usecols=0, dtype=str)
    anomalies = json.loads(labelsfile.read())['realTweets/Twitter_volume_{}.csv'.format(args.twitter)]
    labels = list(map(lambda timestamp: Anomaly if timestamp in anomalies else Normal, timestamps))

    datafile.close()
    labelsfile.close()

    train = data[:args.train_size]
    data = data[args.train_size:]

def analyze_results(stats, elem):
    (tp, tn, fp, fn) = stats
    (flag, label) = elem

    if flag == label == Anomaly:
        return (tp + 1, tn, fp, fn)
    elif flag == label == Normal:
        return (tp, tn + 1, fp, fn)
    elif flag == Anomaly:
        return (tp, tn, fp + 1, fn)
    else:
        return (tp, tn, fp, fn + 1)

classifier = svm.OneClassSVM()

if args.train_size > 0:
    classifier.fit(train)

results = zip(classifier.fit_predict(data), labels[args.train_size:])
stats = functools.reduce(analyze_results, results, (0,0,0,0))

print(",".join(map(str, stats)))

