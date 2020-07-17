#!/usr/bin/env python3

import json
import numpy
import argparse
import functools

Normal = 1
Anomaly = -1

argparser = argparse.ArgumentParser()

argparser.add_argument('--twitter',
        type=str, dest='twitter', default=None,
        help='The Twitter dataset to run the test on (AAPL, GOOG, etc).')

argparser.add_argument('-t', '--train', 
        type=int, dest='train_size', default=0,
        help='The size of the training data set')

argparser.add_argument('-a', '--algorithm',
                       type=str, dest='algorithm',
                       help='The algorithm to use for classifications')

argparser.add_argument('--threshold',
                       type=float, dest='threshold', default=100.0,
                       help="The threshold to use for knn algorithm.")

argparser.add_argument('-k',
                       type=int, dest='k', default=4,
                       help="The hyperparameter K for the KNN algorithm.")

args = argparser.parse_args()

train, data, labels = None, None, None

if args.twitter:
    datapath = './nab/realTweets/realTweets/Twitter_volume_{}.csv'.format(args.twitter)
    labelspath = './labels/combined_labels.json'

    datafile = open(datapath, 'rb')
    labelsfile = open(labelspath, 'rb')

    data = numpy.array(list(map(lambda sample: [sample],
                                numpy.loadtxt(datafile, delimiter=",", skiprows=1, usecols=1, dtype=int))))
    datafile.seek(0) 
    timestamps = numpy.loadtxt(datapath, delimiter=",", skiprows=1, usecols=0, dtype=str)
    anomalies = json.loads(labelsfile.read())['realTweets/Twitter_volume_{}.csv'.format(args.twitter)]
    labels = list(map(lambda timestamp: Anomaly if timestamp in anomalies else Normal, timestamps))

    datafile.close()
    labelsfile.close()

    train = data[:args.train_size]
    train_labels = labels[:args.train_size]
    data = data[args.train_size:]
    data_labels = labels[args.train_size:]

predict = None

def euclidean_distance(a,b):
    return numpy.lihnalg.norm(a - b)

if args.algorithm == "knn":
    from sklearn.neighbors import KNeighborsClassifier
    from statistics import mean

    classifier = KNeighborsClassifier(n_neighbors=args.k).fit(train, train_labels)

    def predictor(samples):
        neighbor_dists = classifier.kneighbors(samples)[0]
        mean_dists = map(mean, neighbor_dists)
        return map(lambda mean: Normal if mean < args.threshold else Anomaly, mean_dists)

    predict = predictor
elif args.algorithm == "svm":
    from sklearn.svm import OneClassSVM

    classifier = OneClassSVM().fit(train)
    predict = classifier.fit_predict
elif args.algorithm == "naive-bayes":
    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB().fit(train, train_labels)
    predict = classifier.predict
elif args.algorithm == "decision-tree":
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier().fit(train, train_labels)
    predict = classifier.predict
else:
    print("No valid algorithms specified.")
    sys.exit(1)

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

results = zip(predict(data), labels[args.train_size:])
stats = functools.reduce(analyze_results, results, (0,0,0,0))

print("{},{},".format(args.twitter, args.train_size), end="")

if args.algorithm == 'knn':
    print("{},{},".format(args.k, args.threshold), end="")

print(",".join(map(str, stats)))
