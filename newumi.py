#!/usr/bin/env python3

# Copyright (C) 2020 Fabio Colacio
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import collections
import sys
import json
import numpy
import scipy.io

Normal = 1
Anomaly = -1
Pending = 0
Unclassified = 2

class Window:
    def __init__(self, data):
        self.lce = {}
        self.delta = 1
        self.quadrats = {}

        self.samples = collections.deque()
        self.sample_set = set()
        self.instances = collections.defaultdict(int)
        
        for sample in data:
            self.samples.append(sample)
            self.sample_set = self.sample_set | { sample }
            self.instances[sample] += 1

    def _sample_decrement(self, sample):
        if self.instances[sample] == 1:
            del self.instances[sample]
            self.sample_set -= { sample }
        else:
            self.instances[sample] -= 1    
            
    def quantize_feature(feature, delta):
        return 0 if feature == 0 else \
            int(numpy.ceil(feature / delta)) - 1

    def quantize_sample(sample, delta):
        return tuple(map(lambda f: Window.quantize_feature(f, delta), sample))

    # Finds n! / (n - m)! where:
    # n is the number of samples in a quadrat
    # m is the minimum number of samples.
    # For anomaly detection, we are using m=2 always
    def morisita(n):
        return n * (n - 1)
            
    def lce_full(self, delta):
        self.quadrats[delta] = collections.defaultdict(int)
            
        for sample in self.sample_set:
            quadrat = Window.quantize_sample(sample, delta)
            self.quadrats[delta][quadrat] += 1
                    
        quadrat_sums = self.quadrats[delta].values()
        lce_val = sum(map(Window.morisita, quadrat_sums))
        self.lce[delta] = lce_val
        return lce_val

    # Incrementally update the LCE for the window as if samples had been added or removed.
    # The incremental update takes O(dimensions) time for each sample provided.
    def lce_incremental(self, sample, delta, add=True):
        if delta not in self.quadrats or delta not in self.lce:
            self.lce_full(delta)

        newlce = None

        quadrat = Window.quantize_sample(sample, delta)

        oldlce = self.lce[delta]
        newlce = oldlce

        if quadrat in self.quadrats[delta]:
            oldsum = self.quadrats[delta][quadrat]
            oldmi = Window.morisita(oldsum)

            newsum = oldsum + 1 if add else oldsum - 1
            newmi = Window.morisita(oldsum)

            newlce = oldlce - oldmi + newmi

            self.quadrats[delta][quadrat] = newsum
            self.lce[delta] = newlce
        else:
            if add:
                self.quadrats[delta][quadrat] = 1
            else:
                raise ValueError("The sample to remove did not exist in known quadrats.")

        return newlce

    # Incrementally update LCE as if sample had been added.
    # Must call lce_full first.
    # Does not actually add samples to the window.
    def lce_incremental_add(self, sample, delta):
        return self.lce_incremental(sample, delta, True)

    # See lce_incremental_add()
    def lce_incremental_sub(self, sample, delta):
        return self.lce_incremental(sample, delta, False)
    
    def _min_lce(self, lce_func):
        d, tmpd = 1, 1
        l, tmpl = 1, 1

        while tmpl > 0:
            l = tmpl
            d = tmpd

            tmpd /= 2
            tmpl = lce_func(tmpd)

        if d > self.delta:
            tmp = d / 2
            while tmp >= self.delta:
                del self.quadrats[tmp]
                tmp /= 2
        
        self.delta = d
        
        return l, d

    def min_lce_full(self):
        return self._min_lce(self.lce_full)
    
    def min_lce_append(self, sample):
        self.samples.append(sample)
        return self._min_lce(lambda delta: self.lce_incremental_add(sample, delta))

    def min_lce_popleft(self):
        sample = self.samples.popleft()
        self._sample_decrement(sample)
        return self._min_lce(lambda delta: self.lce_incremental_sub(sample, delta))

    def min_lce_pop(self):
        sample = self.samples.pop()
        self._sample_decrement(sample)
        return self._min_lce(lambda delta: self.lce_incremental_sub(sample, delta))
    
    # If the LCE was previously computed, returns the cached value
    # Otherwise min_lce will compute the full LCE in O(ndlog(n))
    def min_lce(self):
        if self.delta != None and self.delta in self.lce:
            return self.lce[self.delta], self.delta
        return self.min_lce_full()
    
def euclidean_dist(a, b):
    x, y = numpy.array(a), numpy.array(b)
    return numpy.linalg.norm(x - y)

def in_threshold(a, b, threshold):
    return euclidean_dist(a,b) <= threshold

def classify(sample, window, last_class, threshold):
    revision = last_class
    new_class = Unclassified

    if sample in window.sample_set or (threshold > 0 and any(map(lambda other: in_threshold(sample, other, threshold), window.samples))):
        window.min_lce_append(sample)
        new_class = Normal
    else:
        trusted_lce, trusted_delta = window.min_lce()
        total_lce, total_delta = window.min_lce_append(sample)

        clustered = trusted_lce < total_lce or \
            total_delta < trusted_delta

        if clustered:
            new_class = Normal
        else:
            new_class = Pending

    if last_class == Pending:
        revision = Normal if new_class == Normal else Anomaly

        if revision == Normal:
            window.min_lce_popleft()

        if revision == Anomaly:
            window.min_lce_pop()
            window.min_lce_pop()
            window.min_lce_append(sample)

    if new_class == Normal:
        window.min_lce_popleft()
    
    return revision, new_class


class MorisitaClassifier:
    """Detect anomalies with the streaming Morisita Index method.

    The MorisitaClassifier class provides a scikit-like API for anomaly detection with fit(), and fit_predict().
    """
    
    def __init__(self, threshold=0.0):
        self.window = None
        self.threshold = threshold
        self.last_class = Normal

    def fit(self, samples):
        """Updates the window and min_lce with the samples provided.
        
        samples must be a list or other iterable containing samples of homogenous dimensionality.
        """
        
        self.window = Window(samples)
        self.window.min_lce()
        return self

    def fit_predict_stream(self, samples):
        """Predicts labels for the samples

        Returns a generator providing tuples containing (revision, new_class) pairs
        where revision is the revised classification of the last sample, and new_class
        is the classification of the new sample.
        
        This function is suitable for streaming use cases where samples is not a finite
        list and the returned generator can be used in a loop.
        """
        
        for sample in samples:
            revision, new_class = classify(sample, self.window, self.last_class, self.threshold)
            self.last_class = new_class
            yield revision, new_class

    def fit_predict(self, samples):
        """Like fit_predict_stream() but returns a finite list.

        This is sugar of list(fit_predict_stream(samples)) and is provided
        for compatibility with the scikit API for anomaly detection.
        """
        return list(self.fit_predict_stream(samples))

def main():
    import argparse
    import time
    import functools

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--win-size', type=int, dest='win_size', default=250,
                           help='The window size to use (defaults to 250).')
    
    argparser.add_argument('--thresh', type=float, dest='thresh', default=0.0,
                           help='The euclidean distance threshold to use (defaults to 0).')
    
    argparser.add_argument('--twitter', type=str, dest='twitter', default=None,
                           help='The twitter dataset to run the test on (AAPL, GOOG, etc)')

    argparser.add_argument('-if', '--infile', dest='infile', type=str,
                           help='Path to the dataset to test.')
    
    argparser.add_argument('--yahoo', dest='yahoo', default=False, action='store_true',
                           help='Parse the file as a yahoo benchmark.')

    argparser.add_argument('--kdd', dest='kdd', default=False, action='store_true',
                           help='Parse the file as a kdd smtp or http file.')
    
    args = argparser.parse_args()

    data = None
    labels = None
    
    if args.twitter != None:
        datapath = './nab/realTweets/realTweets/Twitter_volume_{}.csv'.format(args.twitter)
        labelspath = './labels/combined_labels.json'
        labelskey = 'realTweets/Twitter_volume_{}.csv'.format(args.twitter)

        datafile = open(datapath, 'rb')
        labelsfile = open(labelspath, 'rb')

        data = numpy.loadtxt(datafile, delimiter=',', skiprows=1, usecols=1, dtype=int)
        gmax = max(data)
        data = list(map(lambda x: (x / gmax,), data))
        datafile.seek(0)
        
        timestamps = numpy.loadtxt(datapath, delimiter=',', skiprows=1, usecols=0, dtype=str)
        anomalies = json.loads(labelsfile.read())[labelskey]
        labels = list(map(lambda timestamp: Anomaly if timestamp in anomalies else Normal, timestamps))
                
        datafile.close()
        labelsfile.close()
    elif args.yahoo:
        data = numpy.loadtxt(args.infile, delimiter=',', skiprows=1, usecols=1, dtype=float)
        gmax = max(data)
        data = list(map(lambda x: (x / gmax,), data))
        
        labels = list(map(lambda x: Normal if x == 0 else Anomaly,
                          numpy.loadtxt(args.infile, delimiter=',', skiprows=1, usecols=2, dtype=int)))
    elif args.kdd:
        import h5py

        handle = h5py.File(args.infile, 'r')
        data = handle['X']
        data = list(zip(data[0], data[1], data[2]))
        gmin = functools.reduce(lambda v, s: (min(v[0],s[0]), min(v[1],s[1]), min(v[2],s[2])), data)
        data = list(map(lambda s: (s[0] - gmin[0], s[1] - gmin[1], s[2] - gmin[2]), data))
        gmax = functools.reduce(lambda v, s: (max(v[0],s[0]), max(v[1],s[1]), max(v[2],s[2])), data)
        data =  list(map(lambda s: (s[0] / gmax[0], s[1] / gmax[1], s[2] / gmax[2]), data))
        labels = list(map(lambda x: Normal if x == 0.0 else Anomaly, handle['y'][0]))
    else:
        print("Please specify a dataset to run.")
        sys.exit(1)

    idx = 0
    window_samples = []
    while len(window_samples) < args.win_size:
        if labels[idx] == Normal:
            window_samples.append(data[idx])
        idx += 1
    data = data[idx:]
    labels = labels[idx:]

    start_time = time.process_time_ns()
    
    classifier = MorisitaClassifier(args.thresh).fit(window_samples)
    predictions = classifier.fit_predict(data)
    
    end_time = time.process_time_ns()
    elapsed = end_time - start_time

    tp, tn, fp, fn = 0, 0, 0, 0
    last_label = Normal
    for i in range(len(predictions)):
        revision, new_class = predictions[i]
        
        this_label = labels[i]
        last_label = Normal if i == 0 else labels[i - 1]
        
        if revision == last_label == Anomaly:
            tp += 1
        elif revision == last_label == Normal:
            tn += 1
        elif revision == Anomaly:
            fp += 1
        elif revision == Normal:
            fn += 1

        if i == len(labels) - 1:
            if new_class == Pending and this_label == Anomaly:
                tp += 1
            elif new_class == Normal and this_label == Normal:
                tn += 1
            elif new_class == Pending:
                fp += 1
            else:
                fn += 1
                
        i += 1

    dset = args.twitter if args.twitter != None else args.infile
    print("{},{},{},{},{},{},{},{:.2f},{}".format("mi",dset,args.win_size,tp,tn,fp,fn,elapsed / 1000000000,args.thresh))

if __name__ == '__main__':
    main()
