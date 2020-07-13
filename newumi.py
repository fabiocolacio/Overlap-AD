#!/usr/bin/env python3

import collections
import sys
import json
import numpy

Normal = 1
Anomaly = -1
Pending = 0
Unclassified = 2

class Window:
    def __init__(self, data):
        self.lce = {}
        self.delta = 1
        self.quadrats = {}

        self.samples = collections.deque(data)
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
            
            tmpl = lce_func(tmpd)
            tmpd /= 2

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

    if any(map(lambda other: in_threshold(sample, other, threshold), window.samples)):
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
        revision = Normal if new_class == Normal \
            else Anomaly

    if new_class == Normal:
        window.min_lce_popleft()

    if last_class == Pending and revision == Normal:
        window.min_lce_popleft()

    if revision == Anomaly:
        window.min_lce_popleft()

    return revision, new_class

def main():
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--win-size', type=int, dest='win_size', default=250,
                           help='The window size to use (defaults to 250).')
    
    argparser.add_argument('--thresh', type=float, dest='thresh', default=0.0,
                           help='The euclidean distance threshold to use (defaults to 0).')
    
    argparser.add_argument('--twitter', type=str, dest='twitter', default=None,
                           help='The twitter dataset to run the test on (AAPL, GOOG, etc)')

    args = argparser.parse_args()

    data = None
    labels = None
    
    if args.twitter != None:
        datapath = './nab/realTweets/realTweets/Twitter_volume_{}.csv'.format(args.twitter)
        labelspath = './labels/combined_labels.json'
        labelskey = 'realTweets/Twitter_volume_{}.csv'.format(args.twitter)

        datafile = open(datapath, 'rb')
        labelsfile = open(labelspath, 'rb')

        data = list(map(lambda x: (x / 470,),
                    numpy.loadtxt(datafile, delimiter=',', skiprows=1, usecols=1, dtype=int)))
        datafile.seek(0)
        timestamps = numpy.loadtxt(datapath, delimiter=',', skiprows=1, usecols=0, dtype=str)
        anomalies = json.loads(labelsfile.read())[labelskey]
        labels = list(map(lambda timestamp: Anomaly if timestamp in anomalies else Normal, timestamps))
        
        datafile.close()
        labelsfile.close()
    else:
        println("Please specify a dataset to run.")
        sys.exit(1)
    
    window = Window(data[:args.win_size])
    window.min_lce()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    last_class = Normal

    for i in range(args.win_size, len(data)):
        sample = data[i]
        last_label = labels[i - 1]
        
        revision, last_class = classify(sample, window, last_class, args.thresh)

        if revision == Anomaly == last_label:
            tp += 1
        elif revision == Normal == last_label:
            tn += 1
        elif revision == Anomaly:
            fp += 1
        else:
            fn += 1

    print("{},{},{},{}".format(tp,tn,fp,fn))
        

if __name__ == '__main__':
    main()
