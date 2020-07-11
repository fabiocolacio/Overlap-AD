#!/usr/bin/env python3

import numpy
import collections
from functools import reduce
import sys

Normal = 1
Anomaly = -1
Pending = 0
Unclassified = 2

class Window:
    def __init__(self, size):
        self.lce = {}
        self.delta = 1
        self.quadrats = {}
        self.samples = collections.deque()

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
        if delta in self.quadrats:
            del self.quadrats[delta]
            
        for sample in self.data:
            quadrat = Window.quantize_sample(sample, delta)

            if quadrat in self.quadrats[delta]:
                self.quadrats[delta][quadrat] += 1
            else:
                self.quadrats[delta][quadrat] = 1
        
        quadrat_sums = self.quadrats[delta].values()
        ret =  sum(map(Window.morisita, quadrat_sums))
        self.lce[delta] = ret
        
        return ret

    # Incrementally update the LCE for the window as if samples had been added or removed.
    # The incremental update takes O(dimensions) time for each sample provided.
    def lce_incremental(self, samples, delta, add=True):
        if delta not in self.quadrats or delta not in self.lce:
            raise Exception("Must call lce_full before lce_incremental can be called.")

        if len(samples) < 1:
            raise ValueError("Must include at least one sample.")

        newlce = None
        
        for sample in samples:
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

    # Incrementally update LCE as if samples had been added.
    # Must call lce_full first.
    # Does not actually add samples to the window.
    def lce_incremental_add(self, samples, delta):
        return self.lce_incremental(samples, delta, True)

    # See lce_incremental_add()
    def lce_incremental_sub(self, samples, delta):
        return self.lce_incremental(samples, delta, False)
    
    def min_lce(self, lce_func):
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
        return self.min_lce(self.lce_full)

    def min_lce_incremental_add(self, samples):
        return self.min_lce(lambda delta: self.lce_incremental_add(samples, delta))

    def min_lce_incremental_sub(self, samples):
        return self.min_lce(lambda delta: self.lce_incremental_sub(samples, delta))

def euclidean_dist(a, b):
    return numpy.linalg.norm(a - b)

def in_threshold(a, b, threshold):
    return euclidean_dist(a,b) <= threshold

def classify(sample, window, last_class, threshold):
    revision = last_class
    new_class = Unclassified

    if any(map(lambda other: in_threshold(sample, other, threshold), window)):
        new_class = Normal
    else:
        trusted_lce, trusted_delta = min_lce()
        total_lce, total_delta = min_lce()

        clustered = trusted_lce < total_lce or \
            total_delta < trusted_delta

        if clustered:
            new_class = Noraml
        else:
            new_class = Pending

    if last_class == Pending:
        revision = Normal if new_class = Normal \
            else Anomaly

    if new_class == Normal:
        window.popleft()

    if last_class == Pending and revision == Normal:
        window.popleft()

    if revision == Anomaly:
        window.pop()

    window.append(sample)
    
    return revision, new_class

def main():
    data = []
    labels = []
    features = len(data[0])
    
    window = data[:window_size]
    stream = iter(data[window_size:])

if __name__ == __main__:
    main()
