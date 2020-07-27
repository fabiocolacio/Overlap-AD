import statistics
import collections

Anomaly = -1
Normal = 1

class MovingAverageClassifier:
    def __init__(self, threshold=2):
        self.threshold = threshold

    def _update(self):
        self.mean = statistics.mean(self.data)
        self.stdev = statistics.stdev(self.data, xbar=self.mean)
        
    def fit(self, X):
        self.data = collections.deque(X, maxlen=len(X))
        self._update()
        return self

    def fit_predict(self, X):
        output = []
        
        for sample in X:
            self.data.append(sample)
            self._update()

            if self.stdev > 0:
                zscore = (sample - self.mean) / self.stdev
                if abs(zscore) > self.threshold:
                    output.append(Anomaly)
                else:
                    output.append(Normal)
            else:
                if sample == self.data[0]:
                    output.append(Normal)
                else:
                    output.append(Anomaly)
                    
        return output
        

class MovingMedianClassifier:
    def __init__(self):
        pass

    def fit(X):
        pass

    def fit_predict(X):
        pass
