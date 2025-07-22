import numpy as np

class NaiveBayesPredictor:
    def __init__(self, store):
        self.store = store

    def predict_single(self, x):
        posteriors = {}
        for c in self.store.classes:
            log_prob = np.log(self.store.class_priors[c])
            for i, val in enumerate(x):
                prob = self.store.get_likelihood(i, val, c)
                log_prob += np.log(prob)
            posteriors[c] = log_prob
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return [self.predict_single(x) for x in X]
