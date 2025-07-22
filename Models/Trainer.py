import numpy as np

class NaiveBayesTrainer:
    def __init__(self, store):
        self.store = store

    def fit(self, X, y):
        self.store.set_classes(y)
        total_samples = len(y)

        for c in self.store.classes:
            X_c = X[y == c]
            self.store.store_prior(c, len(X_c) / total_samples)

            for i in range(X.shape[1]):
                values, counts = np.unique(X_c[:, i], return_counts=True)

                if i not in self.store.feature_values:
                    self.store.feature_values[i] = np.unique(X[:, i])

                total_count = sum(counts)
                k = len(self.store.feature_values[i])
                value_count_dict = dict(zip(values, counts))

                for val in self.store.feature_values[i]:
                    count = value_count_dict.get(val, 0)
                    prob = (count + 1) / (total_count + k)
                    self.store.store_likelihood(i, val, c, prob)
