class ProbabilityStore:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = []
        self.feature_values = {}

    def set_classes(self, y):
        import numpy as np
        self.classes = np.unique(y)

    def store_prior(self, cls, prior):
        self.class_priors[cls] = prior

    def store_likelihood(self, feature_index, value, cls, probability):
        pass

    def get_likelihood(self, feature_index, value, cls, default=1e-6):
        pass