from Models.Probability_Store import ProbabilityStore
from Models.Trainer import NaiveBayesTrainer
from Models.Predictor import NaiveBayesPredictor

class NaiveBayesClassifier:
    def __init__(self):
        self.store = ProbabilityStore()
        self.trainer = NaiveBayesTrainer(self.store)
        self.predictor = NaiveBayesPredictor(self.store)

    def fit(self, X, y):
        self.trainer.fit(X, y)

    def predict(self, X):
        return self.predictor.predict(X)

    def predict_single(self, x):
        return self.predictor.predict_single(x)