import numpy as np

class Validator:
    def __init__(self, y_true, y_pred, positive_label=1, negative_label=-1):
        self.y_true = y_true
        self.y_pred = y_pred
        self.positive = positive_label
        self.negative = negative_label

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self._compute_matrix()

    def _compute_matrix(self):
        for true, pred in zip(self.y_true, self.y_pred):
            if true == self.positive and pred == self.positive:
                self.TP += 1
            elif true == self.negative and pred == self.negative:
                self.TN += 1
            elif true == self.negative and pred == self.positive:
                self.FP += 1
            elif true == self.positive and pred == self.negative:
                self.FN += 1

    def summary(self):
        return {
            "TP": self.TP,
            "TN": self.TN,
            "FP": self.FP,
            "FN": self.FN
        }

    def accuracy(self):
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total else 0

    def precision(self):
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) else 0

    def recall(self):
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) else 0

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

    def print_report(self):
        print("Confusion Matrix:")
        print(f"TP: {self.TP}, FP: {self.FP}")
        print(f"FN: {self.FN}, TN: {self.TN}")
        print("\nMetrics:")
        print(f"Accuracy:  {self.accuracy():.2f}")
        print(f"Precision: {self.precision():.2f}")
        print(f"Recall:    {self.recall():.2f}")
        print(f"F1 Score:  {self.f1_score():.2f}")
