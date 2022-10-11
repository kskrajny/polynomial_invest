from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class CustomMLPClassifier(MLPClassifier):
    def statistics(self, X, y, sample_weight=None):
        return (
            accuracy_score(y, self.predict(X), sample_weight=sample_weight),
            confusion_matrix(y, self.predict(X), labels=None)
        )
