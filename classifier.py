from abc import ABC

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def statistics_universal(self, X, y, tau, sample_weight=None):
    pred = self.predict(X)
    return (
        accuracy_score(y, pred, sample_weight=sample_weight),
        confusion_matrix(y, pred, labels=None),
        np.count_nonzero(y == pred) * tau - np.count_nonzero(y != pred) * tau,
        np.count_nonzero(np.all([y == pred, pred == 1], axis=0)) * tau
        - np.count_nonzero(np.all([y != pred, pred == 1], axis=0)) * tau,
        np.count_nonzero(np.all([y == pred, pred == 0], axis=0)) * tau
        - np.count_nonzero(np.all([y != pred, pred == 0], axis=0)) * tau,
    )


class CustomMLPClassifier(MLPClassifier):
    def statistics(self, X, y, tau, sample_weight=None):
        return statistics_universal(self, X, y, tau, sample_weight=sample_weight)


class CustomGradientBoostingClassifier(GradientBoostingClassifier, ABC):
    def statistics(self, X, y, tau, sample_weight=None):
        return statistics_universal(self, X, y, tau, sample_weight=sample_weight)


class CustomCatBoostClassifier(CatBoostClassifier):
    def statistics(self, X, y, tau, sample_weight=None):
        return statistics_universal(self, X, y, tau, sample_weight=sample_weight)
