#!/usr/bin/python
# -*- Coding: utf-8 -*-

import numpy as np
from sklearn.metrics import roc_curve

class CutOff:

    def __init__(self, train_proba, y):
        self.proba = train_proba
        self.y = y

    def get_threshold_vec(self, proba, y):
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=proba)
        c2p = (tpr - 1)**2 + fpr**2
        return thresholds[c2p == min(c2p)][0]

    def get_threshold_ydx(self, proba, y):
        # Youden index法
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=proba)
        ydx = tpr - (fpr-1)
        return thresholds[ydx == max(ydx)][0]

    def get_binary_vec(self, test_proba):
        thres = self.get_threshold_vec(self.proba, self.y)
        predict = test_proba >= thres
        return np.vectorize(int)(predict)

    def get_binary_ydx(self, test_proba):
        thres = self.get_threshold_ydx(self.proba, self.y)
        predict = test_proba >= thres
        return np.vectorize(int)(predict)

if __name__ == "__main__":
    pass