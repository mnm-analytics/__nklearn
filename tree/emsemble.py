#!/usr/bin/python
# -*- Coding: utf-8 -*-

import random

class RandomForest:
    
    def __init__(self, n_estimators=5, feature_fraction=1.0, max_depth=2, min_sample_split=1):
        self.forest = None
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

    def fit(self, X, y):
        self.forest = list()
        cols = list(X.columns)
        n_features = int(X.shape[1] * self.feature_fraction)
        for i in range(self.n_estimators):
            _cols = random.sample(cols, n_features)
            _X = X[_cols]
            tree = DTree(self.max_depth, self.min_sample_split)
            tree.fit(_X, y)
            self.forest.append(tree)
        
    def predict_proba(self, X):
        if self.forest == None:
            print("fitして下さい。")
            return
        preds = list()
        for tree in self.forest:
            pred = tree.predict_proba(X)
            preds.append(pred)
        preds = np.array(preds)
        return preds.prod(0)**(1/len(self.forest))

    def predict(self, X, threshold=0.5):
        pred = self.predict_proba(X) >= threshold
        return pred.astype(int)


if __name__ == "__main__":
    pass