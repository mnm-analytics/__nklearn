#!/usr/bin/python
# -*- Coding: utf-8 -*-

from node import Node

class DTree():
    
    def __init__(self, max_depth=2, min_sample_split=1):
        self.tree = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        
    def fit(self, X, y):
        tree = Node(X=X, y=y, max_depth=self.max_depth, min_sample_split=self.min_sample_split)
        tree.split_node(1)
        self.tree = tree

    def predict_proba(self, X):
        
        if self.tree == None:
            print("fitして下さい。")
            return
        
        pred = list()
        N = X.shape[0]
        
        for i in range(N):
            x = X.loc[i]
            _y = self.tree.get_label(x)
            pred.append(_y)
        return np.array(pred)

    def predict(self, X, threshold=0.5):
        pred = self.predict_proba(X) >= threshold
        return pred.astype(int)


if __name__ == "__main__":
    pass