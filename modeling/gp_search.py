#!/usr/bin/python
# -*- Coding: utf-8 -*-

import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

class ClfOpt:

    def __init__(self, n_splits=5, random_state=1234):
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def get_data(self, X, y):
        tra_Xs, val_Xs, tra_ys, val_ys = [], [], [], []
        for tra_idx, val_idx in self.kf.split(X,y):
            tra_X, tra_y, val_X, val_y = X.iloc[tra_idx], y.iloc[tra_idx], X.iloc[val_idx], y.iloc[val_idx]
            tra_Xs.append(tra_X)
            tra_ys.append(tra_y)
            val_Xs.append(val_X)
            val_ys.append(val_y)
        return tra_Xs, tra_ys, val_Xs, val_ys

    def get_opt_svc(self, X, y, n_trials=100):
        def objective(trial):
            param = {
                'C':trial.suggest_loguniform('C', 2**-5, 2**15),
                'gamma':trial.suggest_loguniform('gamma', 2**-15, 2**3),
                'kernel':"rbf"
            }
            clf = SVC(**param)
            pred = clf.predict(val_X)
            scores = cross_validate(clf, scoring="accuracy", X=X, y=y, cv=self.kf)
            return 1.0 - scores['test_score'].mean()

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        model = SVC(**study.best_params)
        model.fit(X, y)
        return model

    def get_opt_rf(self, X, y, n_trials=100):
        def objective(trial):
            params = {
                'n_estimators': int(trial.suggest_loguniform('n_estimators', 1e+2, 1e+3)),
                'max_depth': int(trial.suggest_loguniform('max_depth', 2, 32)),
                'min_samples_split': trial.suggest_int("min_samples_split", 8, 16)
            }
            clf = RandomForestClassifier(**params)
            scores = cross_validate(clf, scoring="roc_auc", X=X, y=y, cv=self.kf)
            return 1.0 - scores['test_score'].mean()

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        model = RandomForestClassifier(**study.best_params)
        model.fit(X, y)
        return model

    def get_opt_logi(self, X, y, n_trials=100):
        def objective(trial):
            params = {
                'C': trial.suggest_loguniform('C', 1, 50),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
            clf = LogisticRegression(**params)
            scores = cross_validate(clf, scoring="roc_auc", X=X, y=y, cv=self.kf)
            return 1.0 - scores['test_score'].mean()

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        model = LogisticRegression(**study.best_params)
        model.fit(X, y)
        return model


if __name__ == "__main__":
    pass