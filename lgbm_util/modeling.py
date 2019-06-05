#!/usr/bin/python
# -*- Coding: utf-8 -*-

import uuid
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

class MyLGB:

    def __init__(self, model, X, best_score):
        self.model = model
        self.valid = valid
        self.best_score = best_score

    def predict(self):
        return model.predict(X)

class CvEnsembleModeling:

    def __init__(self, param=None, metric='binary_logloss', n_splits=5, random_state=1234):
        if param == None:
            self.param = {
                'boosting_type': ['goss'],
                'objective': ['binary'],
                'metric': [metric],    
                'max_depth':[-1]
            }
        else:
            self.param = param
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.models = None

    def get_lgbd(self, X, y):
        train, valid = [], []
        for train_idx, valid_idx in self.kf.split(X,y):
            train_X, train_y, valid_X, valid_y = X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx]
            train.append(lgb.Dataset(train_X, train_y))
            valid.append(lgb.Dataset(valid_X, valid_y))
        return train, valid

    def get_opt_lgbd(self, X, y, n_trials=100, ):
        def objective(trial):
            params = {
                'boosting_type': 'goss', 'objective': 'binary', 'metric': 'auc',
                'num_leaves': trial.suggest_int("num_leaves", 10, 500),
                'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 0, 400),
                'feature_fraction': trial.suggest_uniform("feature_fraction", 0.01, 1.0),
                'bagging_fraction': trial.suggest_uniform("bagging_fraction", 0.01, 1.0),
                'verbose' : 0
            }
            clf = lgb.train(params, lgb.Dataset(tra_X, tra_y), num_boost_round=500, early_stopping_rounds=early_stopping_rounds,
                            valid_sets=lgb.Dataset(val_X, val_y),
                            verbose_eval=verbose_eval
                            )
            score = roc_auc_score(y_true=val_y, y_score=clf.predict(val_X))
            return 1.0 - score

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        params = study.best_params
        params['boosting_type'] = 'goss',
        params['objective'] = 'binary',
        params['metric'] = 'auc',

        model = lgb.train(
            params,
            train, valid_sets=lgb.Dataset(val_X, val_y),
            num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
            verbose_eval = verbose_eval
        )
        return model

    def get_kf(self, X, y):
        tX, ty, vX, vy = [], [], [], []
        for train_idx, valid_idx in self.kf.split(X,y):
            train_X, train_y, valid_X, valid_y = X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx]
            tX.append(train_X)
            ty.append(train_y)
            vX.append(valid_X)
            vy.append(valid_y)
        return tX, ty, vX, vy

    def fit(self, X, y, num_boost_round=1000,early_stopping_rounds=5, verbose_eval=False):
        train, valid = self.get_lgbd(X, y)
        self.models = []
        for t, v in zip(train, valid):
            model = lgb.train(
                self.param,
                t, valid_sets=v,
                num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                verbose_eval = verbose_eval
            )
            self.models.append(model)
        return self.models

    def opt_fit(self, X, y, n_trials=100, num_boost_round=1000,early_stopping_rounds=5, verbose_eval=False):
        tX, ty, vX, vy = self.get_kf(X, y)
        self.models, self.best_scores = [], []
        i = 0
        for tra_X, tra_y, val_X, val_y in zip(tX, ty, vX, vy):
            i += 1
            print("MODEL%s========"%i)
            train = lgb.Dataset(tra_X, tra_y)
            valid = lgb.Dataset(val_X, val_y)
            def objective(trial):
                train = lgb.Dataset(tra_X, tra_y)
                # 試行にUUIDを設定
                # trial_uuid = str(uuid.uuid4())
                # trial.set_user_attr("uuid", trial_uuid)
                params = {
                    'boosting_type': 'goss',
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': trial.suggest_int("num_leaves", 10, 500),
                    'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
                    'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 0, 400),
                    'feature_fraction': trial.suggest_uniform("feature_fraction", 0.01, 1.0),
                    'bagging_fraction': trial.suggest_uniform("bagging_fraction", 0.01, 1.0),
                    # 'top_rate':trial.suggest_uniform('top_rate', 0.0, 1.0),
                    # 'other_rate':trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate']),
                    'verbose' : 0
                }
                # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
                clf = lgb.train(params, train, num_boost_round=500, early_stopping_rounds=early_stopping_rounds,
                                train_set=train, valid_sets=valid,
                                # callbacks=[pruning_callback],
                                verbose_eval=verbose_eval
                                )
                score = roc_auc_score(y_true=val_y, y_score=clf.predict(val_X))
                return 1.0 - score

            study = optuna.create_study()
            study.optimize(objective, n_trials=n_trials)
            params = study.best_params
            params['boosting_type'] = 'goss',
            params['objective'] = 'binary',
            params['metric'] = 'auc',

            model = lgb.train(
                params,
                train_set=train, valid_sets=valid,
                num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                verbose_eval = verbose_eval
            )
            MyLGB(model, val_X, study.best_value)
            self.models.append(model)
            self.best_scores.append(study.best_value)
        return self.models, self.best_scores

    def predict(self, X):
        preds = []
        if self.models == None: return None
        X = np.array(X)
        for model in self.models:
            preds.append(model.predict(X))
        preds = np.array(preds)
        return preds.mean(0)

    def valid(self, X, y, metric):
        pred = self.predict(X)
        return metric(y_true=y, y_score=pred)

if __name__ == "__main__":
    pass