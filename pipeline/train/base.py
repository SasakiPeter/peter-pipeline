from copy import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from catboost import Pool


class Trainer:
    '''
    # Usage
    model = Trainer(CatBoostClassifier(**CAT_PARAMS))
    model.train(x_train, y_train, x_valid, y_valid, fit_params={})
    '''

    MODELS = {
        'CatBoostRegressor', 'CatBoostClassifier',
        'LGBMRegressor', 'LGBMClassifier',
        'RandomForestRegressor', 'RandomForestClassifier',
        'LinearRegression', 'LogisticRegression',
        'Ridge', 'Lasso',
        'SVR', 'SVC',
    }

    def __init__(self, model):
        model_type = type(model).__name__
        assert model_type in self.MODELS
        self.model = model
        self.model_type = model_type

    def train(self, X, y, X_valid=None, y_valid=None,
              cat_features=None, eval_metrics=None, fit_params={}):

        if self.model_type[:8] == 'CatBoost':
            train_data = Pool(data=X, label=y, cat_features=cat_features)
            valid_data = Pool(data=X_valid, label=y_valid,
                              cat_features=cat_features)
            self.model.fit(X=train_data, eval_set=valid_data, **fit_params)
            self.best_iteration = self.model.get_best_iteration()

        elif self.model_type[:4] == 'LGBM':
            if cat_features is None:
                cat_features = []
            self.model.fit(X, y, eval_set=[(X, y), (X_valid, y_valid)],
                           #    categorical_feature=cat_features,
                           **fit_params)
            self.best_iteration = self.model.best_iteration_
        else:
            # カテゴリ変数あるやつを、LinearRegressionにかけたら、バグりそう
            self.model.fit(X, y, **fit_params)
            self.best_iteration = -1

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_feature_importances(self):
        if self.model_type in [
            'CatBoostRegressor', 'CatBoostClassifier',
            # 'LGBMRegressor', 'LGBMClassifier',
            'RandomForestRegressor', 'RandomForestClassifier'
        ]:
            return self.model.feature_importances_
        elif self.model_type in ['LGBMRegressor', 'LGBMClassifier']:
            return self.model.booster_.feature_importance(
                importance_type='gain')
        elif self.model_type in ['LinearRegression', 'LogisticRegression',
                                 'Ridge', 'Lasso']:
            return self.model.coef_
        elif self.model_type in ['SVR', 'SVC']:
            if self.model.get_params()['kernel'] == 'linear':
                return self.model.coef_
            else:
                return 0
        else:
            return 0

    def get_params(self):
        if self.model_type[:8] == 'CatBoost':
            # return のほうがよくない？
            print(self.model.get_params())
        else:
            print('{}')

    def predict(self, X):
        return self.model.predict(X)

    def binary_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importances(self, columns):
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps = self.get_feature_importances()
        order = np.argsort(imps)
        plt.barh(np.array(columns)[order], imps[order])
        plt.show()


class CrossValidator:
    '''
    # Usage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cat_cv = CrossValidator(CatBoostClassifier(**CAT_PARAMS), skf)
    cat_cv.run(
        X, y, x_test,
        eval_metrics=[roc_auc_score], prediction='predict',
        train_params={'cat_features': CAT_IDXS, 'fit_params': CAT_FIT_PARAMS},
        verbose=0
    )
    '''

    def __init__(self, model=None, datasplit=None):
        self.basemodel = copy(model)
        self.datasplit = datasplit
        self.models = []
        self.oof = None
        self.pred = None
        self.imps = None
        self.id_test = None
        self.scores = None

    def run(self, X, y, X_test=None, id_test=None,
            group=None, n_splits=None,
            eval_metrics=None, prediction='predict',
            transform=None, train_params={}, verbose=True):

        assert isinstance(X, pd.core.frame.DataFrame)

        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits
        self.oof = np.zeros(len(X), dtype=np.float)
        if X_test is not None:
            self.pred = np.zeros(len(X_test), dtype=np.float)

        self.imps = np.zeros((X.shape[1], K))
        self.scores = pd.DataFrame()

        self.id_test = id_test

        for fold_i, (train_idx, valid_idx) in enumerate(
                self.datasplit.split(X, y, group)):

            x_train, x_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y[train_idx], y[valid_idx]

            if X_test is not None:
                x_test = X_test.copy()

            # foldごとに特徴量を作り直す、すなわち、target encoding用
            if transform is not None:
                x_train, x_valid, y_train, y_valid, x_test = transform(
                    Xs=(x_train, x_valid), ys=(y_train, y_valid),
                    X_test=x_test)

            if verbose > 0:
                print(f'\n-----\n {K} fold cross validation. \
                  \n Starting fold {fold_i+1}\n-----\n')
                print(f'[CV]train: {len(train_idx)} / valid: {len(valid_idx)}')
            if verbose <= 0 and 'fit_params' in train_params.keys():
                train_params['fit_params']['verbose'] = 0
            model = Trainer(copy(self.basemodel))
            model.train(x_train, y_train, x_valid, y_valid, **train_params)
            self.models.append(model.get_model())

            if verbose > 0:
                print(f'best iteration is {model.get_best_iteration()}')

            if prediction == 'predict':
                self.oof[valid_idx] = model.predict(x_valid)
            elif prediction == 'binary_proba':
                self.oof[valid_idx] = model.binary_proba(x_valid)
            else:
                self.oof[valid_idx] = model.predict(x_valid)

            if X_test is not None:
                if prediction == 'predict':
                    self.pred += model.predict(x_test) / K
                elif prediction == 'binary_proba':
                    self.pred += model.binary_proba(x_test) / K
                else:
                    self.pred += model.predict(x_test) / K

            self.imps[:, fold_i] = model.get_feature_importances()

            for eval_name, metric in eval_metrics.items():
                score = metric(y_valid, self.oof[valid_idx])
                self.scores.loc[eval_name, fold_i] = score

            if verbose >= 0:
                log_str = f'[CV] Fold {fold_i+1}:'
                log_str += ''.join(
                    [f' {key}={self.scores.loc[key, fold_i]:.5f}'
                     for key in eval_metrics.keys()])
                log_str += f' (iter {model.get_best_iteration()})'
                print(log_str)

        self.scores['mean'], self.scores['sd'] = self.scores.mean(
            axis=1), self.scores.std(axis=1, ddof=0)
        self.scores['se'] = self.scores['sd'] / np.sqrt(K)
        self.scores['ci'] = self.scores['se'] * 1.96

        log_str = f'[CV] Overall:'
        for key in eval_metrics.keys():
            mean = self.scores.loc[key, 'mean']
            sd = self.scores.loc[key, 'sd']
            log_str += f' {key}={mean:.5f} (SD={sd:.5f})'
        print(log_str)

    def plot_feature_importances(self, columns):
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps_mean = np.mean(self.imps, axis=1)
        imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
        order = np.argsort(imps_mean)
        plt.barh(np.array(columns)[order],
                 imps_mean[order], xerr=imps_se[order])
        plt.show()

    def save_feature_importances(self, columns, path):
        plt.figure(figsize=(5, -(-len(columns) // 3)))
        imps_mean = np.mean(self.imps, axis=1)
        # imps_sd = np.std(self.imps, axis=1, ddof=0)
        imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
        # imps_ci = np.std(self.imps, axis=1) / \
        #     np.sqrt(self.imps.shape[0]) * 1.96
        order = np.argsort(imps_mean)
        # order = order[-np.count_nonzero(imps_mean == 0):]
        plt.barh(np.array(columns)[order],
                 imps_mean[order], xerr=imps_se[order])
        # plt.xlabel('')
        # plt.ylabel('')
        plt.savefig(path)

    def save(self, path):
        objects = [
            self.basemodel, self.datasplit, self.models,
            self.oof, self.pred, self.imps, self.id_test, self.scores
        ]
        with open(path, 'wb') as f:
            pickle.dump(objects, f)

    def load(self, path):
        with open(path, 'rb') as f:
            objects = pickle.load(f)
        self.basemodel, self.datasplit, self.models, \
            self.oof, self.pred, self.imps, self.id_test, self.scores = objects

    def save_prediction(self, path, ID='ID', y='y', header=True):
        df = pd.DataFrame()
        df[ID] = self.id_test
        df[y] = self.pred
        df.to_csv(path, encoding='utf-8', index=False, header=header)


def weighted_average(weight, matrix):
    if np.count_nonzero(weight < 0) > 0:
        return weight.dot(matrix) * 0
    else:
        return weight.dot(matrix)


# これもK-foldすべき　もっというと、上に統合すべき
class Blender:
    """
    train, get_weight, predictはTrainer
    run, save_predictionはCrossValidatorに組み込む
    """

    def train(self, X, y, X_valid=None, y_valid=None,
              eval_metric=None):
        X_list = X.values.T

        def objective(weight):
            oof_blend = weighted_average(weight, X_list)
            return eval_metric(y, oof_blend)

        result = minimize(
            objective,
            x0=np.ones(len(X_list)) / len(X_list),
            constraints=(
                {'type': 'eq', 'fun': lambda x: 1 - x.sum()},
            ),
            method='Nelder-Mead'
        )
        self.weight = result.x

    def get_weight(self):
        return self.weight

    def predict(self, X):
        return weighted_average(self.weight, X.values.T)

    def run(self,  X, y, X_test=None, id_test=None,
            eval_metric=None):
        self.id_test = id_test
        self.train(X, y, eval_metric=eval_metric)
        self.oof = self.predict(X)
        self.pred = self.predict(X_test)
        self.score = eval_metric(y, self.oof)

    def save_prediction(self, path, ID='ID', y='y', header=True):
        df = pd.DataFrame()
        df[ID] = self.id_test
        df[y] = self.pred
        df.to_csv(path, encoding='utf-8', index=False, header=header)
