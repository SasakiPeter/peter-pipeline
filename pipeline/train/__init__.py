import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, log_loss,
    r2_score
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from pipeline.conf import settings
from pipeline.preprocess import load_train, load_test, label_encoding
from pipeline.train.base import CrossValidator, Blender
from pipeline.train.utils import get_cvs_by_layer
from pipeline.utils.directory import provide_dir
from pipeline.utils.metrics import rmse


def get_preprocess(name):
    preprocesses = {
        'LabelEncoding': label_encoding
    }
    return preprocesses[name]


def get_split(algo, n_splits, seed):
    regression = {
        'CatBoostRegressor',
        'LGBMRegressor',
        'RandomForestRegressor',
        'LinearRegression',
        'Ridge', 'Lasso',
        'SVR',
    }
    classification = {
        'CatBoostClassifier',
        'LGBMClassifier',
        'RandomForestClassifier',
        'LogisticRegression',
        'SVC',
    }
    if algo in regression:
        return KFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    elif algo in classification:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        raise NotImplementedError(
            f'Trainer class must provide a {algo} model')


def get_model(name, params):
    models = {
        'CatBoostRegressor': CatBoostRegressor,
        'CatBoostClassifier': CatBoostClassifier,
        'LGBMRegressor': LGBMRegressor,
        'LGBMClassifier': LGBMClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'LinearRegression': LinearRegression,
        'LogisticRegression': LogisticRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'SVR': SVR,
        'SVC': SVC,
    }
    return models[name](**params)


def get_eval_metric(name):
    eval_metrics = {
        'RMSE': rmse,
        'R2': r2_score,
        'logloss': log_loss,
        'AUC': roc_auc_score
    }
    return eval_metrics[name]


def train_by_layer(layer, X, y, X_test, id_test, cv_summary, folder_path):
    for name, params in layer.items():
        if params['PREPROCESS']:
            preprocess = params['PREPROCESS']
            preprocess_funcs = [get_preprocess(name) for name in preprocess]
            for f in preprocess_funcs:
                X, X_test = f(X, X_test)

        algo_name, section_id = name.split('_')
        eval_metrics = {metric: get_eval_metric(metric)
                        for metric in params['EVAL_METRICS']}

        n_splits = params['CV']['n_splits']
        seed = params['CV']['seed']

        kf = get_split(algo_name, n_splits, seed)
        cv = CrossValidator(get_model(algo_name, params['PARAMS']), kf)
        cv.run(
            X, y, X_test, id_test,
            eval_metrics=eval_metrics,
            prediction=params['PREDICT_FORMAT'],
            train_params={
                'cat_features': settings.CAT_IDXS,
                'fit_params': params['FIT_PARAMS']
            },
            verbose=1
        )
        models_path = f'{folder_path}/{section_id}.pkl'
        cv.save(models_path)
        cv_scores_path = f'{folder_path}/{section_id}.csv'
        cv.scores.to_csv(cv_scores_path, encoding='utf-8')

        # feature importance
        image_path = f'{folder_path}/{section_id}.png'
        columns = X.columns.values
        cv.save_feature_importances(columns, image_path)

        for metric in eval_metrics.keys():
            mean = cv.scores.loc[metric, 'mean']
            se = cv.scores.loc[metric, 'se']
            ci = cv.scores.loc[metric, 'ci']
            cv_summary.loc[section_id, f'{metric}_mean'] = f'{mean:.5f}'
            cv_summary.loc[section_id, f'{metric}_se'] = f'{se:.5f}'
            cv_summary.loc[section_id, f'{metric}_ci'] = f'{ci:.5f}'
    return cv_summary


def get_oof_by_layer(layer):
    X = pd.DataFrame()
    X_test = pd.DataFrame()
    cvs = get_cvs_by_layer(layer)
    for section_id, cv in cvs.items():
        X[section_id] = cv.oof
        X_test[section_id] = cv.pred
    return X, X_test


def train():
    # initialize
    folder_path = f'models/{settings.PROJECT_ID}'
    provide_dir(folder_path)
    cv_summary = pd.DataFrame()

    # loading data
    X, y = load_train()
    X_test, id_test = load_test()

    # first layer
    first_layer = settings.FIRST_LAYER
    cv_summary = train_by_layer(
        first_layer, X, y, X_test, id_test, cv_summary, folder_path)

    # Trainer, CrossValidatorに組み込む予定
    blender = Blender()
    blender.run(X, y, X_test, id_test, eval_metric=rmse)

    path = f'{folder_path}/blend.csv'
    blender.save_prediction(path)
    with open(f'{folder_path}/blend.txt', 'w') as f:
        f.write(f'{blender.score}')

    # second layer
    second_layer = settings.SECOND_LAYER
    X, X_test = get_oof_by_layer(first_layer)
    cv_summary = train_by_layer(
        second_layer, X, y, X_test, id_test, cv_summary, folder_path)

    path = f'{folder_path}/summary.csv'
    cv_summary.to_csv(path, encoding='utf-8')

    # 5分割ぐらいでやっても、ほとんどエラーバーで覆い尽くされて意味ない
    # ここ、お試しでやっているので要リファクタ
    import matplotlib.pyplot as plt
    import numpy as np
    columns = cv_summary.index.values
    for metric in ['R2', 'RMSE']:
        plt.figure(figsize=(5, -(-len(columns) // 3)))
        mean = cv_summary.loc[:, f'{metric}_mean']
        se = cv_summary.loc[:, f'{metric}_se']
        order = np.argsort(mean)
        plt.barh(np.array(columns)[order],
                 mean[order], xerr=se[order])
        plt.savefig(f'{folder_path}/summary-{metric}.png')
