PROJECT_ID = '5'


# Data source definition
DATA_PATH = {
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}

# semi target っていうかtarget複数設定して、それぞれの
# 学習器に応じて、切り替えられるように実装したい
DATA_FORMAT = {
    'id': 'ID',
    'target': 'y',
    # 'not_X':
}

CAT_IDXS = [0, 1, 2, 3, 4, 5, 6, 7]


# Training

TRAIN_CV = {
    'n_splits': 5,
    'seed': 1
}

N_LAYERS = 2

FIRST_LAYER = {
    'LGBMRegressor_LGB1': {
        'PREPROCESS': ['LabelEncoding'],
        'CV': {
            'n_splits': 5,
            'seed': 1
        },
        'PARAMS': {
            'objective': 'regression',
            'boosting': 'gbdt',
            'tree_learner': 'serial',
            'nthread': -1,
            'seed': 0,

            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 10,

            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'bagging_seed': 0,

            'save_binary': True,

            'max_bin': 255,
            'learning_rate': 0.03,

            'min_sum_hessian_in_leaf': 0.01,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_gain_to_split': 0.01,

            'verbose': -1,
            'metric': 'rmse',
            'histogram_pool_size': 1024,
            'n_estimators': 1000,
        },
        'FIT_PARAMS': {
            'early_stopping_rounds': 300,
        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    'CatBoostRegressor_CTB1': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 5,
            'seed': 1
        },
        'PARAMS': {
            'loss_function': 'RMSE',
            'eval_metric': 'R2',
            'random_seed': 608,
            'learning_rate': 0.03,

            'bootstrap_type': 'Bayesian',
            'sampling_frequency': 'PerTreeLevel',
            'sampling_unit': 'Object',

            # up to 16
            'depth': 4,
            # try diff value
            # 'l2_leaf_reg': 2,
            # 'random_strength': 1,
            # 'bagging_temperature': 0,
            'border_count': 254,

            # golden feature
            # 'per_float_feature_quantization': '0:border_count=1024'

            'grow_policy': 'SymmetricTree',
            'nan_mode': 'Forbidden',
            # 陽性の重みを増やす
            # 'scale_pos_weight': 9,
            'iterations': 1000,
        },
        'FIT_PARAMS': {
            'early_stopping_rounds': 300,
        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    'CatBoostRegressor_CTB2': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 5,
            'seed': 1
        },
        'PARAMS': {
            'loss_function': 'RMSE',
            'eval_metric': 'R2',
            'random_seed': 608,
            'learning_rate': 0.03,

            'bootstrap_type': 'Bayesian',
            'sampling_frequency': 'PerTreeLevel',
            'sampling_unit': 'Object',

            # up to 16
            'depth': 7,
            # try diff value
            # 'l2_leaf_reg': 2,
            # 'random_strength': 1,
            # 'bagging_temperature': 0,
            'border_count': 254,

            # golden feature
            # 'per_float_feature_quantization': '0:border_count=1024'

            'grow_policy': 'SymmetricTree',
            'nan_mode': 'Forbidden',
            # 陽性の重みを増やす
            # 'scale_pos_weight': 9,
            'iterations': 1000,
        },
        'FIT_PARAMS': {
            'early_stopping_rounds': 300,
        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    'RandomForestRegressor_RF-1': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 5,
            'seed': 1
        },
        'PARAMS': {
            'n_estimators': 500,
            'criterion': 'mse',
            'max_depth': 10,
            'max_features': 'sqrt'
        },
        'FIT_PARAMS': {

        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    # 'LinearRegression',
}

SECOND_LAYER = {
    'LinearRegression_STK1': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 5,
            'seed': 10
        },
        'PARAMS': {

        },
        'FIT_PARAMS': {

        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    'Ridge_STK2': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 5,
            'seed': 10
        },
        'PARAMS': {
            'alpha': 1.0,
            'random_state': 1
        },
        'FIT_PARAMS': {

        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    # 'Blender_BLD2': {
    #     'PREPROCESS': [],
    #     'CV': {
    #         'n_splits': 5,
    #         'seed': 10
    #     },
    #     'PARAMS': {
    #         'objective': 'rmse',
    #         'random_state': 1
    #     },
    #     'FIT_PARAMS': {

    #     },
    #     'EVAL_METRICS': [
    #         'RMSE',
    #         'R2'
    #     ],
    #     'PREDICT_FORMAT': 'predict'
    # },
}
