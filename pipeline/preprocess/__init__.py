import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pipeline.conf import settings


def load_train():
    path = settings.DATA_PATH['train']
    df = pd.read_csv(path)
    not_X = [settings.DATA_FORMAT['id'], settings.DATA_FORMAT['target']]
    X = df.drop(not_X, axis=1)
    y = df[settings.DATA_FORMAT['target']]
    print(X.shape, y.shape)
    return X, y


def load_test():
    path = settings.DATA_PATH['test']
    df = pd.read_csv(path)
    X = df.drop([settings.DATA_FORMAT['id']], axis=1)
    ID = df[settings.DATA_FORMAT['id']]
    print(X.shape)
    return X, ID


def get_object_columns(df):
    return [key for key, value in df.dtypes.items() if value == 'object']


def label_encoding(X_train, X_test):
    df = pd.concat([X_train, X_test], axis=0)
    columns = get_object_columns(df)
    for col in columns:
        le = LabelEncoder()
        le.fit(df[col].values)
        X_train[col] = le.transform(X_train[col].values)
        X_test[col] = le.transform(X_test[col].values)
    return X_train, X_test


def preprocess():
    X_train, y_train = load_train()
    X_test, id_test = load_test()
    X_train, X_test = label_encoding(X_train, X_test)
    print('動作確認OK')
