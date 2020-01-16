import pandas as pd
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
