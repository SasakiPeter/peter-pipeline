from pipeline.conf import settings
from pipeline.train import CrossValidator


# 全ての学習器に対して、予測ファイルの出力を行う。
# PRODJECT_IDをフォルダ名にもち、section_idがファイル名になるようにする。
def make_submit_file():
    cv = CrossValidator()
    models_path = f'models/{settings.PROJECT_ID}-LGB1.pkl'
    # models_path = f'models/{settings.PROJECT_ID}-{section_id}.pkl'
    cv.load(models_path)
    # X, y = load_train()
    # columns = X.columns.values
    # cv.save_feature_importances(columns, 'hoge.png')
    # print(dir(cv))
    print(cv.scores)

    # submit_path = f'submits/{settings.PROJECT_ID}/{section_id}.csv'
    # cv.save_prediction(submit_path)


def predict():
    pass
