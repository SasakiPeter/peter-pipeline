from pipeline.conf import settings
from pipeline.train import CrossValidator


def make_submit_file():
    cv = CrossValidator()
    models_path = f'models/{settings.PROJECT_ID}.pkl'
    cv.load(models_path)
    # X, y = load_train()
    # columns = X.columns.values
    # cv.save_feature_importances(columns, 'hoge.png')

    submit_path = f'submits/{settings.PROJECT_ID}.csv'
    cv.save_prediction(submit_path)
