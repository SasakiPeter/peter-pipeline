from pipeline.conf import settings
from pipeline.train.base import CrossValidator


def get_cvs_by_layer(layer):
    cvs = {}
    for name, params in layer.items():
        algo_name, section_id = name.split('_')
        model_path = f'models/{settings.PROJECT_ID}/{section_id}.pkl'
        cv = CrossValidator()
        cv.load(model_path)
        cvs[section_id] = cv
    return cvs
