from pipeline.conf import settings
from pipeline.utils.directory import provide_dir
from pipeline.train.utils import get_cvs_by_layer


def make_submit_files_by_layer(layer):
    dir_path = f'submits/{settings.PROJECT_ID}'
    provide_dir(dir_path)

    cvs = get_cvs_by_layer(layer)
    for section_id, cv in cvs.items():
        submit_path = f'{dir_path}/{section_id}.csv'
        cv.save_prediction(submit_path)


def predict():
    layers = [settings.FIRST_LAYER, settings.SECOND_LAYER]
    for layer in layers:
        make_submit_files_by_layer(layer)
