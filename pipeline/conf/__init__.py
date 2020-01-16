import os
import importlib

ENVIRONMENT_VARIABLE = "SETTINGS_MODULE"


class Settings:
    def __init__(self, settings_module):
        self.SETTINGS_MODULE = settings_module
        mod = importlib.import_module(self.SETTINGS_MODULE)

        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)

                setattr(self, setting, setting_value)


settings = Settings(os.environ.get(ENVIRONMENT_VARIABLE))
