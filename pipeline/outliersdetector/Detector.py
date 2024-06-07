import logging
import dill
import os

import numpy as np


dirname = os.path.dirname(__file__)


class Detector:
    def __init__(self, *args, **kwargs):
        print("run")
    def predict(self, X, feature_names=[]):
        with open(os.path.join(dirname, "preprocessor.dill"), "rb") as prep_f:
            self.preprocessor = dill.load(prep_f)
        with open(os.path.join(dirname, "model.dill"), "rb") as model_f:
            self.od = dill.load(model_f)
        logging.info("Input: " + str(X))

        X_prep = self.preprocessor.transform(X)
        output = self.od.predict(X_prep)['data']['is_outlier']

        logging.info("Output: " + str(output))
        return output