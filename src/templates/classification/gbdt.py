#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_FILE: path to a YAML file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from lightgbm import LGBMClassifier

from base.sklearn import SklearnModel


class GBDTModel(SklearnModel):
    def __init__(self) -> None:
        gbdt = LGBMClassifier
        super().__init__(gbdt, "prediction", "gbdt", "${MODEL_PARAMS}")
        self.fixed_params.update({"objective": "binary", "n_estimators": 1000})

    def score_features(self):
        return self.clf.best_estimator_.feature_importances_

    def select_features(self, scores):
        return [True for _ in scores]


if __name__ == "__main__":
    model = GBDTModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}")
    model.predict_proba("${TEST_NPZ}", "${SCORES_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
