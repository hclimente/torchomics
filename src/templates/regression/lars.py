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
  - y_pred.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.linear_model import Lars

from base.sklearn import SklearnModel


class LarsModel(SklearnModel):
    def __init__(self) -> None:
        lars = Lars
        super().__init__(lars, "prediction", "lars", "${MODEL_PARAMS}")

    def score_features(self):
        return self.clf.best_estimator_.coef_

    def select_features(self, scores):
        return scores != 0


if __name__ == "__main__":
    model = LarsModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
