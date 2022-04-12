#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_FILE: path to a YAML file with the hyperparameters
    - None
Output files:
  - y_pred.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.linear_model import Lasso

from base.sklearn import SklearnModel
import utils as u


class LassoModel(SklearnModel):
    def __init__(self) -> None:
        lasso = Lasso
        super().__init__(lasso, "prediction", "lasso", "${MODEL_PARAMS}")

    def score_features(self):
        return self.clf.best_estimator_.coef_

    def select_features(self, scores):
        return scores != 0


if __name__ == "__main__":
    model = LassoModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
    u.save_proba_npz([], model.best_hyperparams)
