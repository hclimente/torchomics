#!/usr/bin/env python
"""
Input variables:
    - PARAMS: model parameters
    - TEST: path to numpy array with validation Y vector.
    - Y_PROBA: path to numpy array with prediction vector.
Output files:
    - prediction_stats: path to a single-line tsv with the TSV results.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

import utils as u

_, y, _ = u.read_data("${TEST_NPZ}")
y_proba = np.load("${PROBA_NPZ}")["proba"]

try:
    auc = roc_auc_score(y, y_proba)
except ValueError:
    auc = "NA"

u.save_analysis_tsv(run="${PARAMS}", metric=["auc"], value=[auc])
