#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class Linear2(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        self.causal = np.array(range(0, 101, 10))
        X = X[:, self.causal]

        y = X.sum(axis=1)

        return y


if __name__ == "__main__":
    Linear2(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
