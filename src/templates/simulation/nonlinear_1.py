#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class NonLinear1(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        self.causal = np.array(range(0, 31, 10))
        X = X[:, self.causal]

        x1 = 5 * X[:, 0]
        x2 = 2 * np.sin(np.pi * X[:, 1] / 2)
        x3 = 2 * X[:, 2] * (X[:, 2] > 0).astype(int)
        x4 = 2 * np.exp(5 * X[:, 3])

        y = x1 + x2 + x3 + x4

        return y


if __name__ == "__main__":
    NonLinear1(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
