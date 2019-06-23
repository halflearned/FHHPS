from itertools import product

import numpy as np

from fhhps.estimator import cov_excess_matrix


def test_cov_excess_matrix():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    Z = np.array([[7, 8, 9],
                  [10, 11, 12]])
    for s, t in product(range(3), range(3)):
        out = cov_excess_matrix(X, Z, t, s)
        assert np.all(out[:, 0] == 1)
        assert np.all(out[:, 1] == X[:, t] * X[:, s])
        assert np.all(out[:, 2] == Z[:, t] * Z[:, s])
        assert np.all(out[:, 3] == X[:, t] + X[:, s])
        assert np.all(out[:, 4] == Z[:, t] + Z[:, s])
        assert np.all(out[:, 5] == X[:, t] * Z[:, s] + X[:, s] * Z[:, t])
