from time import time

import numpy as np


def difference(*args, t):
    output = []
    for x in args:
        dx = x[:, t, None] - x[:, t - 1, None]
        output.append(dx)
    if len(output) == 1:
        return output[0]
    else:
        return output


def clock_seed():
    """ Simple utility to generate random file_hash based on time """
    return int(time() * 1e8 % 1e8)


def read_pickle(path):
    import pickle  # Package dill fails here!
    with open(path, "rb") as f:
        obj = pickle.load(file=f)
    return obj


def save_pickle(obj, path):
    import pickle  # Package dill fails here!
    with open(path, "wb") as f:
        pickle.dump(obj=obj, file=f)


def extract(*args, t):
    output = []
    for x in args:
        dx = x[:, t, None]
        output.append(dx)
    if len(output) == 1:
        return output[0]
    else:
        return output


def fake_data(n):
    X = np.random.uniform(-4, 4, size=(n, 3))
    Z = np.random.uniform(-4, 4, size=(n, 3))
    ABC1 = np.random.multivariate_normal(mean=[5, 0.5, 0.5],
                                         cov=np.array([[5., .4, .4],
                                                       [.4, 1, .4],
                                                       [.4, .4, 1]]),
                                         size=n)
    UVW2 = np.random.multivariate_normal(mean=[1., 0.1, 0.1],
                                         cov=np.array([[2, .05, .05],
                                                       [.05, .2, .05],
                                                       [.05, .05, .2]]),
                                         size=n)
    UVW3 = np.random.multivariate_normal(mean=[1., 0.1, 0.1],
                                         cov=np.array([[2, .05, .05],
                                                       [.05, .1, .05],
                                                       [.05, .05, .1]]),
                                         size=n)
    ABC2 = ABC1 + UVW2
    ABC3 = ABC2 + UVW3
    Y1 = ABC1[:, 0] + ABC1[:, 1] * X[:, 0] + ABC1[:, 2] * Z[:, 0]
    Y2 = ABC2[:, 0] + ABC2[:, 1] * X[:, 1] + ABC2[:, 2] * Z[:, 1]
    Y3 = ABC3[:, 0] + ABC3[:, 1] * X[:, 2] + ABC3[:, 2] * Z[:, 2]
    Y = np.vstack([Y1, Y2, Y3]).T
    return X, Z, Y