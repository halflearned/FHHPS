from time import time

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, block_diag


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


def simple_data(n):
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


def generate_data(n,
                  EA=2, EB=.4, EC=0.3,
                  EX=0, EZ=0,
                  EU=.3, EV=.1, EW=0.1,
                  VA=9, VB=.2, VC=.2,
                  VU=1, VV=.1, VW=.1,
                  rho=.5,
                  seed=None):
    names = ["A1", "B1", "C1",
             "X1", "X2", "X3",
             "Z1", "Z2", "Z3",
             "U2", "V2", "W2",
             "U3", "V3", "W3"]

    rng = np.random.RandomState(seed)
    corr_coef_regressors = toeplitz([1] + [rho] * 8)
    corr_shocks = toeplitz([1, rho, rho])
    corr = block_diag(corr_coef_regressors,
                      corr_shocks,
                      corr_shocks)

    m = np.array([EA, EB, EC,
                  EX, EX, EX,
                  EZ, EZ, EZ,
                  EU, EV, EW,
                  EU, EV, EW])
    v = np.array([VA, VB, VC,
                  1., 1., 1.,
                  1., 1., 1.,
                  VU, VV, VW,
                  VU, VV, VW])

    scaling = np.diag(np.sqrt(v))
    S = scaling @ corr @ scaling
    df = pd.DataFrame(data=rng.multivariate_normal(mean=m, cov=S, size=n),
                      columns=names)

    df["A2"] = df["A1"] + df["U2"]
    df["B2"] = df["B1"] + df["V2"]
    df["C2"] = df["C1"] + df["W2"]
    df["A3"] = df["A2"] + df["U3"]
    df["B3"] = df["B2"] + df["V3"]
    df["C3"] = df["C2"] + df["W3"]
    df["Y1"] = df["A1"] + df["B1"] * df["X1"] + df["C1"] * df["Z1"]
    df["Y2"] = df["A2"] + df["B2"] * df["X2"] + df["C2"] * df["Z2"]
    df["Y3"] = df["A3"] + df["B3"] * df["X3"] + df["C3"] * df["Z3"]

    cov = pd.DataFrame(S, columns=names, index=names)

    true_shock_cov = pd.DataFrame([
        [cov.loc["U2", "U2"]]
    ])

    data = dict(n=n,
                df=df,
                means=pd.Series(m, index=names),
                variances=pd.Series(v, index=names),
                cov=cov,
                corr=pd.DataFrame(corr, columns=names, index=names))

    return data


if __name__ == "__main__":
    fakedata = generate_data(n=100000)
    fakedata["corr"]
