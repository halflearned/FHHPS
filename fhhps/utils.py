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

    data = dict(n=n,
                df=df,
                means=pd.Series(m, index=names),
                variances=pd.Series(v, index=names),
                cov=cov,
                corr=pd.DataFrame(corr, columns=names, index=names))

    return data

