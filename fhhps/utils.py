from time import time

import numpy as np
import pandas as pd
from numba import njit
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
                  VA=1.5, VB=.4, VC=.4,
                  VU=1, VV=.1, VW=.1,
                  rho=.5,
                  seed=None):
    """
    Generates jointly Normal data
    """
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


@njit()
def bandwidth_selector(X, method="scott"):
    """
    A = minimum of std(X, ddof=1) and normalized IQR(X)
    C = depends on method

    References
    ----------
    Silverman (1986) p.47
    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    if method == "scott":
        C = 1.059
    elif method == "silverman":
        C = 0.9
    else:
        C = 1

    k = X.shape[1]
    n = X.shape[0]
    bw = np.empty(shape=k, dtype=np.float64)
    for i in range(k):
        x = X[:, i]
        A1 = np.diff(np.percentile(x, q=[25, 75])) / 1.349
        A2 = np.std(x)
        A = np.minimum(A1, A2).item()
        bw[i] = C * A * n ** (-0.2)
    return bw


def true_conditional_mean(muA, muB, sigmaAB, sigmaB, value):
    return muA + sigmaAB @ np.linalg.inv(sigmaB) @ (value - muB)


def true_output_cond_mean(fake):
    xz = ["X1", "X2", "X3", "Z1", "Z2", "Z3"]
    y = ["Y1", "Y2", "Y3"]
    muA = np.array(fake["df"][y].mean()).reshape(-1, 1)
    muB = np.array(fake["means"][xz]).reshape(-1, 1)
    sigmaB = np.array(fake["cov"].loc[xz, xz])
    sigmaAB = np.array(fake["df"].cov().loc[y, xz])
    n = len(fake["df"])
    cond_means = np.empty((n, 3))
    for i in range(n):
        xz_val = np.array(fake["df"][xz].iloc[i]).reshape(-1, 1)
        cond_means[i] = true_conditional_mean(muA, muB, sigmaAB, sigmaB, xz_val).flatten()
    return cond_means
