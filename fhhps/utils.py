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
    df["A3"] = df["A1"] + df["U2"] + df["U3"]

    df["B2"] = df["B1"] + df["V2"]
    df["B3"] = df["B1"] + df["V2"] + df["V3"]

    df["C2"] = df["C1"] + df["W2"]
    df["C3"] = df["C1"] + df["W2"] + df["W3"]

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


def get_true_output_cond_means(fake):
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values
    X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
    Z1, Z2, Z3 = Z[:, 0], Z[:, 1], Z[:, 2]

    EA1, EB1, EC1 = get_true_coef_cond_means(fake).T

    shock_means = true_shock_means(fake)
    EU2, EV2, EW2 = shock_means[:, 1]
    EU3, EV3, EW3 = shock_means[:, 2]

    EY1 = EA1 + EB1 * X1 + EC1 * Z1
    EY2 = EA1 + EU2 + (EB1 + EV2) * X2 + (EC1 + EW2) * Z2
    EY3 = EA1 + EU2 + EU3 + (EB1 + EV2 + EV3) * X3 + (EC1 + EW2 + EW3) * Z3
    cond_means = np.column_stack([EY1, EY2, EY3])
    return cond_means


def get_true_output_cond_cov(fake):
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values

    shock_cov = true_shock_cov(fake)
    VarU2, VarV2, VarW2, CovU2V2, CovU2W2, CovV2W2 = shock_cov[:, 1]
    VarU3, VarV3, VarW3, CovU3V3, CovU3W3, CovV3W3 = shock_cov[:, 2]

    X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
    Z1, Z2, Z3 = Z[:, 0], Z[:, 1], Z[:, 2]

    VarA1, VarB1, VarC1, CovA1B1, CovA1C1, CovB1C1 = get_true_coef_cond_cov(fake)[0]
    VarA2 = VarA1 + VarU2
    VarA3 = VarA1 + VarU2 + VarU3
    VarB2 = VarB1 + VarV2
    VarB3 = VarB1 + VarV2 + VarW3
    VarC2 = VarB1 + VarW2
    VarC3 = VarB1 + VarW2 + VarW3
    CovA2B2 = CovA1B1 + CovU2V2
    CovA3B3 = CovA1B1 + CovU2V2 + CovU3V3
    CovA2C2 = CovA1C1 + CovU2W2
    CovA3C3 = CovA1C1 + CovU2W2 + CovU3W3
    CovB2C2 = CovA1B1 + CovV2W2
    CovB3C3 = CovA1B1 + CovV2W2 + CovV3W3

    VarY1 = VarA1 + VarB1 * X1 ** 2 + VarC1 * Z1 ** 2 + \
            2 * CovA1B1 * X1 + 2 * CovA1C1 * Z1 + 2 * CovB1C1 * X1 * Z1
    VarY2 = VarA2 + VarB2 * X2 ** 2 + VarC2 * Z2 ** 2 + \
            2 * CovA2B2 * X2 + 2 * CovA2C2 * Z2 + 2 * CovB2C2 * X2 * Z2
    VarY3 = VarA3 + VarB3 * X3 ** 2 + VarC3 * Z3 ** 2 + \
            2 * CovA3B3 * X3 + 2 * CovA3C3 * Z3 + 2 * CovB3C3 * X3 * Z3
    CovY1Y2 = VarA1 + VarB1 * X1 * X2 + VarC1 * Z1 * Z2 + \
              CovA1B1 * (X1 + X2) + CovA1C1 * (Z1 + Z2) + CovB1C1 * (Z2 * X1 + Z1 * X2)
    CovY1Y3 = VarA1 + VarB1 * X1 * X3 + VarC1 * Z1 * Z3 + \
              CovA1B1 * (X1 + X3) + CovA1C1 * (Z1 + Z3) + CovB1C1 * (Z3 * X1 + Z1 * X3)
    CovY2Y3 = VarA1 + VarB1 * X2 * X3 + VarC1 * Z2 * Z3 + \
              CovA1B1 * (X2 + X3) + CovA1C1 * (Z2 + Z3) + CovB1C1 * (Z2 * X3 + Z3 * X2) \
              + VarU2 + VarV2 * X2 * X3 + VarW2 * Z2 * Z3 + \
              CovU2V2 * (X2 + X3) + CovU2W2 * (Z2 + Z3) + CovV2W2 * (Z2 * X3 + Z3 * X2)
    cond_cov = np.column_stack([VarY1, VarY2, VarY3, CovY1Y2, CovY1Y3, CovY2Y3])
    return cond_cov


def get_true_coef_cond_means(fake):
    df = fake["df"]
    xz_idx = ["X1", "X2", "X3", "Z1", "Z2", "Z3"]
    ab_idx = ["A1", "B1", "C1"]

    XZ = df[xz_idx].values
    AB = df[ab_idx].values

    muAB = AB.mean(0).reshape(-1, 1)
    muXZ = fake["means"][xz_idx].values.reshape(-1, 1)

    sigma = df[xz_idx + ab_idx].cov()
    sigmaXZ = sigma.loc[xz_idx, xz_idx].values
    sigmaXZinv = np.linalg.inv(sigmaXZ)
    sigmaABXZ = sigma.loc[ab_idx, xz_idx].values

    cond_means = (muAB + sigmaABXZ @ sigmaXZinv @ (XZ.T - muXZ)).T
    return cond_means


def get_true_coef_cond_cov(fake):
    # Note: joint normality implies homoskedasticity
    n = len(fake["df"])
    xz_idx = ["X1", "X2", "X3", "Z1", "Z2", "Z3"]
    ab_idx = ["A1", "B1", "C1"]

    sigma = fake["cov"]
    sigmaAB = sigma.loc[ab_idx, ab_idx].values
    sigmaXZ = sigma.loc[xz_idx, xz_idx].values
    sigmaXZinv = np.linalg.inv(sigmaXZ)
    sigmaABXZ = sigma.loc[ab_idx, xz_idx].values

    sigma_cond = sigmaAB - sigmaABXZ @ sigmaXZinv @ sigmaABXZ.T
    cov_idx = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
    cond_cov = sigma_cond[cov_idx[:, 0], cov_idx[:, 1]]
    out = np.tile(cond_cov, (n, 1))
    return out


def true_shock_means(fake):
    shock_means = np.column_stack([
        np.zeros(3),
        fake["means"][["U2", "V2", "W2"]],
        fake["means"][["U3", "V3", "W3"]]
    ])
    return shock_means


def true_shock_cov(fake):
    s2_idx = ["U2", "V2", "W2"]
    s3_idx = ["U3", "V3", "W3"]
    cov_idx = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
    shock_means = np.column_stack([
        np.zeros(6),
        fake["cov"].loc[s2_idx, s2_idx].values[cov_idx[:, 0], cov_idx[:, 1]],
        fake["cov"].loc[s3_idx, s3_idx].values[cov_idx[:, 0], cov_idx[:, 1]]
    ])
    return shock_means
