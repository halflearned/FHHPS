#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEGACY version of FHHPS.
This version has been DEPRECATED.
"""

import warnings
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import PolynomialFeatures


# from tqdm import tqdm


class NegativeVarianceError(Exception):
    pass


class ExtremeCorrelationError(Exception):
    pass


class ModelNotYetFitError(Exception):
    pass


class BootstrapNotYetRunError(Exception):
    pass


class DataNotYetAddedError(Exception):
    pass


def epanechnikov(u, h):
    return np.where(np.abs(u) < h, 0.75 * (1 - (u / h) ** 2), 0)


def uniform(u, h):
    return 1 / (h / 2) * np.where(np.abs(u) < h / 2, 1, 0)


def gaussian(u, h):
    return norm(scale=h).pdf(u)


def g1inv(x1, x2):
    if x2 != x1:
        return 1 / (x2 - x1) * np.array([[x2, -x1],
                                         [-1, 1]])
    else:
        warn("Found value X1 = X2 up to machine precision. " \
             "Conditional moments will likely be NaNs for this value.",
             RuntimeWarning)
        return np.full((2, 2), np.nan)


def g2inv(x1, x2):
    if x2 != x1:
        return 1 / (x2 - x1) ** 2 * \
               np.array([[x2 ** 2, x1 ** 2, -2 * x1 * x2],
                         [1, 1, -2],
                         [-x2, -x1, x1 + x2]])
    else:
        warn("Found value X1 = X2 up to machine precision. " \
             "Conditional moments will likely be NaNs for this value.",
             RuntimeWarning)
        return np.full((3, 3), np.nan)


class FHHPS:

    def __init__(self,
                 c_shocks=4,
                 c_nw=0.1,
                 c1_cens=1,
                 c2_cens=1,
                 alpha_shocks=0.20,
                 alpha_nw=1 / 2,
                 alpha_cens_1=0.25,
                 alpha_cens_2=0.125,
                 kernel="epanechnikov",
                 poly_order=2):

        # Setup
        self.csh = c_shocks
        self.cnw = c_nw
        self.c1cens = c1_cens
        self.c2cens = c2_cens
        self.ash = alpha_shocks
        self.anw = alpha_nw
        self.acens1 = alpha_cens_1
        self.acens2 = alpha_cens_2
        self.poly_order = poly_order
        self.kernel = epanechnikov

        # To be populated later
        self._data = None
        self._n = None
        self._fitted = False
        self._conditional = None
        self._bootstrapped_values = None
        self._conditional_bootstrapped_values = None

        # Convenient variable groupings 
        self._shock_vars = ["EU", "EV", "SU", "SV", "CUV", "CorrUV"]
        self._rc1_vars = ["EA1", "EB1", "SA1", "SB1", "CA1B1", "CorrA1B1"]
        self._rc2_vars = ["EA2", "EB2", "SA2", "SB2", "CA2B2", "CorrA2B2"]
        self._conditional_rc_vars = ["EA1x", "EB1x", "SA1x", "SB1x", "CA1B1x", "CorrA1B1x"]
        self._latex_names = {"EU": "$E[U_2]$",
                             "EV": "$E[V_2]$",
                             "SU": "$Std[U_2]$",
                             "SV": "$Std[V_2]$",
                             "CUV": "$Cov[U_2, V_2]$",
                             "CorrUV": "$Corr[U_2, V_2]$",
                             "EA1": "$E[A_1]$",
                             "EB1": "$E[B_1]$",
                             "SA1": "$Std[A_1]$",
                             "SB1": "$Std[B_1]$",
                             "CA1B1": "$Cov[A_1, B_1]$",
                             "CorrA1B1": "$Corr[A_1, B_1]$",
                             "EA1x": "$E[A_1|X]$",
                             "EB1x": "$E[B_1|X]$",
                             "EA1sqx": "$E[A_1^2|X]$",
                             "EB1sqx": "$E[B_1^2|X]$",
                             "EA1B1x": "$E[AB|X]$",
                             "SA1x": "$Std[A_1|X]$",
                             "SB1x": "$Std[B_1|X]$",
                             "CA1B1x": "$Cov[A_1, B_1|X]$",
                             "CorrA1B1x": "$Corr[A_1, B_1|X]$",
                             "EA2": "$E[A_2]$",
                             "EB2": "$E[B_2]$",
                             "SA2": "$Std[A_2]$",
                             "SB2": "$Std[B_2]$",
                             "CA2B2": "$Cov[A_2, B_2]$",
                             "CorrA2B2": "$Corr[A_2, B_2]$"}

    def __repr__(self):
        return "FHHPS(c_shocks={}, ".format(self.csh) + \
               "c_nw={}, ".format(self.cnw) + \
               "c1_cens={}, ".format(self.c1cens) + \
               "c2_cens={}, ".format(self.c2cens) + \
               "alpha_shocks={} ,".format(self.ash) + \
               "alpha_nw={}, ".format(self.anw) + \
               "alpha_cens_1={}, ".format(self.acens1) + \
               "alpha_cens_2={}, ".format(self.acens2) + \
               "kernel={}, ".format(self.kernel.__name__) + \
               "poly_order={})".format(self.poly_order)

    def __str__(self):
        return "FHHPS method using: \n" + \
               "\tc_shocks: {}\n".format(self.csh) + \
               "\tc_nw: {}\n".format(self.cnw) + \
               "\tc1_cens: {}\n".format(self.c1cens) + \
               "\tc2_cens: {}\n".format(self.c2cens) + \
               "\talpha_shocks: {}\n".format(self.ash) + \
               "\talpha_nw: {}\n".format(self.anw) + \
               "\talpha_cens_1: {}\n".format(self.acens1) + \
               "\talpha_cens_2: {}\n".format(self.acens2) + \
               "\tkernel: {}\n".format(self.kernel.__name__) + \
               "\tpoly_order: {}\n".format(self.poly_order)

    @property
    def data(self):
        if self._data is None:
            raise ModelNotYetFitError("No data yet. Fit your model first.")
        else:
            return pd.DataFrame({"X1": self._data["X1"].flatten(),
                                 "X2": self._data["X2"].flatten(),
                                 "Y1": self._data["Y1"].flatten(),
                                 "Y2": self._data["Y2"].flatten()})

    @property
    def n(self):
        if self._n is None:
            raise DataNotYetAddedError("No data yet. Fit your model first.")
        else:
            return self._n

    @property
    def shocks(self):
        try:
            return self._shocks
        except NameError:
            raise ModelNotYetFitError("No data yet. Fit your model first.")

    def _prepare_data(self, X1, X2, Y1, Y2):
        self._n = X1.shape[0]
        data = [X1, X2, Y1, Y2]
        for i, v in enumerate(data):
            if isinstance(v, pd.Series):
                data[i] = data[i].values.reshape(self._n, 1)
            elif isinstance(v, np.ndarray):
                data[i] = data[i].reshape(self._n, 1)
            elif isinstance(v, list):
                data[i] = np.array(data[i]).reshape(self._n, 1)
            else:
                raise ValueError("Could not understand input data type." \
                                 "Each one of X1,X2,Y1,Y2 must be:" \
                                 "a pandas Series, a numpy array, or a list.")

        self._data = dict(zip(["X1", "X2", "Y1", "Y2"], data))
        return self._data

    def _setup_tuning_parameters(self):
        # Set up tuning parameters
        sx = np.std(self._data["X2"])
        self.bw_nw = self.cnw * sx * self._n ** (-self.anw)
        self.bw_shocks = self.csh * self._n ** (-self.ash)
        self.t1 = self.c1cens * sx * self._n ** (-self.acens1)
        self.t2 = self.c2cens * sx * self._n ** (-self.acens2)

    def add_data(self, X1, X2, Y1, Y2):
        self._prepare_data(X1, X2, Y1, Y2)
        self._setup_tuning_parameters()

    def fit(self):

        if self._data is None:
            raise DataNotYetAddedError("Please use the add_data method first.")
        self._conditional = False

        # Estimation
        self.estimate_shocks()
        self.get_ycmoms()
        self.get_ab_first_cmoms()
        self.get_ab_second_cmoms()
        self.get_ab_umoments()
        self._fitted = True

    def conditional_fit(self, x1, x2):
        if self._data is None:
            raise DataNotYetAddedError("Please use the add_data method first.")

        self.estimate_shocks()
        self.get_ycmoms(x1, x2)
        self.get_ab_first_cmoms(x1, x2)
        self.get_ab_second_cmoms(x1, x2)
        self._fitted = True

    def estimates(self):
        if self._ab_umoments is None:
            raise ModelNotYetFitError("Please use the fit method first.")

        res = self._shocks
        res.update(self._ab_umoments)
        tab = pd.DataFrame(index=[self._n],
                           data=res)
        return tab

    def conditional_estimates(self):
        # if self._ab_first_cmoms is None:
        #    raise ModelNotYetFitError("Please use the conditional_fit method first."

        EAx = self._ab_cmoms["EA1x"][0, 0]
        EBx = self._ab_cmoms["EB1x"][0, 0]
        EAsqx = self._ab_cmoms["EA1sqx"][0, 0]
        EBsqx = self._ab_cmoms["EB1sqx"][0, 0]
        EABx = self._ab_cmoms["EA1B1x"][0, 0]

        out = pd.DataFrame(index=[self._n],
                           data={"EU": self._shocks["EU"],
                                 "EV": self._shocks["EV"],
                                 "SU": self._shocks["SU"],
                                 "SV": self._shocks["SV"],
                                 "CUV": self._shocks["CUV"],
                                 "CorrUV": self._shocks["CorrUV"],
                                 "EA1x": EAx,
                                 "EB1x": EBx,
                                 "SA1x": np.sqrt(EAsqx - EAx ** 2),
                                 "SB1x": np.sqrt(EBsqx - EBx ** 2),
                                 "CA1B1x": EABx - EAx * EBx})

        out["CorrA1B1x"] = out["CA1B1x"] / (out["SA1x"] * out["SB1x"])
        return out

    def estimate_shocks(self):

        X1 = self._data["X1"]
        X2 = self._data["X2"]
        DX2 = X2 - X1
        DY = (self._data["Y2"] - self._data["Y1"]).flatten()

        poly = PolynomialFeatures(degree=self.poly_order,
                                  include_bias=False)
        XX1 = poly.fit_transform(np.hstack([X2, DX2.reshape(-1, 1)]))
        XX2 = poly.fit_transform(np.hstack([2 * X2, X2 ** 2, DX2.reshape(-1, 1) ** 2]))

        w = self.kernel(DX2, self.bw_shocks).flatten()

        ols1 = LinearRegression()
        ols1.fit(XX1, DY, w)
        EU = ols1.intercept_
        EV = ols1.coef_[0]

        ols2 = LinearRegression()
        XX2 = np.hstack([2 * X2, X2 ** 2])
        DYsq = DY ** 2
        ols2.fit(XX2, DYsq, w)
        EUsq = ols2.intercept_
        EUV, EVsq = ols2.coef_.flatten()[:2]
        VU, VV, CUV = EUsq - EU ** 2, EVsq - EV ** 2, EUV - EU * EV
        CorrUV = CUV / (np.sqrt(VU) * np.sqrt(VV))

        if VU < 0 or VV < 0:
            warn("Shocks variances are negative." \
                 "Consider changing the parameters.",
                 RuntimeWarning)
        if CorrUV < -1 or CorrUV > 1:
            warn("Shocks correlation is not within [-1,1]" \
                 "Consider changing the parameters.",
                 RuntimeWarning)

        self._shocks = {"EU": EU,
                        "EV": EV,
                        "EUsq": EUsq,
                        "EUV": EUV,
                        "EVsq": EVsq,
                        "VU": VU,
                        "VV": VV,
                        "CUV": CUV,
                        "SU": np.sqrt(VU),
                        "SV": np.sqrt(VV),
                        "CorrUV": CorrUV}

        return self._shocks

    def get_ycmoms(self, x1=None, x2=None):

        K = pairwise_kernels(
            X=np.hstack([self._data["X1"], self._data["X2"]]),
            Y=np.array([[x1, x2]]) if x1 is not None else None,
            metric="rbf",
            gamma=1 / self.bw_nw)

        denom = np.sum(K, axis=0)

        def fitter(y):
            num = np.sum(K * y, axis=0)
            return (num / denom).reshape(-1, 1)

        self._ycmoms = {"EY1": fitter(self._data["Y1"]),
                        "EY2": fitter(self._data["Y2"]),
                        "EY1sq": fitter(self._data["Y1"] ** 2),
                        "EY2sq": fitter(self._data["Y2"] ** 2),
                        "EY1Y2": fitter(self._data["Y1"] * self._data["Y2"])}

        return self._ycmoms

    def get_ab_first_cmoms(self, x1=None, x2=None):

        EY1 = self._ycmoms["EY1"]
        EY2 = self._ycmoms["EY2"]
        EU = self._shocks["EU"]
        EV = self._shocks["EV"]
        n = self._n if x1 is None else 1
        X1 = np.array([x1]) if x1 is not None else self._data["X1"]
        X2 = np.array([x2]) if x2 is not None else self._data["X2"]

        EY = np.hstack([EY1, EY2 - EU - EV * X2])

        mab = np.full((2, n, 1), np.nan)

        for k, (x1, x2, ey) in enumerate(zip(X1.ravel(),
                                             X2.ravel(),
                                             EY)):
            mab[:, k] = g1inv(x1, x2) @ ey.reshape(-1, 1)

        self._ab_cmoms = {"EA1x": mab[0, :],
                          "EB1x": mab[1, :]}

        return self._ab_cmoms

    def get_ab_second_cmoms(self, x1=None, x2=None):

        EU = self._shocks["EU"]
        EV = self._shocks["EV"]
        EUsq = self._shocks["EUsq"]
        EVsq = self._shocks["EVsq"]
        EUV = self._shocks["EUV"]
        X1 = np.array([x1]) if x1 is not None else self._data["X1"]
        X2 = np.array([x2]) if x2 is not None else self._data["X2"]
        EA1 = self._ab_cmoms["EA1x"]
        EB1 = self._ab_cmoms["EB1x"]
        EY1sq = self._ycmoms["EY1sq"]
        EY2sq = self._ycmoms["EY2sq"]
        EY1Y2 = self._ycmoms["EY1Y2"]
        n = self._n if x1 is None else 1

        CY = np.hstack([
            EY1sq,
            EY2sq - (EUsq + EUV * 2 * X2 + EVsq * X2 ** 2) - 2 * (EA1 + EB1 * X2) * (EU + EV * X2),
            EY1Y2 - (EA1 + EB1 * X1) * (EU + EV * X2)
        ])

        sab = np.full((3, n, 1), np.nan)

        for k, (x1, x2, cy) in enumerate(zip(X1.ravel(),
                                             X2.ravel(),
                                             CY)):
            sab[:, k] = g2inv(x1, x2) @ cy.reshape(-1, 1)

        self._ab_cmoms.update({"EA1sqx": sab[0],
                               "EB1sqx": sab[1],
                               "EA1B1x": sab[2]})

        return self._ab_cmoms

    def get_ab_umoments(self):

        absDX = np.abs(self._data["X2"] - self._data["X1"])
        valid1 = absDX > self.t1
        valid2 = absDX > self.t2

        s = self._shocks

        mom = {"EA1": np.mean(self._ab_cmoms["EA1x"][valid1]),
               "EB1": np.mean(self._ab_cmoms["EB1x"][valid1]),
               "EA1sq": np.mean(self._ab_cmoms["EA1sqx"][valid2]),
               "EB1sq": np.mean(self._ab_cmoms["EB1sqx"][valid2]),
               "EA1B1": np.mean(self._ab_cmoms["EA1B1x"][valid2])}

        mom.update({"VA1": mom["EA1sq"] - mom["EA1"] ** 2,
                    "VB1": mom["EB1sq"] - mom["EB1"] ** 2,
                    "CA1B1": mom["EA1B1"] - mom["EA1"] * mom["EB1"]})

        mom.update({"EA2": mom["EA1"] + s["EU"],
                    "EB2": mom["EB1"] + s["EV"],
                    "VA2": mom["VA1"] + s["VU"],
                    "VB2": mom["VB1"] + s["VV"],
                    "CA2B2": mom["CA1B1"] + s["CUV"]})

        mom.update({"SA1": np.sqrt(mom["VA1"]),
                    "SB1": np.sqrt(mom["VB1"]),
                    "SA2": np.sqrt(mom["VA2"]),
                    "SB2": np.sqrt(mom["VB2"])})

        mom.update({"CorrA1B1": mom["CA1B1"] / (mom["SA1"] * mom["SB1"]),
                    "CorrA2B2": mom["CA2B2"] / (mom["SA2"] * mom["SB2"])})

        self._ab_umoments = mom

        return self._ab_umoments

    def conditional_random_coefficient_moments(self):
        try:
            table = pd.DataFrame({k: x.flatten() for k, x
                                  in self._ab_cmoms.items()})
        except KeyError:
            raise ModelNotYetFitError("Please fit your model first.")

        with np.errstate(invalid="ignore"):
            table["SA1x"] = np.sqrt(table["EA1sqx"] - table["EA1x"] ** 2)
            table["SB1x"] = np.sqrt(table["EB1sqx"] - table["EB1x"] ** 2)
            table["CA1B1x"] = table["EA1B1x"] - table["EA1x"] * table["EB1x"]
            table["CorrA1B1x"] = table["CA1B1x"] / (table["SA1x"] * table["SB1x"])
            table = table[["EA1x", "EB1x", "SA1x", "SB1x", "CA1B1x", "CorrA1B1x"]]
            table.columns = [self._latex_names[c] for c in table.columns]

        return table

    def show_tuning_parameters(self):
        if self._data is None:
            raise DataNotYetAddedError("Please use the add_data method first.")

        print("Nadarawa-Watson bandwidth: " \
              "\t\tbw_nw \t\t= c_nw * n^(-alpha_nw) = {:1.2f}*{:1.0f}^{:1.3f} = {:1.3f}" \
              .format(self.cnw, self._n, self.anw, self.bw_nw))

        print("Local polynomial regression bandwidth: " \
              "\tbw_shocks \t= c_shocks * n^(-alpha_shocks) = {:1.2f}*{:1.0f}^{:1.2f} = {:1.3f}" \
              .format(self.csh, self._n, self.ash, self.bw_shocks))

        print("Censoring threshold (first moments): " \
              "\tt1 \t\t= c1_cens * n^(-alpha_cens_1) = {:1.2f}*{:1.0f}^{:1.2f} = {:1.3f}" \
              .format(self.c1cens, self._n, self.acens1, self.t1))

        print("Censoring threshold (second moments): " \
              "\tt2 \t\t= c2_cens * n^(-alpha_cens_2) = {:1.2f}*{:1.0f}^{:1.2f} = {:1.3f}" \
              .format(self.c2cens, self._n, self.acens2, self.t2))

    def bootstrap(self, x1=None, x2=None, n_iterations=100):
        if self._fitted is False:
            raise ModelNotYetFitError("Model must be fit first.")

        conditional = x1 is not None and x2 is not None

        bs_fhhps = eval(repr(self))
        data = self.data.copy()
        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in tqdm(range(n_iterations), desc="Running boostrap"):
                idx = np.random.randint(self._n, size=self._n)
                df = data.loc[idx]
                bs_fhhps.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
                if conditional:
                    bs_fhhps.conditional_fit(x1, x2)
                    s = bs_fhhps.conditional_estimates()
                else:
                    bs_fhhps.fit()
                    s = bs_fhhps.estimates()
                results.append(s)

        if conditional:
            self._conditional_bootstrapped_values = pd.concat(results, axis=0)
            return self._conditional_bootstrapped_values
        else:
            self._bootstrapped_values = pd.concat(results, axis=0)
            return self._bootstrapped_values

    def conditional_summary(self, latex_names=False):

        if self._conditional_bootstrapped_values is None:
            raise BootstrapNotYetRunError(
                "Please conditional_fit your model and bootstrap it first.")

        bs_tab = self._conditional_bootstrapped_values.describe([0.025, 0.5, 0.975])
        bs_tab.index = ["n_bootstrap", "Estimate", "Std. Error", "Min",
                        "Lower CI (2.5%)", "Median", "Upper CI (97.5%)", "Max"]

        est = self.conditional_estimates()
        with pd.option_context("mode.chained_assignment", None):
            shock_tab = bs_tab[self._shock_vars]
            rc_tab = bs_tab[self._conditional_rc_vars]
            shock_tab.loc["Estimate"] = [est[s].values for s in self._shock_vars]
            rc_tab.loc["Estimate"] = [est[s].values for s in self._conditional_rc_vars]

        if latex_names:
            shock_tab.columns = [self._latex_names[s] for s in shock_tab.columns]
            rc_tab.columns = [self._latex_names[s] for s in rc_tab.columns]

        return shock_tab, rc_tab

    def summary(self, latex_names=False):

        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first. ")

        bs_tab = self._bootstrapped_values.describe([0.025, 0.5, 0.975])
        bs_tab.index = ["n_bootstrap", "Estimate", "Std. Error", "Min",
                        "Lower CI (2.5%)", "Median", "Upper CI (97.5%)", "Max"]

        est = self.estimates()

        with pd.option_context("mode.chained_assignment", None):

            shock_tab = bs_tab[self._shock_vars]
            shock_tab.loc["Estimate"] = [est[s].values for s in self._shock_vars]

            rc1_tab = bs_tab[self._rc1_vars]
            rc2_tab = bs_tab[self._rc2_vars]

            rc1_tab.loc["Estimate"] = [est[s].values for s in self._rc1_vars]
            rc2_tab.loc["Estimate"] = [est[s].values for s in self._rc2_vars]

        if latex_names:
            shock_tab.columns = [self._latex_names[s] for s in shock_tab.columns]
            rc1_tab.columns = [self._latex_names[s] for s in rc1_tab.columns]
            rc2_tab.columns = [self._latex_names[s] for s in rc2_tab.columns]

        return shock_tab, rc1_tab, rc2_tab

    def plot_density(self, subplot_kwargs=None, density_kwargs=None):

        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")

        subplot_kwargs = subplot_kwargs or dict(figsize=(10, 10))
        density_kwargs = density_kwargs or dict(alpha=.8)

        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")

        shockfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        shockfig.subplots_adjust(hspace=.4)
        shockfig.suptitle("Bootstrapped shock moments", fontsize=16)
        shockaxs = []
        for s, ax in zip(self._shock_vars, axs.flatten()):
            ax = self._bootstrapped_values[s].plot.density(ax=ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize=14)
            ax.axvline(self._shocks[s], *ax.get_ylim(), linewidth=3, color="black")
            shockaxs.append(ax)

        handles = {"shock_figure": shockfig,
                   "shock_axes": shockaxs}

        for k, rc_vars in enumerate([self._rc1_vars, self._rc2_vars]):
            rcfig, axs = plt.subplots(3, 2, **subplot_kwargs)
            rcfig.subplots_adjust(hspace=.4)
            rcfig.suptitle("Bootstrapped random coefficient moments", fontsize=16)
            rcaxs = []
            for s, ax in zip(rc_vars, axs.flatten()):
                ax = self._bootstrapped_values[s].plot.density(ax=ax, **density_kwargs)
                ax.set_title(self._latex_names[s], fontsize=14)
                ax.axvline(self._ab_umoments[s], *ax.get_ylim(), linewidth=3, color="black")
                rcaxs.append(ax)
                if "Corr" in s:
                    xlim = ax.get_xlim()
                    ax.set_xlim(max(-1.5, xlim[0]), min(1.5, xlim[1]))

            handles.update({"rc{}_figure".format(k + 1): rcfig,
                            "rc{}_axes".format(k + 1): rcaxs})

        return handles

    def plot_conditional_density(self, subplot_kwargs=None, density_kwargs=None):

        if self._conditional_bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")

        subplot_kwargs = subplot_kwargs or dict(figsize=(10, 10))
        density_kwargs = density_kwargs or dict(alpha=.8)

        estimates = self.conditional_estimates()

        shockfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        shockfig.subplots_adjust(hspace=.4)
        shockfig.suptitle("Bootstrapped shock moments", fontsize=16)
        shockaxs = []
        for s, ax in zip(self._shock_vars, axs.flatten()):
            ax = self._conditional_bootstrapped_values[s].plot.density(ax=ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize=14)
            ax.axvline(estimates[s].values, *ax.get_ylim(), linewidth=3, color="black")
            shockaxs.append(ax)

        rcfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        rcfig.subplots_adjust(hspace=.4)
        rcfig.suptitle("Bootstrapped conditional random coefficient moments", fontsize=16)
        rcaxs = []

        for s, ax in zip(self._conditional_rc_vars, axs.flatten()):
            ax = self._conditional_bootstrapped_values[s].plot.density(ax=ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize=14)
            ax.axvline(estimates[s].values, *ax.get_ylim(), linewidth=3, color="black")
            rcaxs.append(ax)
            if "Corr" in s:
                xlim = ax.get_xlim()
                ax.set_xlim(max(-1.5, xlim[0]), min(1.5, xlim[1]))

        return (shockfig, shockaxs), (rcfig, rcaxs)

    def censored_summary(self):

        if self._data is None:
            raise DataNotYetAddedError("Please use the add_data method first.")

        df = self.data
        absDX = np.abs(df["X2"] - df["X1"])
        idx_cens1 = absDX <= self.t1
        idx_cens2 = absDX <= self.t2
        qs = [.025, 0.25, 0.5, 0.75, 0.975]

        disc1_description = df.loc[idx_cens1, ["Y1", "Y2"]].describe(qs)
        kept1_description = df.loc[~idx_cens1, ["Y1", "Y2"]].describe(qs)
        disc2_description = df.loc[idx_cens2, ["Y1", "Y2"]].describe(qs)
        kept2_description = df.loc[~idx_cens2, ["Y1", "Y2"]].describe(qs)

        table = pd.concat([disc1_description,
                           kept1_description,
                           disc2_description,
                           kept2_description], axis=1)
        table.columns = ["Discarded $Y_1$ (1)",
                         "Discarded $Y_2$ (1)",
                         "Kept $Y_1$ (1)",
                         "Kept $Y_2$ (1)",
                         "Discarded $Y_1$ (2)",
                         "Discarded $Y_2$ (2)",
                         "Kept $Y_1$ (2)",
                         "Kept $Y_2$ (2)"]

        table.loc["percent"] = table.loc["count"] / df.shape[0]

        table = table.loc[['count', 'percent', 'mean', 'std',
                           'min', '2.5%', '25%', '50%', '75%', '97.5%', 'max']]

        return table


if __name__ == "__main__":

    import argparse
    from time import time
    from os import listdir

    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-f', '--filename',
                        help='Comma-separated file containing data',
                        required=True)

    parser.add_argument('-X1', '--X1-column',
                        help='Column name or number corresponding to variable X1 (default: first column)',
                        required=False)
    parser.add_argument('-X2', '--X2-column',
                        help='Column name or number corresponding to variable X2 (default: second column)',
                        required=False)
    parser.add_argument('-Y1',
                        '--Y1-column',
                        help='Column name or number corresponding to variable Y1 (default: third column)',
                        required=False)
    parser.add_argument('-Y2', '--Y2-column',
                        help='Column name or number corresponding to variable Y2 (default: fourth column)',
                        required=False)

    parser.add_argument('-x1',
                        '--x1_point',
                        help="X1 point on which to condition.",
                        required=False,
                        type=float)
    parser.add_argument('-x2',
                        '--x2_point',
                        help="X2 point on which to condition.",
                        required=False,
                        type=float)

    parser.add_argument('-anw', '--alpha_nw',
                        help='Parameter alpha_nw',
                        required=False,
                        default=0.167,
                        type=float)
    parser.add_argument('-cnw', '--c_nw',
                        help='Parameter c_nw',
                        required=False,
                        default=0.5,
                        type=float)
    parser.add_argument('-ash', '--alpha_shocks',
                        help='Parameter alpha_nw',
                        required=False,
                        default=0.2,
                        type=float)
    parser.add_argument('-csh', '--c_shocks',
                        help='Parameter c_shocks',
                        required=False,
                        default=4.0,
                        type=float)

    parser.add_argument('-cc1', '--c1_cens',
                        help='Coefficient of threshold bandwidth (first moments).',
                        required=False,
                        default=1,
                        type=float)
    parser.add_argument('-cc2', '--c2_cens',
                        help='Coefficient of threshold bandwidth (second moments).',
                        required=False,
                        default=1,
                        type=float)
    parser.add_argument('-ac1', '--alpha_cens_1',
                        help='Exponent of threshold bandwidth (first moments).',
                        required=False,
                        default=0.24,
                        type=float)
    parser.add_argument('-ac2', '--alpha_cens_2',
                        help='Exponent of threshold bandwidth (second moments).',
                        required=False,
                        default=0.12,
                        type=float)
    parser.add_argument('-pl', '--poly_order',
                        help='Polynomial order to use when estimating sec_shocks.',
                        required=False,
                        default=2,
                        type=int)

    parser.add_argument('-b', '--bootstrap_iterations',
                        help="Number of times to bootstrap",
                        required=False,
                        default=100,
                        type=int)

    parser.add_argument('-suffix', '--output_file_suffix',
                        help="Suffix to add to output file names. Default is a timestamp.",
                        required=False,
                        default="",
                        type=str)

    parser.add_argument('-tab', '--output_table_type',
                        help="File type of output table.",
                        required=False,
                        default="latex",
                        choices=["latex", "csv"],
                        type=str)

    args = parser.parse_args()

    suffix = args.output_file_suffix or str(int(time() * 100))

    print("\n#### FHHPS #####\n")
    columns = ["Y1" or args.Y1_column,
               "Y2" or args.Y2_column,
               "X1" or args.X1_column,
               "X2" or args.X2_column]

    print("Reading csv file.")
    df = pd.read_csv(args.filename,
                     usecols=columns,
                     header=0).dropna()
    df.columns = ["X1", "X2", "Y1", "Y2"]

    print("Read file. First rows look like this.")
    print(df.head())

    print("\nInitializing FHHPS.")
    algo = FHHPS(c_shocks=args.c_shocks,
                 c_nw=args.c_nw,
                 c1_cens=args.c1_cens,
                 c2_cens=args.c2_cens,
                 alpha_shocks=args.alpha_shocks,
                 alpha_nw=args.alpha_nw,
                 alpha_cens_1=args.alpha_cens_1,
                 alpha_cens_2=args.alpha_cens_2,
                 poly_order=args.poly_order)

    print(algo)

    print("\nAdding data.")
    algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])

    print("\nComputed the following additional tuning parameters:")
    algo.show_tuning_parameters()

    if args.x1_point is not None and args.x2_point is not None:
        # FIT
        print("\n\nEstimating CONDITIONAL moments...", end="")
        algo.conditional_fit(args.x1_point, args.x2_point)
        print("OK!\n\n")

        # BOOTSTRAP
        algo.bootstrap(args.x1_point, args.x2_point, n_iterations=args.bootstrap_iterations)
        print("\nSaving bootstrapped values to CSV file...", end="")
        algo._conditional_bootstrapped_values.to_csv(
            "conditional_bootstrapped_values_{}.csv".format(suffix))
        print("OK!")

        # FIGURES
        print("\n\nProducing figures...", end="")
        (shockfig, _), (rcfig, _) = algo.plot_conditional_density()
        shockfig.savefig("bootstrap_shocks_{}.pdf".format(suffix))
        rcfig.savefig("conditional_bootstrap_random_coefficients_{}.pdf".format(suffix))
        print("OK!")

        # TABLES
        print("\n\nProducing summary tables...", end="")
        uselatex = args.output_table_type == "latex"
        shock_tab, rc_tab = algo.conditional_summary(latex_names=uselatex)
        if uselatex:
            shock_tab.to_latex("bootstrap_shock_{}.tex".format(suffix))
            rc_tab.to_latex("conditional_bootstrap_random_coefficients_{}.tex".format(suffix))
        else:
            shock_tab.to_csv("_bootstrap_shock_{}.csv".format(suffix))
            rc_tab.to_csv("conditional_bootstrap_random_coefficients_{}.tex".format(suffix))



    else:
        # FIT
        print("\nEstimating...", end="")
        algo.fit()

        # BOOTSTRAP    
        algo.bootstrap(n_iterations=args.bootstrap_iterations)
        print("\nSaving bootstrapped values to CSV file...", end="")
        algo._bootstrapped_values.to_csv("bootstrapped_values_{}.csv".format(suffix))

        # FIGURES
        print("\n\nProducing figures and summary tables.", end="")
        handles = algo.plot_density()
        handles["shock_figure"].savefig("bootstrap_shocks_{}.pdf".format(suffix))
        handles["rc1_figure"].savefig("bootstrap_random_coefficients_1_{}.pdf".format(suffix))
        handles["rc2_figure"].savefig("bootstrap_random_coefficients_2_{}.pdf".format(suffix))

        # TABLES
        uselatex = args.output_table_type == "latex"

        shock_tab, rc1_tab, rc2_tab = algo.summary(latex_names=uselatex)
        cens_tab = algo.censored_summary()
        condrc_tab = algo.conditional_random_coefficient_moments()

        if uselatex:
            shock_tab.to_latex("bootstrap_shock_{}.tex".format(suffix), escape=False)
            rc1_tab.to_latex("bootstrap_random_coefficients_1_{}.tex".format(suffix), escape=False)
            rc2_tab.to_latex("bootstrap_random_coefficients_2_{}.tex".format(suffix), escape=False)
            condrc_tab.to_latex("conditional_random_coefficients_{}.tex".format(suffix),
                                escape=False)
        else:
            shock_tab.to_csv("bootstrap_shock_{}.csv".format(suffix))
            rc1_tab.to_csv("bootstrap_random_coefficients_1_{}.tex".format(suffix))
            rc2_tab.to_csv("bootstrap_random_coefficients_2_{}.tex".format(suffix))
            condrc_tab.to_csv("conditional_random_coefficients_{}.csv".format(suffix), escape=False)

    if uselatex:
        cens_tab.to_latex("censored_statistics_{}.tex".format(suffix), escape=False)
    else:
        cens_tab.to_csv("censored_statistics_{}.csv".format(suffix))

    print("OK!")
    print("\n\nDone. You should see these new files:")
    for f in listdir():
        if suffix in f:
            print(f)
    print("\n\n")
