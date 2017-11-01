#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:03:26 2017

@author: vitorhadad
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from warnings import warn
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings



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
        return np.where(np.abs(u) < h, 0.75*(1-(u/h)**2), 0)
     
def uniform(u, h):
    return 1/(h/2) * np.where(np.abs(u) < h/2, 1, 0) 

def gaussian(u, h):
    return norm(scale = h).pdf(u)
   

def g1inv(x1, x2):
    if x2 != x1:
        return 1/(x2 - x1) * np.array([[x2, -x1],
                                       [-1,   1]])
    else:
        warn("Found value X1 = X2 up to machine precision. "\
             "Conditional moments will likely be NaNs for this value.", 
             RuntimeWarning)
        return np.full((2,2), np.nan)


def g2inv(x1, x2): 
    if x2 != x1:
        return 1/(x2 - x1)**2 * \
                np.array([[x2**2, x1**2, -2*x1*x2],
                          [    1,     1,       -2],
                          [  -x2,   -x1,  x1 + x2]])
    else:
        warn("Found value X1 = X2 up to machine precision. "\
             "Conditional moments will likely be NaNs for this value.", 
             RuntimeWarning)
        return np.full((3,3), np.nan)


class FHHPS:
    
    def __init__(self,
                c_shocks = 1,
                c_nw    = 0.1, 
                c1_cens  = 1,
                c2_cens = 1, 
                alpha_shocks = 0.21,
                alpha_nw     = 1/2, 
                alpha_cens_1 = 0.24,
                alpha_cens_2 = 0.12,
                kernel = "epanechnikov",
                poly_order = 2):
                
        
        self.csh     = c_shocks
        self.cnw     = c_nw
        self.c1cens  = c1_cens
        self.c2cens  = c2_cens
        self.ash     = alpha_shocks
        self.anw     = alpha_nw
        self.acens1 = alpha_cens_1
        self.acens2 = alpha_cens_2
        self.poly_order = poly_order
        self._data = None
        self._n = None
        self._fitted = False
        self._conditional = None
        self._shock_vars = ["EU", "EV", "SU", "SV", "CUV", "CorrUV"]
        self._rc_vars = ["EA", "EB", "SA", "SB", "CAB", "CorrAB"]
        self._conditional_rc_vars = ["EAx", "EBx", "SAx", "SBx", "CABx", "CorrABx"]
        self._bootstrapped_values = None
        self._conditional_bootstrapped_values = None
        self.kernel = epanechnikov
        self._latex_names = {"EU": "$E[U_2]$",
                             "EV": "$E[V_2]$",
                             "SU": "$Std[U_2]$",
                             "SV": "$Std[V_2]$",
                             "CUV": "$Cov[U_2, V_2]$",
                             "CorrUV": "$Corr[U_2, V_2]$",
                             "EA": "$E[A_1]$",
                             "EB": "$E[B_1]$",
                             "SA": "$Std[A_1]$",
                             "SB": "$Std[B_1]$",
                             "CAB": "$Cov[A_1, B_1]$",
                             "CorrAB": "$Corr[A_1, B_1]$",
                             "EAx": "$E[A_1|X]$",
                             "EBx": "$E[B_1|X]$",
                             "SAx": "$Std[A_1|X]$",
                             "SBx": "$Std[B_1|X]$",
                             "CABx": "$Cov[A_1, B_1|X]$",
                             "CorrABx": "$Corr[A_1, B_1|X]$"}
        

        
        
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
                raise ValueError("Could not understand input data type."\
                                 "Each one of X1,X2,Y1,Y2 must be:"\
                                 "a pandas Series, a numpy array, or a list.")
            
        self._data = dict(zip(["X1","X2","Y1","Y2"], data))
        return self._data
    
    
    def _setup_tuning_parameters(self):
        # Set up tuning parameters
        sx             = np.std(self._data["X2"])
        self.bw_nw     = self.cnw     * sx * self._n ** (-self.anw) 
        self.bw_shocks = self.csh          * self._n ** (-self.ash)
        self.t1        = self.c1cens  * sx * self._n ** (-self.acens1)
        self.t2        = self.c2cens  * sx * self._n ** (-self.acens2)
    
        
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
        tab = pd.DataFrame(index = [self._n],
                            data = res)
        return tab
    
    
    
    def conditional_estimates(self):
        #if self._ab_first_cmoms is None:
        #    raise ModelNotYetFitError("Please use the conditional_fit method first."
        
        EAx = self._ab_cmoms["EAx"][0,0]
        EBx = self._ab_cmoms["EBx"][0,0]
        EAsqx = self._ab_cmoms["EAsqx"][0,0]
        EBsqx = self._ab_cmoms["EBsqx"][0,0]
        EABx = self._ab_cmoms["EABx"][0,0]
        
        out = pd.DataFrame(index = [self._n],
                           data = {"EU": self._shocks["EU"],
                                   "EV": self._shocks["EV"],
                                   "SU": self._shocks["SU"],
                                   "SV": self._shocks["SV"],
                                   "CUV": self._shocks["CUV"],
                                   "CorrUV": self._shocks["CorrUV"],
                                   "EAx": EAx,
                                   "EBx": EBx,
                                   "SAx": np.sqrt(EAsqx - EAx**2),
                                   "SBx": np.sqrt(EBsqx - EBx**2),
                                   "CABx": EABx - EAx*EBx})
            
        out["CorrABx"] = out["CABx"]/(out["SAx"]*out["SBx"])
        return out
    
   
    

    def estimate_shocks(self):
        
        X1   = self._data["X1"]
        X2   = self._data["X2"]
        DX2  = X2 - X1
        DY   = (self._data["Y2"] - self._data["Y1"]).flatten()
        
        poly = PolynomialFeatures(degree = self.poly_order,
                                  include_bias = False)
        #XX1 = poly.fit_transform(np.hstack([X2])) #DX2.reshape(-1, 1)]))
        #XX2 = poly.fit_transform(np.hstack([2*X2, X2**2]))#, DX2.reshape(-1, 1)**2]))
        XX1 = poly.fit_transform(np.hstack([X2, DX2.reshape(-1, 1)]))
        XX2 = poly.fit_transform(np.hstack([2*X2, X2**2, DX2.reshape(-1, 1)**2]))
        
        w    = self.kernel(DX2, self.bw_shocks).flatten()
        
        ols1 = LinearRegression()
        ols1.fit(XX1, DY, w)
        EU = ols1.intercept_
        EV = ols1.coef_[0]
        
        ols2 = LinearRegression()
        DYsq = DY**2
        ols2.fit(XX2, DYsq, w)
        EUsq = ols2.intercept_
        EUV, EVsq = ols2.coef_.flatten()[:2]
        VU, VV, CUV = EUsq - EU**2, EVsq - EV**2, EUV  - EU*EV
        CorrUV = CUV/(np.sqrt(VU)*np.sqrt(VV))
        
        if VU < 0 or VV < 0:
            warn("\nShocks variances are negative."\
                 "\nConsider changing the parameters.", 
                 RuntimeWarning)
        if CorrUV < -1 or CorrUV > 1:
            warn("\nShocks correlation is not within [-1,1]"\
                 "\nConsider changing the parameters.", 
                RuntimeWarning)
            
        self._shocks = {"EU":   EU, 
                        "EV":   EV,
                        "EUsq": EUsq, 
                        "EUV":  EUV, 
                        "EVsq": EVsq,
                        "VU":   VU,
                        "VV":   VV,
                        "CUV":  CUV,
                        "SU": np.sqrt(VU),
                        "SV": np.sqrt(VV),
                        "CorrUV": CorrUV}
    
        return self._shocks



    def get_ycmoms(self, x1 = None, x2 = None):
    
        K = pairwise_kernels(
                X = np.hstack([self._data["X1"], self._data["X2"]]),
                Y = np.array([[x1, x2]]) if x1 is not None else None,
                metric = "rbf", 
                gamma = 1/self.bw_nw)
                
        denom = np.sum(K, axis = 0)
        
        def fitter(y):
            num = np.sum(K*y, axis = 0)
            return (num/denom).reshape(-1, 1)
        
        self._ycmoms = {"EY1":   fitter(self._data["Y1"]),
                        "EY2":   fitter(self._data["Y2"]),
                        "EY1sq": fitter(self._data["Y1"]**2),
                        "EY2sq": fitter(self._data["Y2"]**2),
                        "EY1Y2": fitter(self._data["Y1"]*self._data["Y2"])}
        
        return self._ycmoms




    def get_ab_first_cmoms(self, x1 = None, x2 = None):
        
        EY1 = self._ycmoms["EY1"]
        EY2 = self._ycmoms["EY2"]
        EU = self._shocks["EU"]
        EV = self._shocks["EV"]
        n  = self._n if x1 is None else 1
        X1 = np.array([x1]) if x1 is not None else self._data["X1"]
        X2 = np.array([x2]) if x2 is not None else self._data["X2"]
        
        EY = np.hstack([EY1, EY2-EU-EV*X2])

        mab = np.full((2, n, 1), np.nan)
            
        for k, (x1, x2, ey) in enumerate(zip(X1.ravel(), 
                                             X2.ravel(),
                                             EY)):
            mab[:, k] = g1inv(x1, x2) @ ey.reshape(-1, 1)
                        
        
        self._ab_cmoms = {"EAx": mab[0,:],
                          "EBx": mab[1,:]}            
        
        return self._ab_cmoms 
    
    
    
    
    def get_ab_second_cmoms(self, x1 = None, x2 = None):
            
        EU = self._shocks["EU"]
        EV = self._shocks["EV"]
        EUsq = self._shocks["EUsq"]
        EVsq = self._shocks["EVsq"]
        EUV  = self._shocks["EUV"]
        X1 = np.array([x1]) if x1 is not None else self._data["X1"]
        X2 = np.array([x2]) if x2 is not None else self._data["X2"]
        EA1 = self._ab_cmoms["EAx"]
        EB1 = self._ab_cmoms["EBx"]
        EY1sq = self._ycmoms["EY1sq"]
        EY2sq = self._ycmoms["EY2sq"]
        EY1Y2 = self._ycmoms["EY1Y2"]
        n  = self._n if x1 is None else 1
        
        CY = np.hstack([
            EY1sq, 
            EY2sq - (EUsq + EUV*2*X2 + EVsq*X2**2) - 2*(EA1 + EB1*X2)*(EU + EV*X2),
            EY1Y2 - (EA1 + EB1*X1) * (EU + EV*X2)
        ])
    
        sab = np.full((3, n, 1), np.nan)
            
        for k, (x1, x2, cy) in enumerate(zip(X1.ravel(), 
                                             X2.ravel(),
                                             CY)):
            sab[:, k] = g2inv(x1, x2) @ cy.reshape(-1, 1)
        

        self._ab_cmoms.update({"EAsqx": sab[0],
                               "EBsqx": sab[1],
                                "EABx": sab[2]})
    
        
        
        return self._ab_cmoms

    
       

    def get_ab_umoments(self):
        
        absDX  = np.abs(self._data["X2"] - self._data["X1"])
        valid1 = absDX > self.t1
        valid2 = absDX > self.t2
        
        mom =  {"EA":   np.mean(self._ab_cmoms["EAx"][valid1]),
                "EB":   np.mean(self._ab_cmoms["EBx"][valid1]),
                "EAsq": np.mean(self._ab_cmoms["EAsqx"][valid2]),
                "EBsq": np.mean(self._ab_cmoms["EBsqx"][valid2]),
                "EAB":  np.mean(self._ab_cmoms["EABx"][valid2])}
    
        mom.update({"VA":  mom["EAsq"] - mom["EA"]**2,
                    "VB":  mom["EBsq"] - mom["EB"]**2,
                    "CAB": mom["EAB"]  - mom["EA"]*mom["EB"]})
        
        mom.update({"SA": np.sqrt(mom["VA"]),
                    "SB": np.sqrt(mom["VB"])})
        
        mom.update({"CorrAB": mom["CAB"] / (mom["SA"]*mom["SB"])})
        
        self._ab_umoments = mom
        
        return self._ab_umoments




    def show_tuning_parameters(self):
        if self._data is None:
            raise DataNotYetAddedError("Please use the add_data method first.")
        
        s = "\nbw_nw: {:1.5f}".format(self.bw_nw) + \
            "\nbw_shocks: {:1.5f}".format(self.bw_shocks) + \
            "\nt1: {:1.5f}".format(self.t1) + \
            "\nt2: {:1.5f}".format(self.t2)
        print(s)
    
    
    
    
    def bootstrap(self, x1 = None, x2 = None, n_iterations = 100):
        if self._fitted is False:
            raise ModelNotYetFitError("Model must be fit first.")

        conditional = x1 is not None and x2 is not None

        bs_fhhps = eval(repr(self))
        data = self.data.copy()
        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in tqdm(range(n_iterations), desc = "Running boostrap"):
                idx = np.random.randint(self._n, size = self._n)
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
            self._conditional_bootstrapped_values = pd.concat(results, axis = 0)
            return self._conditional_bootstrapped_values
        else:
            self._bootstrapped_values = pd.concat(results, axis = 0)
            return self._bootstrapped_values
    
    
    
    
    
    def conditional_summary(self, latex_names = False):
        
        if self._conditional_bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please conditional_fit your model and bootstrap it first.")
        
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
    
    
    
    def summary(self, latex_names = False):
        
        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first. ")
        
        bs_tab = self._bootstrapped_values.describe([0.025, 0.5, 0.975])
        bs_tab.index = ["n_bootstrap", "Estimate", "Std. Error", "Min", 
                        "Lower CI (2.5%)", "Median", "Upper CI (97.5%)", "Max"]
        
        est = self.estimates()
        with pd.option_context("mode.chained_assignment", None):
            shock_tab = bs_tab[self._shock_vars]
            rc_tab = bs_tab[self._rc_vars]
            shock_tab.loc["Estimate"] = [est[s].values for s in self._shock_vars]
            rc_tab.loc["Estimate"] = [est[s].values for s in self._rc_vars]
        
        
        if latex_names:
            shock_tab.columns = [self._latex_names[s] for s in shock_tab.columns]
            rc_tab.columns = [self._latex_names[s] for s in rc_tab.columns]

        return shock_tab, rc_tab
        
    
    
        
        
    def plot_density(self, subplot_kwargs = None, density_kwargs = None):
        
        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")
        
        subplot_kwargs = subplot_kwargs or dict(figsize = (10, 10))
        density_kwargs = density_kwargs or dict(alpha = .8)
        
        if self._bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")
        
        shockfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        shockfig.subplots_adjust(hspace = .4)
        shockfig.suptitle("Bootstrapped shock moments", fontsize = 16)
        shockaxs = []
        for s, ax in zip(self._shock_vars, axs.flatten()):
            ax = self._bootstrapped_values[s].plot.density(ax = ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize = 14)
            ax.axvline(self._shocks[s], *ax.get_ylim(), linewidth = 3, color = "black")
            shockaxs.append(ax)
            
        rcfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        rcfig.subplots_adjust(hspace = .4)
        rcfig.suptitle("Bootstrapped random coefficient moments", fontsize = 16)
        rcaxs = []
        for s, ax in zip(self._rc_vars, axs.flatten()):
            ax = self._bootstrapped_values[s].plot.density(ax = ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize = 14)
            ax.axvline(self._ab_umoments[s], *ax.get_ylim(), linewidth = 3, color = "black")
            rcaxs.append(ax)
            if "Corr" in s:
                xlim = ax.get_xlim()
                ax.set_xlim(max(-1.5, xlim[0]), min(1.5, xlim[1]))
            
        return (shockfig, shockaxs), (rcfig, rcaxs)
            
    
    
    def plot_conditional_density(self, subplot_kwargs = None, density_kwargs = None):
        
        if self._conditional_bootstrapped_values is None:
            raise BootstrapNotYetRunError("Please fit your model and bootstrap it first.")
        
        subplot_kwargs = subplot_kwargs or dict(figsize = (10, 10))
        density_kwargs = density_kwargs or dict(alpha = .8)
        
        estimates = self.conditional_estimates()
        
        shockfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        shockfig.subplots_adjust(hspace = .4)
        shockfig.suptitle("Bootstrapped shock moments", fontsize = 16)
        shockaxs = []
        for s, ax in zip(self._shock_vars, axs.flatten()):
            ax = self._conditional_bootstrapped_values[s].plot.density(ax = ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize = 14)
            ax.axvline(estimates[s].values, *ax.get_ylim(),  linewidth = 3, color = "black")
            shockaxs.append(ax)
            
            
        rcfig, axs = plt.subplots(3, 2, **subplot_kwargs)
        rcfig.subplots_adjust(hspace = .4)
        rcfig.suptitle("Bootstrapped conditional random coefficient moments", fontsize = 16)
        rcaxs = []
        
        for s, ax in zip(self._conditional_rc_vars, axs.flatten()):
            ax = self._conditional_bootstrapped_values[s].plot.density(ax = ax, **density_kwargs)
            ax.set_title(self._latex_names[s], fontsize = 14)
            ax.axvline(estimates[s].values, *ax.get_ylim(), linewidth = 3, color = "black")
            rcaxs.append(ax)
            if "Corr" in s:
                xlim = ax.get_xlim()
                ax.set_xlim(max(-1.5, xlim[0]), min(1.5, xlim[1]))
            
        return (shockfig, shockaxs), (rcfig, rcaxs)
                 
            
        
        
    
    



if __name__ == "__main__":
    
    import argparse
    from time import time
    from os import listdir
    
    parser = argparse.ArgumentParser(description='Description of your program')
    
    parser.add_argument('-f','--filename', 
                        help='Comma-separated file containing data',
                        required=True)
    
    parser.add_argument('-X1','--X1-column', 
                        help='Column name or number corresponding to variable X1 (default: first column)', 
                        required=False)
    parser.add_argument('-X2','--X2-column', 
                        help='Column name or number corresponding to variable X2 (default: second column)',
                        required=False)
    parser.add_argument('-Y1',
                        '--Y1-column',
                        help='Column name or number corresponding to variable Y1 (default: third column)',
                        required=False)
    parser.add_argument('-Y2','--Y2-column',
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

    
    parser.add_argument('-anw','--alpha_nw',
                        help='Parameter alpha_nw',
                        required=False,
                        default = 0.167,
                        type=float)
    parser.add_argument('-cnw','--c_nw',
                        help='Parameter c_nw',
                        required=False,
                        default = 0.5,
                        type=float)
    parser.add_argument('-ash','--alpha_shocks',
                        help='Parameter alpha_nw',
                        required=False,
                        default = 0.2,
                        type=float)
    parser.add_argument('-csh','--c_shocks',
                        help='Parameter c_shocks',
                        required=False,
                        default = 3.0,
                        type=float)
#    parser.add_argument('-kernel','--kernel',
#                        help='Kernel to use in Nadaraya-Watson step.',
#                        required=False,
#                        default="epa",
#                        type=str,
#                        choices = ["epa", "gaussian", "uniform"])
    parser.add_argument('-cc1','--c1_cens',
                        help='Coefficient of threshold bandwidth (first moments).',
                        required=False,
                        default=1,
                        type=float)
    parser.add_argument('-cc2','--c2_cens',
                        help='Coefficient of threshold bandwidth (second moments).',
                        required=False,
                        default=1,
                        type=float)
    parser.add_argument('-ac1','--alpha_cens_1',
                        help='Exponent of threshold bandwidth (first moments).',
                        required=False,
                        default=0.24,
                        type=float)
    parser.add_argument('-ac2','--alpha_cens_2',
                        help='Exponent of threshold bandwidth (second moments).',
                        required=False,
                        default=0.12,
                        type=float)
    parser.add_argument('-pl','--poly_order',
                        help='Polynomial order to use when estimating shocks.',
                        required=False,
                        default=2,
                        type=int)
    
    parser.add_argument('-b', '--bootstrap_iterations',
                       help="Number of times to bootstrap",
                       required=False,
                       default=100,
                       type=int)
    
    parser.add_argument('-outs', '--output_file_suffix',
                       help="Suffix to add to output file names. Default is a timestamp.",
                       required=False,
                       default="",
                       type=str)
    
    parser.add_argument('-tab', '--output_table_type',
                       help="File type of output table.",
                       required=False,
                       default="latex",
                       choices = ["latex", "csv"],
                       type=str)
    
    
    
    args = parser.parse_args()
    
    suffix = args.output_file_suffix or str(int(time()*100))
    
    print("\n\n#### FHHPS #####\n\n")
    columns = ["Y1" or args.Y1_column,
               "Y2" or args.Y2_column,
               "X1" or args.X1_column,
               "X2" or args.X2_column]
    
    print("\nReading csv file...", end = "")
    df = pd.read_csv(args.filename,
                     usecols = columns,
                     header = 0).dropna()  
    df.columns = ["X1", "X2", "Y1", "Y2"]
    
    print("OK!\nRead file. First rows look like this.")
    print(df.head())
    
    
    print("\n\nInitializing FHHPS... OK!")
    algo = FHHPS(c_shocks = args.c_shocks,
                c_nw    = args.c_nw, 
                c1_cens  = args.c1_cens,
                c2_cens = args.c2_cens, 
                alpha_shocks = args.alpha_shocks,
                alpha_nw     = args.alpha_nw, 
                alpha_cens_1 = args.alpha_cens_1,
                alpha_cens_2 = args.alpha_cens_2,
                #kernel = args.kernel,
                poly_order = args.poly_order)

    print(algo)
    
    print("\n\nAdding data...", end = "")
    algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
    
    print("OK! Computed these tuning parameters.")
    algo.show_tuning_parameters()
    
        
    
    if args.x1_point is not None and args.x2_point is not None:
        # FIT
        print("\n\nEstimating CONDITIONAL moments...", end="")
        algo.conditional_fit(args.x1_point, args.x2_point)
        print("OK!\n\n")
        
        # BOOTSTRAP
        algo.bootstrap(args.x1_point, args.x2_point, n_iterations = args.bootstrap_iterations)
        print("\nSaving bootstrapped values to CSV file...", end = "")
        algo._conditional_bootstrapped_values.to_csv("conditional_bootstrapped_values_{}.csv".format(suffix))
        print("OK!")
        
        # FIGURES
        print("\n\nProducing figures...", end = "")
        (shockfig, _), (rcfig, _) = algo.plot_conditional_density()
        shockfig.savefig("bootstrap_shocks_{}.pdf".format(suffix))
        rcfig.savefig("conditional_bootstrap_random_coefficients_{}.pdf".format(suffix))    
        print("OK!")
        
        # TABLES
        print("\n\nProducing summary tables...", end = "")
        uselatex = args.output_table_type == "latex"
        shock_tab, rc_tab = algo.conditional_summary(latex_names = uselatex)
        if uselatex:
            shock_tab.to_latex("bootstrap_shock_{}.tex".format(suffix))
            rc_tab.to_latex("conditional_bootstrap_random_coefficients_{}.tex".format(suffix))
        else:
            shock_tab.to_csv("_bootstrap_shock_{}.csv".format(suffix))
            rc_tab.to_csv("conditional_bootstrap_random_coefficients_{}.tex".format(suffix))
        print("OK!")
        print("\n\nDone. You should see these five new files:")
        for f in listdir():
            if suffix in f:
                print(f)
        print("\n\n")
        


    else: 
        # FIT
        print("\n\nEstimating...", end="")
        algo.fit()
        print("OK!\n\n")
    
        # BOOTSTRAP    
        algo.bootstrap(n_iterations = args.bootstrap_iterations)
        print("\nSaving bootstrapped values to CSV file...", end = "")
        algo._bootstrapped_values.to_csv("bootstrapped_values_{}.csv".format(suffix))
        print("OK!")
        
        # FIGURES
        print("\n\nProducing figures...", end = "")
        (shockfig, _), (rcfig, _) = algo.plot_density()
        shockfig.savefig("bootstrap_shocks_{}.pdf".format(suffix))
        rcfig.savefig("bootstrap_random_coefficients_{}.pdf".format(suffix))    
        print("OK!")
        
        # TABLES
        print("\n\nProducing summary tables...", end = "")
        uselatex = args.output_table_type == "latex"
        
        shock_tab, rc_tab = algo.summary(latex_names = uselatex)
        if uselatex:
            shock_tab.to_latex("bootstrap_shock_{}.tex".format(suffix))
            rc_tab.to_latex("bootstrap_random_coefficients_{}.tex".format(suffix))
        else:
            shock_tab.to_csv("bootstrap_shock_{}.csv".format(suffix))
            rc_tab.to_csv("bootstrap_random_coefficients_{}.tex".format(suffix))
        print("OK!")
        print("\n\nDone. You should see these five new files:")
        for f in listdir():
            if suffix in f:
                print(f)
        print("\n\n")
        
    
    
    
    
    
    
    
    
    
    
    
