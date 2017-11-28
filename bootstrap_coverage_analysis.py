#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:55:51 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("bs_coverage/bs_coverage_20000.csv")

truth = {"EA1": 2,
           "EB1": .4,
           "SA1": 3,
           "SB1": np.sqrt(.4),
           "CA1B1": .5*np.sqrt(.4)*3,
           "CorrA1B1": .5,
           "EU": .3,
           "EV": .1,
           "SU": 1,
           "SV": np.sqrt(.1),
           "CUV": np.sqrt(.1)*1*.5,
           "CorrUV": .5,
           "EA2": 2 + .3,
           "EB2": .4 + .1,
           "SA2": np.sqrt(3**2 + 1), # sqrt(VA + VU)  
           "SB2": np.sqrt(.4 +  .1), # sqrt(VB + VV)
           "CA2B2": .5*np.sqrt(.4)*3 + np.sqrt(.1)*1*.5 # CAB + CUV
           }

truth["CorrA2B2"] = truth["CA2B2"]/(truth["SA2"]*truth["SB2"])



# 95% Coverage
coverage = defaultdict(lambda: 0)
for v in truth:
    q05_name = v + "_25"
    q95_name = v + "_975"
    within = (df[q05_name] <= truth[v]) & (truth[v] <= df[q95_name])
    coverage[v] = np.mean(within)
    

rc1 = pd.Series(coverage)[["EA1", "EB1", "SA1", "SB1", "CA1B1", "CorrA1B1"]]
rc1.index = ["$E[A_1]$","$E[B_1]$","$Std[A_1]$","$Std[B_1]$","$Cov[A_1, B_1]$","$Corr[A_1, B_1]$"]
pd.DataFrame(rc1).T.round(3).to_latex(
    "bs_coverage/bs_coverage_table_random_coefficients_1.tex",
    escape = False)#


rc2 = pd.Series(coverage)[["EA2", "EB2", "SA2", "SB2", "CA2B2", "CorrA2B2"]]
rc2.index = ["$E[A_2]$","$E[B_2]$","$Std[A_2]$","$Std[B_2]$","$Cov[A_2, B_2]$","$Corr[A_2, B_2]$"]
pd.DataFrame(rc2).T.round(3).to_latex(
    "bs_coverage/bs_coverage_table_random_coefficients_2.tex",
    escape = False)#


s = pd.Series(coverage)[["EU", "EV", "SU", "SV", "CUV", "CorrUV"]]
s.index = ["$E[U_2]$","$E[V_2]$","$Std[U_2]$","$Std[V_2]$","$Cov[U_2, V_2]$","$Corr[U_2, V_2]$"]
pd.DataFrame(s).T.round(3).to_latex(
        "bs_coverage/bs_coverage_table_shocks.tex",
        escape = False)
