#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:25:42 2017

@author: vitorhadad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:55:51 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
from collections import defaultdict

df = pd.read_csv("bs_coverage/bs_conditional_coverage_20000.csv")
x_points = df[["x1","x2"]].drop_duplicates().values.tolist()

SigmaAB = np.array([[     9,   0.9487],
                    [0.9487,       .4]])

SigmaABX = np.array([[1.5, 1.5], 
                     [.32, .32]])

SigmaXXinv = inv(np.array([[1, .5], 
                           [.5, 1]]))

muAB = np.array([[ 2], 
                 [.4]])

SigmaABx = SigmaAB - SigmaABX @ SigmaXXinv @ SigmaABX.T 

truth = {}
truth["SA1x"] = np.sqrt(SigmaABx[0, 0])
truth["SB1x"] = np.sqrt(SigmaABx[1, 1])
truth["CA1B1x"] = SigmaABx[0, 1]
truth["CorrA1B1x"] = truth["CA1B1x"] / (truth["SA1x"] * truth["SB1x"])

coverage_tables = []

for xpt in x_points:   
    
    xpt = np.array(xpt).reshape(2, 1)
    
    idx = np.all(df[["x1","x2"]] == xpt.flatten(), 1)
    tmp = df.loc[idx]
    
    EA1x_EB1x = muAB + SigmaABX @ SigmaXXinv @ xpt
    truth["EA1x"], truth["EB1x"] = EA1x_EB1x.flatten()
    
    # 95% Coverage    
    coverage = dict()
    for v in truth:
        q05_name = v + "_25"
        q95_name = v + "_975"
        within = (tmp[q05_name] < truth[v]) & (truth[v] < tmp[q95_name])
        coverage[v] = np.mean(within)
    
    coverage_tables.append(pd.Series(coverage))
    
    
table = pd.concat(coverage_tables, axis = 1)
table.columns = [str(s) for s in x_points]
table = table.loc[["EA1x", "EB1x", "SA1x", "SB1x", "CA1B1x", "CorrA1B1x"]]

table.index = ["$E[A_1|X]$", "$E[B_1|X]$", 
          "$Std[A_1|X]$", "$Std[B_1|X]$", 
          "$Cov[A_1, B_1|X]$", "$Corr[A_1, B_1|X]$"]

table.round(3).to_latex("bs_coverage/bs_conditional_coverage_20000.tex",
           escape = False)

print(table.round(3))

