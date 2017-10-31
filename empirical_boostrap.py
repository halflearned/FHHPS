#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:46:55 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd
import fhhps

df_full = pd.read_csv("final.csv")

# Create auxiliary vars
df_full["YM_W"] = np.log(np.exp(df_full["lnY"]) - np.exp(df_full["lnM"])) - df_full["lnW"]
df_full["K_W"]  = df_full["lnK"] - df_full["lnW"]

# Pivot table
variables = [df_full.pivot(index = "FACT_ID", 
                           columns = "YEAR", 
                           values = v) 
                          for v in ["K_W", "YM_W"]]
scheme = df_full.groupby("FACT_ID")["scheme"].first()

    
df = pd.concat(variables + [scheme], axis = 1)
df.columns = ["X1", "X2", "Y1", "Y2", "scheme"]

# Restrict sample
df = df[df["scheme"] == "Sample"].dropna()

# Run fhhps and bootstrap
algo = fhhps.FHHPS(c_shocks = 3,
                   alpha_shocks = .2,
                   c_nw = .5,
                   alpha_nw = 0.167,
                   poly_order = 1)

algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
algo.fit()
bs = algo.bootstrap(n_iterations = 20)

(shockfig, shockaxs), (rcfig, rcaxs) = algo.plot_density()
shockfig.savefig("bootstrap_shocks.pdf")
shockfig.savefig("bootstrap_random_coefficients.pdf")

shock_tab, rc_tab = algo.summary()

#%%
algo.conditional_fit(0, 1)
bsx = algo.bootstrap(x1 = 0, x2 = 1, n_iterations = 20)
#%%
summ = algo.conditional_summary()

algo.plot_conditional_density()

