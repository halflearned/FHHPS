#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:46:55 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd
import fhhps

df = pd.read_csv("indian_data.csv")

# Restrict sample to "Sample" sampling scheme only
df = df[df["scheme"] == "Sample"].dropna()

# Run fhhps and bootstrap
algo = fhhps.FHHPS(c_shocks = 3,
                   alpha_shocks = .2,
                   c_nw = .5,
                   alpha_nw = 0.167,
                   poly_order = 2)

algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
algo.fit()
bs = algo.bootstrap(n_iterations = 100)

(shockfig, shockaxs), (rcfig, rcaxs) = algo.plot_density()
shockfig.savefig("empirical_shocks.pdf")
shockfig.savefig("empirical_random_coefficients.pdf")

shock_tab, rc_tab = algo.summary()

