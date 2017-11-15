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

#df = pd.read_csv("final.csv")

# Restrict sample to "Sample" sampling scheme only
df = df[df["scheme"] == "Sample"]

# Run fhhps and bootstrap
algo = fhhps.FHHPS(c_shocks = 3,
                   alpha_shocks = .2,
                   poly_order = 1,
                   c_nw = .5,
                   c1_cens = 1,
                   alpha_nw = .167)

algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
algo.fit()

print(algo.estimates()[algo._shock_vars])
print(algo.estimates()[algo._rc_vars])


#
#bs = algo.bootstrap(n_iterations = 50)
#
#(shockfig, shockaxs), (rcfig, rcaxs) = algo.plot_density()
#shockfig.savefig("empirical_shocks.pdf")
#shockfig.savefig("empirical_random_coefficients.pdf")
#
#shock_tab, rc_tab = algo.summary()

