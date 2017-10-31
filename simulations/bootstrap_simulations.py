#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:08:20 2017

@author: vitorhadad
"""

import fhhps
from os.path import exists
from data_utils import generate_data

best_config = {500: dict(c_shocks = 2,
                       alpha_shocks = .2,
                       c_nw = .5,
                       alpha_nw = 0.167,
                       poly_order = 2),
              2000: dict(c_shocks = 3,
                       alpha_shocks = .2,
                       c_nw = .5,
                       alpha_nw = 0.167,
                       poly_order = 2),
              5000: dict(c_shocks = 4,
                       alpha_shocks = .2,
                       c_nw = .5,
                       alpha_nw = 0.167,
                       poly_order = 2),
              10000: dict(c_shocks = 4,
                       alpha_shocks = .2,
                       c_nw = .5,
                       alpha_nw = 0.167,
                       poly_order = 2),
              20000: dict(c_shocks = 3,
                       alpha_shocks = .2,
                       c_nw = .5,
                       alpha_nw = 0.33,
                       poly_order = 2)}
    
ns = [500, 2000, 5000, 10000, 20000]

# Fixed seed
seed = 12345

# Fake data
for n in ns:
    
    n = 20000
    df = generate_data(n, 9, .4, 1, .1, seed)
    
    # Initialize and fit FHHPS method
    algo = fhhps.FHHPS(**best_config)
    algo.fit(df["X1"], df["X2"], df["Y1"], df["Y2"])
      
    # Boostrap 100 times
    bs = algo.bootstrap(100)
    
    # Plot results and save them as pdf
    algo.plot()
    
    # Output result to csv
    filename = "bootstrap_simulations_{}.csv".format(n),
    bs.to_csv(filename,
               mode = "a",
               header = not exists(filename))
    break






    
    