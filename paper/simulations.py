#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:11:14 2017

@author: vitorhadad
"""

from data_utils import generate_data, clock_seed
import fhhps
from os.path import exists

    
# RMSE-minimizing tuning parameter configurations for each n
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
         
filename = "fhhps_simulation_results.csv"

# Number of sims for each n
n_sims = 1000 

for _ in range(n_sims): # This can take many hours
    
    for n in ns:
        # Initializes FHHPS with best configuration
        algo = fhhps.FHHPS(**best_config[n])
        
        # Fake data
        df = generate_data(n, vA = 9, vB = .4,
                           vU = 1, vV = .1, 
                           seed = clock_seed())
        
        # FHHPS main routine
        algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
        algo.fit()
        
        # Outputs estimates to file
        algo.estimates().to_csv(filename,
                                mode = "a",
                                header = not exists(filename))
        

        
        
    
    
    