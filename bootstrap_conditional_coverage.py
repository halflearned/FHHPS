#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:45:49 2017

@author: vitorhadad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:35:14 2017

@author: vitorhadad
"""

from data_utils import generate_data, clock_seed
import fhhps
from os.path import exists
import pandas as pd
from os import mkdir


config = dict(c_shocks = 3,
               alpha_shocks = .2,
               c_nw = .5,
               alpha_nw = 0.33,
               poly_order = 2)


if not exists("bs_coverage"):
    mkdir("bs_coverage")
    
    

n_sims = 100
filename = "bs_coverage/bs_conditional_coverage_20000.csv"

qs = [0.005,0.025,0.05,0.95,0.975,0.995]

x_points = [(-1,0), (1,0), (0,1), (0,-1)]


for i_sim in range(n_sims): # This can take many hours
    
    print("Simulation #", i_sim)
    
    # Initializes FHHPS with best configuration for n=20000
    algo = fhhps.FHHPS(**config)
    
    # Fake data
    seed = clock_seed()
    df   = generate_data(20000, 
                       vA = 9, vB = .4,
                       vU = 1, vV = .1, 
                       seed = seed)
    
    # FHHPS main routine
    algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])
    
    algo.fit()
    
    for xpt in x_points:
    
        # Bootstrapping
        algo.bootstrap(*xpt, n_iterations = 500)
        
        # Quantile table
        desc = algo._conditional_bootstrapped_values.quantile(qs)
        cols = [v + "_"+ str(int(1000*q)) for q in qs for v in desc.columns]
        values = desc.values.reshape(1, -1)
        table = pd.DataFrame(data=values,
                             columns=cols,
                             index=[seed])
        table["x1"] = xpt[0]
        table["x2"] = xpt[1]
        table.index.name = "seed"
        
        # Append to file
        table.to_csv(filename,
                     mode = "a",
                     header = not exists(filename))   
        