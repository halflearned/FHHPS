#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:02:33 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_results.csv",
                 header = 0)

#%%
# Figures

shock_vars = ["EU", "EV", "SU", "SV", "CUV", "CorrUV"]
rc_vars = ["EA", "EB", "SA", "SB", "CAB", "CorrAB"]  

df = df.loc[~np.any(df[shock_vars + rc_vars] > 20, 1)]
df = df.loc[df["n"].isin([500, 2000, 5000, 10000, 20000])]



true_values = {"EU": .3,
               "EV": .1,
               "SU": 1,
               "SV": np.sqrt(.1),
               "CUV": .5 * np.sqrt(.1) * 1,
               "CorrUV": .5,
               "EA": 2,
               "EB": .4,
               "SA": 3,
               "SB": np.sqrt(.4),
               "CAB": .5 * np.sqrt(.4) * 3,
               "CorrAB": .5}


latex_names =   {"EU": "$E[U_2]$",
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
                 "CorrAB": "$Corr[A_1, B_1]$"}

xlims = {"EU": (0, .6),
           "EV": (-.1, .3),
           "SU": (.5, 1.5),
           "SV": (0, 1), #np.sqrt(.1),
           "CUV": (-0.05, 0.5), # 0.1581
           "CorrUV": (-.5, 1.5),
           "EA": (1, 3),
           "EB": (0, 1),
           "SA": (2.5, 3.5),
           "SB": (0, 1.2), #np.sqrt(.4),
           "CAB": (0, 1.5), # .5 * np.sqrt(.4) * 3,
           "CorrAB": (-.5, 1.5)}

shockfig, axs = plt.subplots(3, 2, figsize = (10, 10))
shockfig.subplots_adjust(hspace = .4)
shockfig.suptitle("Simulated shock moment estimates", fontsize = 16)
shockaxs = []
for s, ax in zip(shock_vars, axs.flatten()):
    df.groupby("n")[s].plot.density(ax = ax)
    ax.set_title(latex_names[s], fontsize = 14)
    ax.axvline(true_values[s], *ax.get_ylim(), linewidth = 3, color = "black")
    ax.set_xlim(xlims[s])
    ax.legend([500, 2000, 5000, 10000, 20000])
    shockaxs.append(ax)
    
#%%
rcfig, axs = plt.subplots(3, 2, figsize = (10, 10))
rcfig.subplots_adjust(hspace = .4)
rcfig.suptitle("Simulated random coefficient moment estimates", fontsize = 16)
rcaxs = []
for s, ax in zip(rc_vars, axs.flatten()):
    df.groupby("n")[s].plot.density(ax = ax)
    ax.set_title(latex_names[s], fontsize = 14)
    ax.axvline(true_values[s], *ax.get_ylim(), linewidth = 3, color = "black")
    ax.set_xlim(xlims[s])
    ax.legend([500, 2000, 5000, 10000, 20000])
    rcaxs.append(ax)
    if "Corr" in s:
        xlim = ax.get_xlim()
        ax.set_xlim(max(-1.5, xlim[0]), min(1.5, xlim[1]))
        
        
#%%
shockfig.savefig("simulation_shocks.pdf")
rcfig.savefig("simulation_random_coefficients.pdf")
    

#%% # Tables
stats = lambda s: {
       "bias":     lambda x: np.mean(x - true_values[s]),
       "rel_bias": lambda x: np.mean((x - true_values[s])/true_values[s]),
       "abs_bias": lambda x: np.mean(np.abs(x - true_values[s])),
       "rmse":     lambda x: np.sqrt(np.mean((x - true_values[s])**2)),
       "number of sims": lambda x: x.size
      }

for v in shock_vars + rc_vars: 
    tab = df.groupby("n")[v].agg(stats(v))
    tab.to_latex("simulation_statistics_{}.tex".format(v), 
                 escape = False)


