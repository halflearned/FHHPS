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
           "EU": .3,
           "EV": .1,
           "SA1": 3,
           "SB1": np.sqrt(.4),
           "CA1B1": .5*np.sqrt(.4)*3,
           "CorrA1B1": .5,
           "SU": 1,
           "SV": np.sqrt(.1),
           "CUV": np.sqrt(.1)*1*.5,
           "CorrUV": .5}

# 95% Coverage

coverage = defaultdict(lambda: 0)
for v in truth:
    q05_name = v + "_25"
    q95_name = v + "_975"
    within = (df[q05_name] <= truth[v]) & (truth[v] <= df[q95_name])
    coverage[v] = np.mean(within)
   

s = pd.Series(coverage)
print(s)
