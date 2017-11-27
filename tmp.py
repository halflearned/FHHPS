#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:44:43 2017

@author: vitorhadad
"""

from fhhps import FHHPS
import pandas as pd

df= pd.read_csv("fake_data_1.csv")

algo = fhhps = FHHPS()
algo.add_data(df["X1"], df["X2"], df["Y1"], df["Y2"])

algo.fit()

#algo.bootstrap(5)
algo.show_tuning_parameters()