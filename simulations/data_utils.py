#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:45:28 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
from time import time

def clock_seed():
    t = int(time()*1e4)
    return int(str(t)[-8:])


def generate_data(n, vA, vB, vU, vV, seed = None):


    corr = np.array( [[1,  .5, .5,  .5,  0,  0],
                      [.5,  1, .5,  .5,  0,  0],
                      [.5, .5,  1,  .5,  0,  0],
                      [.5, .5, .5,   1,  0,  0],
                      [0,   0,  0,   0,  1, .5],
                      [0,   0,  0,   0, .5,  1]] )

    scaling = np.diag(np.sqrt(np.array([vA, vB, 1, 1, vU, vV])))

    S  = scaling @ corr @ scaling


    m  = np.array([2, .4, 0, 0, .3, .1])

    np.random.seed(seed)
    df = pd.DataFrame(data = np.random.multivariate_normal(mean = m, cov = S, size = n),
                      columns = ["A1", "B1", "X1", "X2", "U2", "V2"])

    df["A2"] = df["A1"] + df["U2"]
    df["B2"] = df["B1"] + df["V2"]
    df["Y1"] = df["A1"] + df["B1"]*df["X1"]
    df["Y2"] = df["A2"] + df["B2"]*df["X2"]

    return df

