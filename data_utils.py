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


def generate_data(n, 
                  EA = 2, EB = .4,
                  EX1 = 0, EX2 = 0,
                  EU = .3, EV = .1, 
                  vA = 9, vB = .4,
                  vU = 1, vV = .1,
                  rho = .5,
                  seed = None):


    corr = np.array( [[1,  rho, rho,  rho,  0,  0],
                      [rho,  1, rho,  rho,  0,  0],
                      [rho, rho,  1,  rho,  0,  0],
                      [rho, rho, rho,   1,  0,  0],
                      [0,   0,  0,   0,  1, rho],
                      [0,   0,  0,   0, rho,  1]] )

    scaling = np.diag(np.sqrt(np.array([vA, vB, 1, 1, vU, vV])))

    S  = scaling @ corr @ scaling

    m = np.array([EA, EB, EX1, EX2, EU, EV])

    np.random.seed(seed)
    df = pd.DataFrame(data = np.random.multivariate_normal(mean = m, cov = S, size = n),
                      columns = ["A1", "B1", "X1", "X2", "U2", "V2"])

    df["A2"] = df["A1"] + df["U2"]
    df["B2"] = df["B1"] + df["V2"]
    df["Y1"] = df["A1"] + df["B1"]*df["X1"]
    df["Y2"] = df["A2"] + df["B2"]*df["X2"]

    return df



