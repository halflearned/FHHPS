
"""
Generates linear panel data with correlated random intercept and slope,
then computes estimates according to FHHPS paper.
"""

import numpy as np
from fhhps import fhhps

# Generating data
n = 10000

S = np.array([[3, 1.5, 1, 1, 0, 0],
               [1.5, 3, 1, 1, 0, 0],
               [1, 1, 3, 1.5, 0, 0],
               [1, 1, 1.5, 3, 0, 0],
               [0, 0, 0, 0, .2, 0.1],
               [0, 0, 0, 0, 0.1, .2]])

m = np.array([2, 1, 0, 0, .3, .1])

df = pd.DataFrame(data = np.random.multivariate_normal(mean = m, cov = S, size = n),
                  columns = ["A1", "B1", "X1", "X2", "U2", "V2"])

df["A2"] = df["A1"] + df["U2"]
df["B2"] = df["B1"] + df["V2"]
df["Y1"] = df["A1"] + df["B1"]*df["X1"]
df["Y2"] = df["A2"] + df["B2"]*df["X2"]
 
# Computation
ab_moments = fhhps(Y1, Y2, X1, X2)