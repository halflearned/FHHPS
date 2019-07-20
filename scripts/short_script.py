import os
import subprocess
from collections import OrderedDict as ODict
from fractions import Fraction
from random import choice

from fhhps.estimator import *
from fhhps.utils import *

import numpy as np


def as_frac(x):
    frac = Fraction(x).limit_denominator(10000)
    if frac.denominator != 1:
        return f"{frac.numerator}/{frac.denominator}"
    else:
        return f"{frac.numerator}"


def on_sherlock():
    return 'GROUP_SCRATCH' in os.environ


def get_unique_filename(prefix="results", rnd=None, commit=True):
    if rnd is None:
        rnd = clock_seed()
    if commit:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        hash = out.strip().decode('ascii')
    else:
        hash = ''
    if on_sherlock():
        sid = os.environ['SLURM_JOB_ID']
        tid = os.environ['SLURM_LOCALID']
        fname = f'{prefix}_{hash}_{sid}_{tid}_{rnd}.csv.bz2'
    else:
        rnd = clock_seed()
        fname = f'{prefix}_{hash}_{rnd}.csv.bz2'
    return fname


filename = os.path.join("script_out", get_unique_filename())
mean_names = ["EA", "EB", "EC"]
cov_names = ["VarA", "VarB", "VarC", "CovAB", "CovAC", "CovBC"]
num_sims = 1000 if on_sherlock() else 1

for s in range(num_sims):

    if on_sherlock():
        n = choice([1000, 5000, 20000])
        output_bw1_const = choice([0.05, .1, .25, .5, 1])
        shock_bw1_const = choice([0.05, .1, .25, .5, 1])
        kernel = "neighbor"  # choice(["gaussian", "neighbor"])
    else:
        output_bw1_const = 0.25
        shock_bw1_const = .1
        n = 5000
        kernel = "neighbor"

    output_bw2_const = output_bw1_const
    shock_bw2_const = shock_bw1_const

    shock_bw1_alpha = 1 / 6
    shock_bw2_alpha = 1 / 6

    output_bw1_alpha = 1 / 10
    output_bw2_alpha = 1 / 10

    censor1_const = 1.
    censor2_const = 1.

    censor1_alpha = 1 / 5
    censor2_alpha = 1 / 5

    output_bw1 = output_bw1_const * n ** (-output_bw1_alpha)
    output_bw2 = output_bw2_const * n ** (-output_bw2_alpha)

    shock_bw1 = shock_bw1_const * n ** (-shock_bw1_alpha)
    shock_bw2 = shock_bw2_const * n ** (-shock_bw2_alpha)

    censor1_bw = censor1_const * n ** (-censor1_alpha)
    censor2_bw = censor2_const * n ** (-censor2_alpha)

    fake = generate_data(n)
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values
    Y = fake["df"][["Y1", "Y2", "Y3"]].values

    shock_means = fit_shock_means(X, Z, Y, bw=shock_bw1, kernel=kernel)
    shock_cov = fit_shock_cov(X, Z, Y, shock_means, bw=shock_bw2, kernel=kernel)

    output_cond_means = fit_output_cond_means(X, Z, Y, bw=output_bw1, kernel=kernel)
    output_cond_cov = fit_output_cond_cov(X, Z, Y, output_cond_means, bw=output_bw2, kernel=kernel)

    coef_cond_means = get_coef_cond_means(X, Z, output_cond_means, shock_means)
    coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)

    mean_valid = get_valid_cond_means(X, Z, censor1_bw)
    cov_valid = get_valid_cond_cov(X, Z, censor2_bw)

    mean_estimate = get_coef_means(coef_cond_means, mean_valid)
    cov_estimate = get_coef_cov(coef_cond_means, coef_cond_cov, cov_valid)
    truth = get_true_coef_cov(fake)

    config = ODict(**{"n": n,
                      "kernel": kernel,
                      "output_bw1_const": as_frac(output_bw1_const),
                      "output_bw2_const": as_frac(output_bw2_const),
                      "output_bw1_alpha": as_frac(output_bw1_alpha),
                      "output_bw2_alpha": as_frac(output_bw2_alpha),
                      "shock_bw1_const": as_frac(shock_bw1_const),
                      "shock_bw2_const": as_frac(shock_bw2_const),
                      "shock_bw1_alpha": as_frac(shock_bw1_alpha),
                      "shock_bw2_alpha": as_frac(shock_bw2_alpha),
                      "mean_valid": np.mean(mean_valid),
                      "cov_valid": np.mean(cov_valid)
                      })

    mean_res = pd.DataFrame(data=[ODict({**config, "name": name, "value": est})
                                  for name, est in zip(mean_names, mean_estimate)])
    cov_res = pd.DataFrame(data=[ODict({**config, "name": name, "value": est})
                                 for name, est in zip(cov_names, cov_estimate)])

    mean_res.to_csv(filename, header=s == 0, index=False, mode="a")
    cov_res.to_csv(filename, header=s == 0, index=False, mode="a")

    print(mean_res)
    print(cov_res)
