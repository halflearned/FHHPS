import os
import subprocess
from collections import OrderedDict as ODict
from fractions import Fraction
from time import time

from numpy.random import choice

from fhhps import *
import pandas as pd
pd.options.display.max_columns = 99

import warnings
warnings.filterwarnings("ignore")

def as_frac(x):
    frac = Fraction(x).limit_denominator(10000)
    if frac.denominator != 1:
        return f"{frac.numerator}/{frac.denominator}"
    else:
        return f"{frac.numerator}"


def on_sherlock():
    return 'SCRATCH' in os.environ


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


if __name__ == "__main__":


    mean_names = ["EA", "EB", "EC"]
    cov_names = ["VarA", "VarB", "VarC", "CovAB", "CovAC", "CovBC"]
    num_sims = 100 if on_sherlock() else 1

    for s in range(num_sims):

        print(f"Simulation {s}")

        if on_sherlock():
            n = choice([2500, 5000, 10000, 20000], p=[0.2, 0.2, .3, .3])

            kernel1 = choice(["neighbor", "gaussian"])
            kernel2 = choice(["neighbor", "gaussian"])

            output_bw1_const_step1 = np.random.choice([.01, .1, 1])
            output_bw1_const_step2 = output_bw1_const_step1
            output_bw2_const = output_bw1_const_step1

            shock_bw1_const = np.random.choice([.01, .1, 1])
            shock_bw2_const = np.random.choice([.01, .1, 1])

            censor1_const = 1#choice([.5, 1, 2])
            censor2_const = 1#choice([.5, 1, 2])
        else:
            output_bw1_const_step1 = .1
            output_bw1_const_step2 = .01
            output_bw2_const = .1
            shock_bw1_const = .1
            n = 50000
            kernel1 = "neighbor"
            kernel2 = "neighbor"
            shock_bw2_const = 1
            censor1_const = 1.
            censor2_const = 1.

        t1 = time()

        shock_bw1_alpha = 1 / 6
        shock_bw2_alpha = 1 / 6

        output_bw1_alpha = 1 / 10
        output_bw2_alpha = 1 / 10

        censor1_alpha = 1 / 5
        censor2_alpha = 1 / 5

        output_bw1_step1 = output_bw1_const_step1 * n ** (-output_bw1_alpha)
        output_bw1_step2 = output_bw1_const_step2 * n ** (-output_bw1_alpha)
        output_bw2 = output_bw2_const * n ** (-output_bw2_alpha)

        shock_bw1 = shock_bw1_const * n ** (-shock_bw1_alpha)
        shock_bw2 = shock_bw2_const * n ** (-shock_bw2_alpha)

        censor1_bw = censor1_const * n ** (-censor1_alpha)
        censor2_bw = censor2_const * n ** (-censor2_alpha)

        print("Generating data...")
        fake = generate_data(n)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        print("Fitting shock means...")
        shock_means = fit_shock_means(X, Z, Y, bw=shock_bw1, kernel=kernel1)
        shock_cov = fit_shock_cov(X, Z, Y, shock_means, bw=shock_bw2, kernel=kernel1)

        print("Fitting output conditional means...")
        output_cond_means_step1 = fit_output_cond_means(X, Z, Y, bw=output_bw1_step1,
                                                        kernel=kernel2)
        output_cond_means_step2 = fit_output_cond_means(X, Z, Y, bw=output_bw1_step2,
                                                        kernel=kernel2)
        
        print("Fitting output conditional covariances...")
        output_cond_cov = fit_output_cond_cov(
            X, Z, Y, output_cond_means_step2, bw=output_bw2, kernel=kernel2, poly=1)

        print("Censoring...")
        mean_valid = get_valid_cond_means(X, Z, censor1_bw)
        coef_cond_means_step1 = get_coef_cond_means(X, Z, output_cond_means_step1, shock_means)
        mean_estimate = get_coef_means(coef_cond_means_step1, mean_valid)

        print("Getting coefficient covariances...")
        coef_cond_means_step2 = get_coef_cond_means(X, Z, output_cond_means_step2, shock_means)
        coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)
        cov_valid = get_valid_cond_cov(X, Z, censor2_bw)
        cov_estimate = get_coef_cov(coef_cond_means_step2, coef_cond_cov, cov_valid)

        t2 = time()

        print(f"Time: {t2 - t1}")            

        print("Output.")
        config = ODict(**{"n": n,
                          "kernel1": kernel1,
                          "kernel2": kernel2,
                          "output_bw1_const_step1": as_frac(output_bw1_const_step1),
                          "output_bw1_const_step2": as_frac(output_bw1_const_step2),
                          "output_bw2_const": as_frac(output_bw2_const),
                          "output_bw1_alpha": as_frac(output_bw1_alpha),
                          "output_bw2_alpha": as_frac(output_bw2_alpha),
                          "shock_bw1_const": as_frac(shock_bw1_const),
                          "shock_bw2_const": as_frac(shock_bw2_const),
                          "shock_bw1_alpha": as_frac(shock_bw1_alpha),
                          "shock_bw2_alpha": as_frac(shock_bw2_alpha),
                          "mean_valid": np.mean(mean_valid),
                          "cov_valid": np.mean(cov_valid),
                          "time": t2 - t1
                          })

        mean_res = pd.DataFrame(data=[ODict({**config, "name": name, "value": est})
                                      for name, est in zip(mean_names, mean_estimate)])
        cov_res = pd.DataFrame(data=[ODict({**config, "name": name, "value": est})
                                     for name, est in zip(cov_names, cov_estimate)])

        print(mean_res)
        print(cov_res)

        if on_sherlock():
            filename = os.path.join("/scratch/users/dulguun/vitor", get_unique_filename())
            mean_res.to_csv(filename, header=False, index=False, mode="a")
            cov_res.to_csv(filename, header=False, index=False, mode="a")
