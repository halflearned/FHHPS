import os
import subprocess
from collections import OrderedDict as ODict
from os.path import join
from time import time

import seaborn as sns
from numpy.random import choice

from fhhps.estimator import *
from fhhps.utils import *

pd.options.display.max_columns = 999
sns.set_style("white")


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


if __name__ == "__main__":

    df = pd.read_csv(join("..", "empirical", "allcott_data_wide.csv"))

    filename = os.path.join("empirical_out", get_unique_filename())
    num_sims = 1000 if on_sherlock() else 1

    n = len(df)

    for _ in range(50):

        output_bw1_const_step1 = choice([.1, 1, 5, 10, 20, 50])
        output_bw1_const_step2 = choice([.1, 1, 5, 10, 20, 50])
        output_bw2_const = choice([.1, 1, 5, 10, 20, 50])
        shock_bw1_const = choice([.1, 1, 5, 10, 20, 50])
        censor1_const = choice([0.01, 0.1, 0.5, 1, 2])
        censor2_const = choice([0.01, 0.1, 0.5, 1, 2])
        kernel = choice(["neighbor", "gaussian"])
        shock_bw2_const = choice([.1, 1, 5, 10, 20])

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

        X = df[['lnK2008', 'lnK2009', 'lnK2010']].values
        Z = df[['lnW2008', 'lnW2009', 'lnW2010']].values
        Y = df[['lnY2008', 'lnY2009', 'lnY2010']].values

        # For the shocks, we do not use the neighbor kernel
        shock_means = fit_shock_means(X, Z, Y, bw=shock_bw1, kernel="gaussian")
        shock_cov = fit_shock_cov(X, Z, Y, shock_means, bw=shock_bw2, kernel="gaussian")

        output_cond_means_step1 = fit_output_cond_means(X, Z, Y, bw=output_bw1_step1, kernel=kernel)
        mean_valid = get_valid_cond_means(X, Z, censor1_bw)
        coef_cond_means_step1 = get_coef_cond_means(X, Z, output_cond_means_step1, shock_means)
        mean_estimate = get_coef_means(coef_cond_means_step1, mean_valid)

        output_cond_means_step2 = fit_output_cond_means(X, Z, Y, bw=output_bw1_step2, kernel=kernel)
        coef_cond_means_step2 = get_coef_cond_means(X, Z, output_cond_means_step2, shock_means)
        output_cond_cov = fit_output_cond_cov(
            X, Z, Y, output_cond_means_step2, bw=output_bw2, kernel=kernel, poly=2)
        coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)

        cov_valid = get_valid_cond_cov(X, Z, censor2_bw)
        cov_estimate = get_coef_cov(coef_cond_means_step2, coef_cond_cov, cov_valid)

        t2 = time()
        config = ODict(**{"n": n,
                          "kernel": kernel,
                          "output_bw1_const_step1": output_bw1_const_step1,
                          "output_bw1_const_step2": output_bw1_const_step2,
                          "output_bw2_const": output_bw2_const,
                          "output_bw1_alpha": output_bw1_alpha,
                          "output_bw2_alpha": output_bw2_alpha,
                          "shock_bw1_const": shock_bw1_const,
                          "shock_bw2_const": shock_bw2_const,
                          "shock_bw1_alpha": shock_bw1_alpha,
                          "shock_bw2_alpha": shock_bw2_alpha,
                          "censor1_const": censor1_const,
                          "censor2_const": censor2_const,
                          "mean_valid": np.mean(mean_valid),
                          "cov_valid": np.mean(cov_valid),
                          "time": t2 - t1
                          })

        mean_names = ["EA", "EB", "EC"]
        cov_names = ["VarA", "VarB", "VarC", "CovAB", "CovAC", "CovBC"]

        config.update(zip(mean_names, mean_estimate))
        config.update(zip(cov_names, cov_estimate))

        if on_sherlock():
            pd.DataFrame(config, index=[abs(hash(str(config)))]) \
                .to_csv(filename, header=False, index=False, mode="a")
