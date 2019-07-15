import os
import subprocess
from random import choice

from fhhps.estimator import *
from fhhps.utils import *


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
names = ["VarA", "VarB", "VarC", "CovAB", "CovAC", "CovBC"]
for s in range(1000):

    if on_sherlock():
        n = choice([1000, 5000, 20000])
    else:
        n = 2000

    output_bw1_const = choice([.1, .5, 1, 3, 5, 10, 20])
    output_bw2_const = output_bw1_const

    shock_bw1_const = choice([.1, .5, 1, 3, 5, 10, 20])
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

    shock_means = fit_shock_means(X, Z, Y, shock_bw1)
    shock_cov = fit_shock_cov(X, Z, Y, shock_means, shock_bw2)

    output_cond_means = fit_output_cond_means(X, Z, Y, output_bw1)
    output_cond_cov = fit_output_cond_cov(X, Z, Y, output_cond_means, output_bw2)

    coef_cond_means = get_coef_cond_means(X, Z, output_cond_means, shock_means)
    coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)

    valid = get_valid_cond_cov(X, Z, 1.)

    estimate = get_coef_cov(coef_cond_means, coef_cond_cov, valid)
    truth = get_true_coef_cov(fake)

    res = pd.DataFrame(data=[{"n": n,
                              "output_bw1_const": output_bw1_const,
                              "output_bw2_const": output_bw2_const,
                              "output_bw1_alpha": output_bw1_alpha,
                              "output_bw2_alpha": output_bw2_alpha,
                              "shock_bw1_const": shock_bw1_const,
                              "shock_bw2_const": shock_bw2_const,
                              "shock_bw1_alpha": shock_bw1_alpha,
                              "shock_bw2_alpha": shock_bw2_alpha,
                              "name": name,
                              "value": est}
                             for name, est in zip(names, estimate)])
    res.to_csv(filename, header=s == 0, index=False, mode="a")
    print(res)
