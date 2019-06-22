import os
import subprocess
from collections import OrderedDict as ODict

from fhhps.estimator import *
from fhhps.utils import *


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


def flatten(table):
    output = {}
    m, n = table.shape
    for i in range(m):
        for j in range(n):
            value = table.iloc[i, j]
            key = table.index[i].replace("t", str(j + 1))
            output[key] = value
    return output


def make_row(config, stat, value):
    row = ODict(**config)
    row["statistic"] = stat
    row["value"] = value
    return row


def make_chunk(config, est):
    results = {**flatten(est.shock_means),
               **est.coefficient_means,
               **flatten(est.shock_cov),
               **est.coefficient_cov}
    rows = []
    for stat, val in results.items():
        rows.append(make_row(config, stat, val))
    return pd.DataFrame(rows)


def on_sherlock():
    return 'GROUP_SCRATCH' in os.environ


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    while True:

        OUT_DIRNAME = __file__.split('.')[0].split('/')[-1] + '_out/'
        if 'GROUP_SCRATCH' in os.environ:
            out_dir = os.environ['GROUP_SCRATCH']
            sid = os.environ['SLURM_JOB_ID']
            tid = os.environ['SLURM_LOCALID']
            WRITE_PATH = os.path.join(out_dir, 'bandits', 'scripts', OUT_DIRNAME)
        else:
            out_dir = os.path.abspath(os.path.dirname(__file__))
            WRITE_PATH = os.path.join(out_dir, OUT_DIRNAME)
        FNAME = get_unique_filename()

        if not os.path.exists(WRITE_PATH):
            os.makedirs(WRITE_PATH)

        if on_sherlock():
            config = ODict()
            config["n"] = np.random.choice([1000, 2000, 10000])
            config["shock_const"] = 5.0
            config["shock_alpha"] = 0.2
            config["coef_const"] = np.random.choice([5., 10., 25, 50])
            config["coef_alpha"] = 0.5
            config["censor1_const"] = 3.0
            config["censor2_const"] = 3.0
        else:
            config = ODict()
            config["n"] = 2500
            config["shock_const"] = 5.0
            config["shock_alpha"] = 0.2
            config["coef_const"] = 5
            config["coef_alpha"] = 0.5
            config["censor1_const"] = 3.0
            config["censor2_const"] = 3.0

        logging.info(f'Saving output in: {WRITE_PATH} as {FNAME}')
        logging.info(config)

        t1 = time()
        fake = generate_data(n=config["n"])
        data = fake["df"]

        est = FHHPSEstimator(shock_const=config["shock_const"],
                             shock_alpha=config["shock_alpha"],
                             coef_const=config["coef_const"],
                             coef_alpha=config["coef_alpha"],
                             censor1_const=config["censor1_const"],
                             censor2_const=config["censor2_const"])
        est.add_data(X=data[["X1", "X2", "X3"]],
                     Z=data[["Z1", "Z2", "Z3"]],
                     Y=data[["Y1", "Y2", "Y3"]])

        est.fit_shock_means()
        est.fit_output_cond_means()
        est.fit_coefficient_means()
        est.fit_output_cond_means()
        est.fit_coefficient_means()
        est.fit_output_cond_cov()
        est.fit_shock_second_moments()
        est.fit_coefficient_second_moments()

        t2 = time()

        if on_sherlock():
            output = make_chunk(config, est)
            output.to_csv(os.path.join(WRITE_PATH, FNAME + ".csv.bz2"))
