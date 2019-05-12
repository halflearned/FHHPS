import os

from fhhps.estimator import *
from fhhps.utils import *


def flatten(table):
    output = {}
    m, n = table.shape
    for i in range(m):
        for j in range(n):
            value = table.iloc[i, j]
            key = table.index[i].replace("t", str(j))
            output[key] = value
    return output


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    while True:

        OUT_DIRNAME = __file__.split('.')[0].split('/')[-1] + '_out/'
        if 'GROUP_SCRATCH' in os.environ:
            out_dir = os.environ['GROUP_SCRATCH']
            sid = os.environ['SLURM_JOB_ID']
            tid = os.environ['SLURM_LOCALID']
            WRITE_PATH = os.path.join(out_dir, 'bandits', 'scripts', OUT_DIRNAME)
            FNAME = f'{sid}_{tid}_{clock_seed()}'
        else:
            out_dir = os.path.abspath(os.path.dirname(__file__))
            WRITE_PATH = os.path.join(out_dir, OUT_DIRNAME)
            FNAME = f'{clock_seed()}'

        if not os.path.exists(WRITE_PATH):
            os.makedirs(WRITE_PATH)

        config = {}
        config["n"] = np.random.choice([1000, 5000, 20000])
        config["shock_const"] = np.random.choice([1.0, 0.5, 0.1])
        config["shock_alpha"] = np.random.choice([0.2])
        config["coef_const"] = np.random.choice([1.0, 2.0, 5.0])
        config["coef_alpha"] = np.random.choice([0.25, 0.5])
        config["censor1_const"] = np.random.choice([0.01, 0.1, 0.2])
        config["censor2_const"] = np.random.choice([0.01, 0.1, 0.2])

        logging.info(f'Saving output in: {WRITE_PATH} as {FNAME}')
        logging.info(config)

        t1 = time()
        fake = generate_data(n=config["n"])
        data = fake["df"]

        try:
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
            results = {**config,
                       **flatten(est.shock_means),
                       **est.coefficient_means,
                       **flatten(est.shock_cov),
                       **est.coefficient_cov}
            results["time"] = t2 - t1
            output = pd.DataFrame(results, index=[0])
            output.to_csv(os.path.join(WRITE_PATH, FNAME + ".csv.bz2"))
        except Exception as e:
            logging.info("Found some error")
            logging.error(e)
