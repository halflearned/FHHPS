import os

from fhhps.estimator import *
from fhhps.utils import *

if __name__ == "__main__":

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

        print(f'Saving output in: {WRITE_PATH} as {FNAME}')

        t1 = time()

        n = np.random.choice([1000, 5000, 20000])
        X, Z, Y = fake_data(n)
        est = FHHPSEstimator()
        est.add_data(X, Z, Y)
        est.fit_shock_first_moments()
        est.fit_output_cond_first_moments()
        est.fit_coefficient_first_moments()
        output = dict(
            n=n,
            shock_first_moments=est.shock_first_moments,
            coefficient_first_moments=est.coefficient_first_moments)
        save_pickle(obj=output,
                    path=os.path.join(WRITE_PATH, FNAME))

        t2 = time()
        print(f"Processed {n} obs in {t2 - t1} seconds (first coeff only).")
