from fhhps.estimator import *
from fhhps.utils import *

n = 5000
X, Z, Y = fake_data(n)
est = FHHPSEstimator()
est.add_data(X, Z, Y)
est.fit_shock_first_moments()
est.fit_output_cond_first_moments()
est.fit_coefficient_first_moments()
print(est.coefficient_first_moments)
