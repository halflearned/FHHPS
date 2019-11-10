import pandas as pd
from os.path import join
from fhhps.estimator import fhhps

# Path to allcott data
df = pd.read_csv(join("..", "empirical", "allcott_data_wide.csv"))

result = fhhps(X=df[['lnK2008', 'lnK2009', 'lnK2010']].values,
               Z=df[['lnW2008', 'lnW2009', 'lnW2010']].values,
               Y=df[['lnY2008', 'lnY2009', 'lnY2010']].values,
               kernel1="gaussian",
               kernel2="neighbor",
               shock_bw1_const=1.,
               shock_bw2_const=1.,
               output_bw1_const_step1=1.,
               output_bw1_const_step2=1.,
               output_bw2_const=1.,
               censor1_const=1.,
               censor2_const=1.)

result.to_csv("empirical_out", header=False, index=False, mode="a")
