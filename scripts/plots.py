from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filenames = glob("/Users/vitorh/Documents/FHHPS/scripts/results/*.csv.bz2")
df = pd.concat([pd.read_csv(f) for f in filenames])

df_grp = df.query(
    "(output_bw1_const == 5) & "
    "(output_bw2_const == 5) & "
    "(shock_bw1_const == 1) & "
    "(shock_bw2_const == 1) & "
    "(output_bw1_alpha == 0.2) &"
    "(output_bw2_alpha == 0.2) &"
    "(shock_bw1_alpha == 0.2) &"
    "(shock_bw2_alpha == 0.2)"
)

g = sns.FacetGrid(data=df_grp, hue="n", col="name",
                  sharey=False, sharex=False,
                  aspect=1, height=3, col_wrap=3)
g.map(sns.kdeplot, "error", shade=True)
g.map(sns.rugplot, "error")
[ax.axvline(0, color="black") for ax in g.axes.flatten()]
plt.show()
