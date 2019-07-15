from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filenames = glob("/Users/vitorh/Documents/fhhps/scripts/script_out/results.csv")
df = pd.concat([pd.read_csv(f) for f in filenames])

df_grp = df.query(
    "(output_bw1_const == 5) & "
    "(output_bw2_const == 5) & "
    "(shock_bw1_const == 1) & "
    "(shock_bw2_const == 1) & "
    "(output_bw1_alpha == 0.2) &"
    "(output_bw2_alpha == 0.2) &"
    "(shock_bw1_alpha == 0.3) &"
    "(shock_bw2_alpha == 0.3)"
)

g = sns.FacetGrid(data=df_grp, hue="n", col="name",
                  sharey=False, sharex=False, legend_out=True,
                  aspect=1, height=3, col_wrap=3)
g.add_legend()
g.map(sns.kdeplot, "error", shade=True)
g.map(sns.rugplot, "error")
g.map(lambda **kwargs: plt.axvline(0, **kwargs), color="black")
g.savefig("/Users/vitorh/Desktop/test.pdf")
plt.show()
