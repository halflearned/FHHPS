# FHHPS
Code used in Fox, Hadad, Hoderlein, Petrin and Sherman (in progress)

## Main idea of the paper

Given the following setup

$$A$$


## Estimation

Each of the $E[Y_i | X_1 = x_1, X_2 = x_2]$'s are estimated using Nadaraya-Watson nonparametric estimators with MISE-minimizing bandwidths:

$$h_{c,n} = \ c \widehat{\sigma}_X  n^{\frac{-1}{6}} \qquad \text{where } \hat{\sigma}_X \text{ is an estimate of }Std(X_1) = Std(X_2) \text{ and }c \in \{.1, .5, .8\}$$

The scaling constants associated with estimating first and second moments are \texttt{c1\_nw} and \texttt{c2\_nw}, respectively.

When inverting the matrices, we exclude observations near the diagonal $X_1 = X_2$. The threshold parameters $t_1$ and $t_2$  that govern how much we  exclude are given by:

$$t_{c, n}  = \ c \widehat{\sigma}_X  n^{\frac{-1}{4}}  \qquad \text{For the first moments}$$
$$t_{c, n}  = \ c \widehat{\sigma}_X  n^{\frac{-1}{8}}  \qquad \text{For the second moments}$$

\noindent where we keep $c = 1$ throughout.

