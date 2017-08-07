# FHHPS

Code for the paper Jeremy Fox, Vitor Hadad, Stefan Hoderlein, Amil Petrin and Robert Sherman (In progress) "Heterogenous Production Functions, Panel Data, and Productivity Dispersion". Henceforth <i>FHHPS</i>.

<center>
<img src="figs/scatter.png" width = 400>
</center>

## Main setup

The setup is a linear panel data with two correlated random coefficients. 

<center>
<img src="figs/fmla1.png" width = 150>
</centeR>

It is assumed that the coefficients follow AR(1) processes

<center>
<img src="figs/fmla2.png" width = 150>
</center>

where the shocks are assumed to be independent of everything in the previous period.

The application we have in mind is the identification and estimation of first and second moments of Cobb Douglas coefficients of a production function. For more details, please contact Vitor Hadad at baisihad@bc.edu.

## Usage

Making sure that the file `fhhps.py` is in the current folder, running the FHHPS script is as easy as

```python
ab_moments = fhhps(Y1, Y2, X1, X2)  # Y1,Y2: regressands; X1,X2: regressors
```

## Example

Please see the file `example.py` for a mock application.




