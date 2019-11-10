# FHHPS

Code used in paper Jeremy Fox, Vitor Hadad, Stefan Hoderlein, Amil Petrin and Robert Sherman (In progress). *Heterogenous Production Functions, Panel Data, and Productivity Dispersion*. Our application is called <i>FHHPS</i>, after the authors' initials.


## Estimation setup

The setup is a **linear panel data** with **two correlated random coefficients**. It is assumed that the coefficients follow AR(1) processes where the shocks are assumed to be independent of everything in the previous K periods.

The application we have in mind is the identification and estimation of first and second moments of Cobb Douglas coefficients of a production function. For more details, please contact Vitor Hadad at vitorh@stanford.edu.

## Installing / Development.

To run `fhhps`, you will need Python 3.7 or newer.

**For MAC OSX / Linux**

Clone our repo.
```bash
git clone https://github.com/halflearned/FHHPS/
cd FHHPS/
```

To install in an existing environment.
```
python setup.py develop
```

To create a new environment (recommended).
```
conda create --name fhhpsenv
conda activate fhhpsenv
python setup.py develop
```

After that you should be able to `import` our package like you would any other.
