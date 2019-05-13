# FHHPS

Code used in paper Jeremy Fox, Vitor Hadad, Stefan Hoderlein, Amil Petrin and Robert Sherman (In progress). *Heterogenous Production Functions, Panel Data, and Productivity Dispersion*. Our application is called <i>FHHPS</i>, after the authors' initials.


## Estimation setup

The setup is a **linear panel data** with **two correlated random coefficients**.  It is assumed that the coefficients follow AR(1) processes where the shocks are assumed to be independent of everything in the previous periods.

The application we have in mind is the identification and estimation of first and second moments of Cobb Douglas coefficients of a production function. For more details, please contact Vitor Hadad at baisihad@bc.edu.

## Installing 

To run `fhhps`, you will need Python 3.5 or newer. If you have no current Python installation, the fastest way to get everything up and running is by copying-and-pasting the following commands to your Terminal.

**For MAC OSX and linux**

Type this into your terminal

```
$ git clone https://github.com/halflearned/FHHPS/
$ cd FHHPS/
$ source environment.sh
$ python setup.py develop
```
