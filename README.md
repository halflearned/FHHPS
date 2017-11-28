# FHHPS

Code used in paper Jeremy Fox, Vitor Hadad, Stefan Hoderlein, Amil Petrin and Robert Sherman (In progress). *Heterogenous Production Functions, Panel Data, and Productivity Dispersion*. Our application is called <i>FHHPS</i>, after the authors' initials.

<center>
<img src="figs/scatter.png" width=300>
</center>

## Estimation setup

The setup is a **linear panel data** with **two correlated random coefficients**. 

<center>
<img src="figs/fmla1.png" height = 80>
</centeR>

It is assumed that the coefficients follow AR(1) processes

<center>
<img src="figs/fmla2.png" height = 80>
</center>

where the shocks are assumed to be independent of everything in the previous periods.

The application we have in mind is the identification and estimation of first and second moments of Cobb Douglas coefficients of a production function. For more details, please contact Vitor Hadad at baisihad@bc.edu.

## Installing 

To run `fhhps`, you will need Python 3.5 or newer. If you have no current Python installation, the fastest way to get everything up and running is by copying-and-pasting the following commands to your Terminal.

**For MAC OSX**

```bash
wget http://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda3/bin/activate
echo PATH="$HOME/miniconda3/bin:$PATH" >> ~/.bashrc
pip install numpy pandas matplotlib scikit-learn tqdm
```
If you don't have root privileges, add `--user` to the last line.  Everything else should work without admin permissions.


**For Windows**

Download and run the Miniconda3 <a href="https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe">installer</a>. Then, in the Prompt, type this to install the necessary dependencies.

```bash
pip install numpy pandas matplotlib scikit-learn tqdm
```
Add the option `--user` if you don't have root privileges in your machine.

#### Confirming

Make sure that everything is installed by typing in your Terminal (Mac) or Prompt (Windows):

```bash
python --version
```

You should see:

```bash
Python 3.x.y :: Anaconda custom (x86_64)  # where x >= 5
```

Next, just download the file `fhhps.py` and follow the instructions below. Alternatively, if you have `git` installed, then just type

```bash
git clone https://github.com/halflearned/FHHPS
```
to grab all files in this repository.

## Getting started

The easiest way to run FHHPS is via the command line application. Suppose we are working with a file `fake_data_1.csv`, whose first rows are:


|    X1 |    X2 |    Y1 |    Y2 |   other_column |
|------:|------:|------:|------:|---------------:|
|  1.64 |  2.12 |  9.7  | 12.11 |           0.46 |
|  0.67 | -0.06 | -0.09 |  1.81 |           0.22 |
| -0.46 | -1.49 |  0.28 |  0.48 |           0.13 |
|  2.84 |  1.63 |  8.56 |  9.02 |           0.99 |
| -0.26 | -0.36 |  0.73 |  1.18 |           0.96 |

To have FHHPS automatically detect columns named `X1`, `X2`, `Y1`, `Y2` and run with default parameters, type the following command on your Terminal or Prompt.

```bash
python fhhps.py -f fake_data_1.csv
```
The application will:

+ Run the estimation algorithm once.
+ Bootstrap 100 times.
+ Create a comma-separated file containing the bootstrapped values for all variables.

It will also produce the following files.

+ PDF plots of the point estimates and the density of bootstrapped estimates of shocks and first- and second-period random coefficient moments.
+ LaTeX tables tables tabulating summary statistics conditional moments and bootstrap estimates of shocks, first-period and second-period random coefficient moments.
+ LaTeX table tabulating how many observations were removed during our censoring scheme, and describing how the dependent variable varies between censored and non-censored observations.

By default, the file names will have a timestamp as suffix, to avoid overwriting existing files. You can use a custom suffix with the option `-suffix=[custom-suffix]`.

### Column names

If your file contains custom column names, use the `-X1`... parameters.

```
python fhhps.py -f [file-name].csv -X1=[column-name] -X2=[column-name] -Y1=[column-name] -Y2=[column-name]
```

### Tuning parameters

You can set the following tuning parameters:
  
+ **Shock parameters**
  + `-csh` or `--c_shocks` (default: 4)
  + `-ash` or `--alpha_shocks` (default: 0.2)
  + `-pl` or `poly_order` (default: 2)


+ **Nadaraya-Watson parameters**
  + `-cnw` or `--c_nw` (default: 0.5)
  + `-anw` or `--alpha_nw` (default: 1/6)

+ **Censoring parameters**
  + `-cc1` or `--c1_cens` (default: 1)
  + `-cc2` or `--c2_cens` (default: 1)
  + `-ac1` or `--alpha_cens_1` (default: 0.25)
  + `-ac2` or `--alpha_cens_2` (default: 0.125)

Notes:

+ Parameter `poly_order` refers to the order of polynomial regression to be used when estimating shock parameters. Other parameters as described in the paper.
+ Default values were chosen to match the "best" configuration used in the empirical application in the paper.

## Advanced usage

### Bootstrapping conditional moments

You can also compute conditional moments using the `-X1_POINT` and `-X2_POINT` options. For example, to compute estimates, bootstrap, figures and tables associated with point (X1, X2) = (-1, 1), type

```python
python fhhps.py -f [filename].csv -X1_POINT=-1 -X2_POINT=1
```

**Remark** In our model, shock moments are assumed to be independent of X_1 and X_2, and so are not affected by conditioning. Summary figures and tables will be produced for shocks as well, but they will be the same as for the unconditional case.





## Python API

You can also run FHHPS from within Python. Users of the `statsmodels` or `scikit-learn` packages will be familiar with the syntax. Here's an example.

```python
# Importing. Make sure fhhps.py is in the same directory!
from fhhps import FHHPS

# Initialize the fhhps object
algo = FHHPS(c_shocks = 4,
             alpha_shocks = .2,
             c_nw = .5,
             alpha_nw = 0.167,
             poly_order = 2)

# Add data previously read, e.g., from a csv file into a pandas DataFrame
algo.add_data(X1 = df["X1"],
              X2 = df["X2"],
              Y1 = df["Y1"],
              Y2 = df["Y2"])
              
# Fit the model once
algo.fit()   

# Now bootstrap e.g. 120 times
algo.boostrap(n_iterations = 120)

# Plot bootstrap densities. Output will be a dictionary of matplotlib figure and axes handles
handles = algo.plot_density()
handles["shock_figure"].savefig("shock_figure.pdf")
handles["rc1_figure"].savefig("shock_figure.pdf")
handles["rc2_figure"].savefig("shock_figure.pdf")

# Grab table. Default output is csv.
shock_tab, rc_tab = algo.summary()

# Save tables to CSV
shock_tab.to_csv("shock_tables.csv")
shock_tab.to_csv("shock_tables.csv")
``` 

For more information, read the docs (coming soon).

## Generating fake data

Use the file `make_data.py`. Observations will be drawn jointly Normally with same default parameters as our simulations in the paper, but modifying the original setup is easy. 

For example, to generate 4000 observations with intercept variance equal to 16 and overall correlation 0.3, and save it to the file `myfakedata.py`, simply type

```bash
python make_data.py -n=4000 -out=myfakedata
```

Note any existing files with the same name will be overwritten.


## Reproducing our results

#### Monte carlo simulations

The file `simulations.py` reproduces all the simulations in our paper. Make sure to have `fhhps.py`, `data_utils` and `simulations.py` in the same folder. Then navigate to that folder in your Terminal or Prompt and type 

```python
python simulations.py
```
Let this run for a few hours, then type

```python
python simulations_analysis.py
```
to reproduce our figures and several tables associated with simulation results. 


#### Bootstrap 

The file `bootstrap_coverage.py` shows how close the bootstrap estimates are to 95% coverage in our main example setup. Similarly, the file `bootstrap_conditional_coverage.py` does the same for estimates of conditional moments.

To reproduce our results, type

```bash
python bootstrap_coverage.py 
python bootstrap_coverage_analysis.py
```

and for conditional estimates, type

```bash
python bootstrap_conditional_coverage.py 
python bootstrap_conditional_coverage_analysis.py
```

This should produce tables similar to ours.




