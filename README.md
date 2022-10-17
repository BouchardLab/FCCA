# FCCA
Feedback Controllability Components Analysis

## Installation
FCCA requires DynamicalComponentsAnalysis, another dimensionality reduction method developed in the Bouchard Lab.


To install this, you can clone the repository and `cd` into the DynamicalComponentsAnalysis folder.

```bash
# use ssh
$ git clone git@github.com:BouchardLab/DynamicalComponentsAnalysis.git
# or use https
$ git clone https://github.com/BouchardLab/DynamicalComponentsAnalysis.git
$ cd DynamicalComponentsAnalysis
```

If you are installing into an active conda environment, you can run

```bash
$ conda env update --file environment.yml
$ pip install -e .
```

If you are installing with `pip` you can run

```bash
$ pip install -e . -r requirements.txt
```

Similar steps can be followed to then install the FCCA package:

```bash
# use ssh
$ git clone git@github.com:BouchardLab/FCCA.git
# or use https
$ git clone https://github.com/BouchardLab/FCCA.git
$ cd FCCA
```

Then:

```bash
$ conda env update --file environment.yml
$ pip install -e .
```

or:

```bash
$ pip install -e . -r requirements.txt
```

## Usage

There are 2 main hyperparameters associated with the method. The first is $d$, the dimensionality of the projection, and second is $T$, which is the timescale over which the feedback controllability loss is calculated. As the objective function is non-convex, an $n_init$ parameter can also be specified which will perform $n_init$ optimizations, returning the result that yields the lowest LQG cost.

```python
from FCCA.fcca import LQGComponentsAnalysis as LQGCA
lqgca_model = LQGCA(T=4, d=2, n_init=10).fit(X)
# Projection matrix corresponding to the optimized projection
lqgca_model.coef_
```
Here $X$ is a multivaritate time series with shape (n_samples, n_features). See the docstring for LQGCA and contact me directly at ankit_kumar@berkeley.edu if there are any questions/technical issues.