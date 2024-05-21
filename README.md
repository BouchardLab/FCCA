# FCCA
Feedback Controllability Components Analysis

## Installation

Assusming you are in the FCCA directory:

```bash
$ conda env update --file environment.yml
$ pip install -e .
```

or:

```bash
$ pip install -e . -r requirements.txt
```

## Usage

There are 2 main hyperparameters associated with the method. The first is $d$, the dimensionality of the projection, and second is $T$, which is the timescale over which the feedback controllability loss is calculated. As the objective function is non-convex, an $n_{init}$ parameter can also be specified which will perform $n_{init}$ optimizations, returning the result that yields the lowest LQG cost.

```python
from FCCA.fcca import LQGComponentsAnalysis as LQGCA
lqgca_model = LQGCA(T=4, d=2, n_init=10).fit(X)
# Projection matrix corresponding to the optimized projection
lqgca_model.coef_
```
Here $X$ is a multivaritate time series with shape (n_samples, n_features).