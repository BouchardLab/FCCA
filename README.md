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

```python
$ from FCCA.fcca import LQGComponentsAnalysis as LQGCA
$ LQGCA(T=4, d=2).fit(X)
```

See the docstring for LQGCA or contact me directly at ankit_kumar@berkeley.edu if there are any questions/technical issues.