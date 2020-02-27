import pytest
import numpy as np
import torch
from numpy.testing import assert_allclose

from dca.synth_data import gen_lorenz_data
from dca.data_util import form_lag_matrix
from dca.cov_util import calc_cross_cov_mats_from_data
from dca_research.pf_dca import (PastFutureDynamicalComponentsAnalysis,
                                 calc_pi_from_cov,
                                 calc_cov_from_cross_cov_mats,
                                 project_cross_cov_mats,
                                 calc_pi_from_cross_cov_mats,
                                 init_coef)


@pytest.fixture
def lorenz_dataset():
    rng = np.random.RandomState(20200129)
    T, d = 6, 4
    X = rng.randn(20000, d)
    XL = gen_lorenz_data(20000)
    X[:, :3] += XL
    ccms = calc_cross_cov_mats_from_data(X, T=T)
    return T, d, X, ccms


def test_projected_cov_calc(lorenz_dataset):
    """Test the project_cross_cov_mats function by also directly projecting
    the data."""
    rng = np.random.RandomState(20200227)
    T, N, X, ccms = lorenz_dataset
    dp = 2
    df = 3
    Vp = init_coef(N, dp, rng, 'random_ortho')
    Vf = init_coef(N, df, rng, 'random_ortho')
    tVp = torch.tensor(Vp)
    tVf = torch.tensor(Vf)

    ccms = calc_cross_cov_mats_from_data(X, T)
    tccms = torch.tensor(ccms)
    pccms = project_cross_cov_mats(ccms, Vp, Vf)
    cov = calc_cov_from_cross_cov_mats(*pccms)

    XL = form_lag_matrix(X, T)
    T2 = T // 2
    big_V = np.zeros((T * N, T2 * dp + T2 * df))
    for ii in range(T2):
        big_V[ii * N:(ii + 1) * N, ii * dp:(ii + 1) * dp] = Vp
        big_V[(ii + T2) * N:(ii + T2 + 1) * N,
              T2 * dp + ii * df:T2 * dp + (ii + 1) * df] = Vf
    Xp = XL.dot(big_V)
    cov2 = np.cov(Xp, rowvar=False)
    assert_allclose(cov, cov2, rtol=3e-3)

    tpccms = project_cross_cov_mats(tccms, tVp, tVf)
    tcov = calc_cov_from_cross_cov_mats(*tpccms)
    assert torch.allclose(tcov, torch.tensor(cov2), rtol=3e-3)
    assert_allclose(tcov.numpy(), cov2, rtol=3e-3)
