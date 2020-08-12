import numpy as np
import torch

from dca.cov_util import (calc_block_toeplitz_logdets,
                          calc_cov_from_cross_cov_mats as calc_cov_from_ccms_dca,
                          project_cross_cov_mats)


def calc_entropy_from_cov(cov):
    """Calculates entropy for a spatiotemporal Gaussian
    process with T N-by-N cross-covariance matrices.

    Parameters
    ----------
    cov : np.ndarray, (N*T, N*T)
        Covariance matrix.

    Returns
    -------
    H : float
        Entropy in nats.
    """
    d = cov.shape[0]
    if isinstance(cov, torch.Tensor):
        logdet = torch.slogdet(cov)[1]
    else:
        logdet = np.slogdet(cov)[1]
    H = 0.5 * (d * (1. + np.log(2. * np.pi)) + logdet)
    return H


def calc_entropy_from_cross_cov_mats(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with T-1 N-by-N cross-covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.

    Returns
    -------
    H : float
        Entropy in nats.
    """
    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    H = calc_entropy_from_cov(cov)
    return H


def calc_entropy_from_cross_cov_mats_block_toeplitz(cross_cov_mats, proj=None):
    """Calculates entropy for a spatiotemporal Gaussian process with T N-by-N
    cross-covariance matrices using the block-Toeplitz algorithm.

    Based on:
    Sowell, Fallaw. "A decomposition of block toeplitz matrices with applications
    to vector time series." 1989a). Unpublished manuscript (1989).

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.

    Returns
    -------
    H : float
        Entropy information in nats.
    """
    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    d = cross_cov_mats_proj.shape[0] * cross_cov_mats_proj.shape[1]
    logdets = calc_block_toeplitz_logdets(cross_cov_mats, proj)
    H = 0.5 * (d * (1. + np.log(2. * np.pi)) + sum(logdets))
    return H


def calc_cov_from_cross_cov_mats(cross_cov_mats, reverse_time=False):
    """Calculates the N*T-by-N*T spatiotemporal covariance matrix based on
    T N-by-N cross-covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    reverse_time : bool
        If `True`, return the covariance for the time reversed `cross_cov_mats`.

    Returns
    -------
    cov : np.ndarray, shape (N*T, N*T)
        Big covariance matrix, stationary in time by construction.
    """
    if reverse_time:
        if isinstance(cross_cov_mats, torch.Tensor):
            cross_cov_mats = torch.transpose(cross_cov_mats, 2, 1)
        else:
            cross_cov_mats = np.transpose(cross_cov_mats, (0, 2, 1))
    cov = calc_cov_from_ccms_dca(cross_cov_mats)
    return cov


def calc_kl_from_cross_cov_mats(cross_cov_mats, proj=None, cholesky=True):
    """Calculates the KL-divergence between the forwards and backwards cross
    covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.
    cholesky : bool
        If `True` and if `cross_cov_mats` is a torch tensor, uses the cholesky
        decomposition to perform the solve operation.

    Returns
    -------
    kl : float
        KL-divergence in nats.
    """
    use_torch = isinstance(cross_cov_mats, torch.Tensor)

    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    covf = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    covr = calc_cov_from_cross_cov_mats(cross_cov_mats_proj, reverse_time=True)

    if use_torch:
        if cholesky:
            covru = torch.cholesky(covr)
            trace = torch.trace(torch.cholesky_solve(covf, covru))
            logdets = torch.slogdet(covr)[1] - torch.slogdet(covf)[1]
            kl = trace + logdets - covf.shape[0]
        else:
            trace = torch.trace(torch.solve(covf, covr)[0])
            logdets = torch.slogdet(covr)[1] - torch.slogdet(covf)[1]
            kl = .5 * (trace + logdets - covf.shape[0])
    else:
        trace = np.trace(np.linalg.solve(covr, covf))
        logdets = np.linalg.slogdet(covr)[1] - np.linalg.slogdet(covf)[1]
        kl = .5 * (trace + logdets - covf.shape[0])

    return kl
