import numpy as np
import torch

from dca.cov_util import (calc_block_toeplitz_logdets,
                          calc_cov_from_cross_cov_mats as calc_cov_from_ccms_dca,
                          project_cross_cov_mats)


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

def calc_mmse_from_cross_cov_mats(cross_cov_mats, proj=None, project_mmse=False, return_covs=False):

    T = cross_cov_mats.shape[0] - 1
    N = cross_cov_mats.shape[-1]

    if proj is not None:
        ccm_proj1 = torch.stack([torch.mm(torch.mm(torch.t(proj), cc), proj) for cc in cross_cov_mats[:-1, ...]])
        ccm_proj2 = []

        ccm_proj2 = [torch.mm(torch.t(proj), torch.t(cc)) for cc in cross_cov_mats[1:]]
        ccm_proj2.reverse()

        covp = calc_cov_from_cross_cov_mats(ccm_proj1)
        covf = cross_cov_mats[0]
        covpf = torch.cat(ccm_proj2)

    else:
        raise ValueError('This block is suspect')
        cov = calc_cov_from_cross_cov_mats(cross_cov_mats)

        covf = cov[-N:, -N:]
        covp = cov[:T*N, :T*N]
        covpf = cov[:T*N, -N:]

    mmse_cov = covf - torch.t(covpf) @ torch.inverse(covp) @ covpf

    if project_mmse:
        mmse_cov = torch.t(proj) @ mmse_cov @ proj

    if return_covs:
        return mmse_cov, covp, covf, covpf
    else:
        return mmse_cov
