import numpy as np
import scipy as sp
import torch
from numpy.lib.stride_tricks import as_strided

from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state

def form_lag_matrix(X, T, stride=1, stride_tricks=True, rng=None, writeable=False):
    """Form the data matrix with `T` lags.

    Parameters
    ----------
    X : ndarray (n_time, N)
        Timeseries with no lags.
    T : int
        Number of lags.
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    stride_tricks : bool
        Whether to use numpy stride tricks to form the lagged matrix or create
        a new array. Using numpy stride tricks can can lower memory usage, especially for
        large `T`. If `False`, a new array is created.
    writeable : bool
        For testing. You should not need to set this to True. This function uses stride tricks
        to form the lag matrix which means writing to the array will have confusing behavior.
        If `stride_tricks` is `False`, this flag does nothing.

    Returns
    -------
    X_with_lags : ndarray (n_lagged_time, N * T)
        Timeseries with lags.
    """
    if not isinstance(stride, int) or stride < 1:
        if not isinstance(stride, float) or stride <= 0. or stride >= 1.:
            raise ValueError('stride should be an int and greater than or equal to 1 or a float ' +
                             'between 0 and 1.')
    N = X.shape[1]
    frac = None
    if isinstance(stride, float):
        frac = stride
        stride = 1
    n_lagged_samples = (len(X) - T) // stride + 1
    if n_lagged_samples < 1:
        raise ValueError('T is too long for a timeseries of length {}.'.format(len(X)))
    if stride_tricks:
        X = np.asarray(X, dtype=float, order='C')
        shape = (n_lagged_samples, N * T)
        strides = (X.strides[0] * stride,) + (X.strides[-1],)
        X_with_lags = as_strided(X, shape=shape, strides=strides, writeable=writeable)
    else:
        X_with_lags = np.zeros((n_lagged_samples, T * N))
        for i in range(n_lagged_samples):
            X_with_lags[i, :] = X[i * stride:i * stride + T, :].flatten()
    if frac is not None:
        rng = check_random_state(rng)
        idxs = np.sort(rng.choice(n_lagged_samples, size=int(np.ceil(n_lagged_samples * frac)),
                                  replace=False))
        X_with_lags = X_with_lags[idxs]

    return X_with_lags


def rectify_spectrum(cov, epsilon=1e-5, logger=None):
    """Rectify the spectrum of a covariance matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix
    epsilon : float
        Minimum eigenvalue for the rectified spectrum.
    verbose : bool
        Whethere to print when the spectrum needs to be rectified.
    """
    eigvals = sp.linalg.eigvalsh(cov)
    n_neg = np.sum(eigvals <= 0.)
    if n_neg > 0:
        cov += (-np.min(eigvals) + epsilon) * np.eye(cov.shape[0])
        if logger is not None:
            string = 'Non-PSD matrix, {} of {} eigenvalues were not positive.'
            logger.info(string.format(n_neg, eigvals.size))


def toeplitzify(cov, T, N, symmetrize=True):
    """Make a matrix block-Toeplitz by averaging along the block diagonal.

    Parameters
    ----------
    cov : ndarray (T*N, T*N)
        Covariance matrix to make block toeplitz.
    T : int
        Number of blocks.
    N : int
        Number of features per block.
    symmetrize : bool
        Whether to ensure that the whole matrix is symmetric.
        Optional (default=True).

    Returns
    -------
    cov_toep : ndarray (T*N, T*N)
        Toeplitzified matrix.
    """
    cov_toep = np.zeros((T * N, T * N))
    for delta_t in range(T):
        to_avg_lower = np.zeros((T - delta_t, N, N))
        to_avg_upper = np.zeros((T - delta_t, N, N))
        for i in range(T - delta_t):
            to_avg_lower[i] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]
        avg_lower = np.mean(to_avg_lower, axis=0)
        avg_upper = np.mean(to_avg_upper, axis=0)
        if symmetrize:
            avg_lower = 0.5 * (avg_lower + avg_upper.T)
            avg_upper = avg_lower.T
        for i in range(T - delta_t):
            cov_toep[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N] = avg_lower
            cov_toep[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N] = avg_upper
    return cov_toep


def calc_chunked_cov(X, T, stride, chunks, cov_est=None, rng=None, stride_tricks=True):
    """Calculate an unormalized (by sample count) lagged covariance matrix
    in chunks to save memory.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    stride : int
        The number of time-points to skip between samples.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
    cov_est : ndarray
        Current estimate of unnormalized cov_est to be added to.

    Return
    ------
    cov_est : ndarray
        Current covariance estimate.
    n_samples
        How many samples were used.
    """
    if cov_est is None:
        cov_est = 0.
    n_samples = 0
    if X.shape[0] < T * chunks:
        raise ValueError('Time series is too short to chunk for cov estimation.')
    ends = np.linspace(0, X.shape[0], chunks + 1, dtype=int)[1:]
    start = 0
    for chunk in range(chunks):
        X_with_lags = form_lag_matrix(X[start:ends[chunk]], T, stride=stride,
                                      rng=rng, stride_tricks=stride_tricks)
        start = ends[chunk] - T + 1
        ni_samples = X_with_lags.shape[0]
        cov_est += np.dot(X_with_lags.T, X_with_lags)
        n_samples += ni_samples
    return cov_est, n_samples


def calc_cross_cov_mats_from_cov(cov, T, N):
    """Calculates T N-by-N cross-covariance matrices given
    a N*T-by-N*T spatiotemporal covariance matrix by
    averaging over off-diagonal cross-covariance blocks with
    constant `|t1-t2|`.
    Parameters
    ----------
    N : int
        Numbner of spatial dimensions.
    T: int
        Number of time-lags.
    cov : np.ndarray, shape (N*T, N*T)
        Spatiotemporal covariance matrix.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices.
    """

    use_torch = isinstance(cov, torch.Tensor)

    if use_torch:
        cross_cov_mats = torch.zeros((T, N, N))
    else:
        cross_cov_mats = np.zeros((T, N, N))

    for delta_t in range(T):
        if use_torch:
            to_avg_lower = torch.zeros((T - delta_t, N, N))
            to_avg_upper = torch.zeros((T - delta_t, N, N))
        else:
            to_avg_lower = np.zeros((T - delta_t, N, N))
            to_avg_upper = np.zeros((T - delta_t, N, N))

        for i in range(T - delta_t):
            to_avg_lower[i, :, :] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i, :, :] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]

        avg_lower = to_avg_lower.mean(axis=0)
        avg_upper = to_avg_upper.mean(axis=0)

        if use_torch:
            cross_cov_mats[delta_t, :, :] = 0.5 * (avg_lower + avg_upper.t())
        else:
            cross_cov_mats[delta_t, :, :] = 0.5 * (avg_lower + avg_upper.T)

    return cross_cov_mats


def calc_cross_cov_mats_from_data(X, T, mean=None, chunks=None, stride=1,
                                  rng=None, reg_ops=None,
                                  stride_tricks=True, logger=None, method='toeplitzify'):
    """Compute the N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    stride_tricks : bool
        Whether to use numpy stride tricks in form_lag_matrix. True will use less
        memory for large T.
    logger : logger
        Logger.
    method : str
        'ML' for EM-based maximum likelihood block toeplitz estimation, 'toeplitzify' for naive
        block averaging.

    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """
    if reg_ops is None:
        reg_ops = dict()

    if isinstance(X, list) or X.ndim == 3:
        for Xi in X:
            if len(Xi) <= T:
                raise ValueError('T must be shorter than the length of the shortest ' +
                                 'timeseries.')
        if mean is None:
            mean = np.concatenate(X).mean(axis=0, keepdims=True)
        X = [Xi - mean for Xi in X]
        N = X[0].shape[-1]
        if chunks is None:
            cov_est = np.zeros((N * T, N * T))
            n_samples = 0
            for Xi in X:
                X_with_lags = form_lag_matrix(Xi, T, stride=stride, stride_tricks=stride_tricks,
                                              rng=rng)
                cov_est += np.dot(X_with_lags.T, X_with_lags)
                n_samples += len(X_with_lags)
            cov_est /= (n_samples - 1.)
        else:
            n_samples = 0
            cov_est = np.zeros((N * T, N * T))
            for Xi in X:
                cov_est, ni_samples = calc_chunked_cov(Xi, T, stride, chunks, cov_est=cov_est,
                                                       stride_tricks=stride_tricks, rng=rng)
                n_samples += ni_samples
            cov_est /= (n_samples - 1.)
    else:
        if len(X) <= T:
            raise ValueError('T must be shorter than the length of the shortest ' +
                             'timeseries.')
        if mean is None:
            mean = X.mean(axis=0, keepdims=True)
        X = X - mean
        N = X.shape[-1]
        if chunks is None:
            X_with_lags = form_lag_matrix(X, T, stride=stride, stride_tricks=stride_tricks,
                                          rng=rng)
            cov_est = np.cov(X_with_lags, rowvar=False)
        else:
            cov_est, n_samples = calc_chunked_cov(X, T, stride, chunks,
                                                  stride_tricks=stride_tricks, rng=rng)
            cov_est /= (n_samples - 1.)

    if method == 'toeplitzify':
        cov_est = toeplitzify(cov_est, T, N)

    rectify_spectrum(cov_est, logger=logger)
    cross_cov_mats = calc_cross_cov_mats_from_cov(cov_est, T, N)
    return cross_cov_mats

def cov_from_ccm_(cross_cov_mats):
    """Calculates the N*T-by-N*T spatiotemporal covariance matrix based on
    T N-by-N cross-covariance matrices.

    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.

    Returns
    -------
    cov : np.ndarray, shape (N*T, N*T)
        Big covariance matrix, stationary in time by construction.
    """

    N = cross_cov_mats.shape[1]
    T = len(cross_cov_mats)
    use_torch = isinstance(cross_cov_mats, torch.Tensor)

    cross_cov_mats_repeated = []
    for i in range(T):
        for j in range(T):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)])
            else:
                if use_torch:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)].t())
                else:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)].T)

    if use_torch:
        cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1)
                         for cov_ii in cov_tensor])
    else:
        cov_tensor = np.reshape(np.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = np.concatenate([np.concatenate([cov_ii_jj for cov_ii_jj in cov_ii], axis=1)
                              for cov_ii in cov_tensor])

    return cov


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
    cov = cov_from_ccm_(cross_cov_mats)
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
