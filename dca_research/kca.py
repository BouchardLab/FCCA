import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
import pdb
import torch
import torch.nn.functional as F

from dca.base import SingleProjectionComponentsAnalysis, ortho_reg_fn, init_coef, ObjectiveWrapper
from dca.cov_util import calc_cross_cov_mats_from_data, calc_cov_from_cross_cov_mats

logging.basicConfig()

# Arrange cross-covariance matrices in Hankel form
def gen_hankel_from_blocks(blocks):

    order = int(blocks.shape[0]/2)
    block_hankel = torch.cat([torch.cat([blocks[i + j + 1, ...] for j in range(order)]) for i in range(order)], dim=1)
    return block_hankel

def gen_toeplitz_from_blocks(blocks):
    
    order = int(blocks.shape[0])
    toeplitz_block_index = lambda idx: blocks[idx, ...] if idx >= 0 else blocks[-1*idx, ...].T

    block_toeplitz = torch.cat([torch.cat([toeplitz_block_index(j - i) 
                                            for j in range(order)], dim=1) 
                                for i in range(order)], dim=0)
    return block_toeplitz

def calc_mmse_from_cross_cov_mats(cross_cov_mats, proj=None, project_mmse=False, return_covs=False):

    T = cross_cov_mats.shape[0] - 1
    N = cross_cov_mats.shape[-1]

    if proj is not None:
        ccm_proj1 = torch.stack([torch.mm(torch.mm(torch.t(proj), cc), proj) for cc in cross_cov_mats[:-1, ...]])
        ccm_proj2 = []

        ccm_proj2 = [torch.mm(torch.t(proj), torch.t(cc)) for cc in cross_cov_mats[1:]]
        ccm_proj2.reverse()
    else:
        ccm_proj1 = cross_cov_mats[:-1]
        ccm_proj2 = [torch.t(cc) for cc in torch.flip(cross_cov_mats)[1:]]
        ccm_proj2.reverse()

    covp = calc_cov_from_cross_cov_mats(ccm_proj1)
    covf = cross_cov_mats[0]
    covpf = torch.cat(ccm_proj2)
    mmse_cov = covf - torch.t(covpf) @ torch.inverse(covp) @ covpf

    if project_mmse:
        mmse_cov = torch.t(proj) @ mmse_cov @ proj

    if return_covs:
        return torch.trace(mmse_cov), covp, covf, covpf
    else:
        return torch.trace(mmse_cov)
    
def build_loss(cross_cov_mats, d, ortho_lambda=1., causal_weights=(1, 0), project_mmse=False):
    """Constructs a loss function which gives the (negative) predictive
    information in the projection of multidimensional timeseries data X onto a
    d-dimensional basis, where predictive information is computed using a
    stationary Gaussian process approximation.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        mutual information is computed.
    d: int
        Number of basis vectors onto which the data X are projected.
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    loss : function
       Loss function which accepts a (flattened) N-by-d matrix, whose
       columns are basis vectors, and outputs the negative predictive information
       corresponding to that projection (plus regularization term).
    """
    N = cross_cov_mats.shape[1]

    def loss(V_flat):
        V = V_flat.reshape(N, d)
        ortho_reg_val = ortho_reg_fn(ortho_lambda, V)

        mmse_fwd = calc_mmse_from_cross_cov_mats(cross_cov_mats, V, project_mmse=project_mmse)
        mmsw_rev = calc_mmse_from_cross_cov_mats(torch.transpose(cross_cov_mats, 1, 2), V, project_mmse=project_mmse)

        return causal_weights[0] * mmse_fwd + causal_weights[1] * mmsw_rev + ortho_reg_val

    return loss


class KalmanComponentsAnalysis(SingleProjectionComponentsAnalysis):
    """Kalman Components Analysis. 

    Runs KCA on multidimensional timeseries data X to discover a projection

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information. When fitting
         a model, the length of the shortest timeseries must be greater than
        T and for good performance should be much greater than T.
    init : str
        Options: "random_ortho", "random", or "PCA"
        Method for initializing the projection matrix.
    n_init : int
        Number of random restarts. Default is 1.
    stride : int
        Number of samples to skip when estimating cross covariance matrices. Settings stride > 1
        will speedup covariance estimation but may reduce the quality of the covariance estimate
        for small datasets.
    chunk_cov_estimate : None or int
        If `None`, cov is estimated from entire time series. If an `int`, cov is estimated
        by chunking up time series and averaging covariances from chucks. This can use less memory
        and be faster for long timeseries. Requires that the length of the shortest timeseries
        in the batch is longer than T * chunk_cov_estimate`.
    tol : float
        Tolerance for stopping optimization. Default is 1e-6.
    ortho_lambda : float
        Coefficient on term that keeps V close to orthonormal.
    verbose : bool
        Verbosity during optimization.
    use_scipy : bool
        Whether to use SciPy or Pytorch L-BFGS-B. Default is True. Pytorch is not well tested.
    device : str
        What device to run the computation on in Pytorch.
    dtype : pytorch.dtype
        What dtype to use for computation.
    rng_or_seed : None, int, or NumPy RandomState
        Random number generator or seed.

    Attributes
    ----------
    T : int
        Default T used for PI.
    T_fit : int
        T used for last cross covariance estimation.
    d : int
        Default d used for fitting the projection.
    d_fit : int
        d used for last projection fit.
    cross covs : torch tensor
        Cross covariance matrices from the last covariance estimation.
    coef_ : ndarray (N, d)
        Projection matrix from fit.
    """
    def __init__(self, d=None, T=None, causal_weights=(1, 0), project_mmse=False, 
                 init="random_ortho", n_init=1, stride=1,
                 chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
                 device="cpu", dtype=torch.float64, rng_or_seed=None):

        super(KalmanComponentsAnalysis,
              self).__init__(d=d, T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol, verbose=verbose,
                             device=device, dtype=dtype, rng_or_seed=rng_or_seed)

        self.causal_weights = causal_weights
        self.project_mmse = project_mmse
        self.ortho_lambda = ortho_lambda
        self.cross_covs = None

    def estimate_data_statistics(self, X, T=None, regularization=None, reg_ops=None):
        """Estimate the cross covariance matrix from data.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        T : int
            T for PI calculation (optional).
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        """
        if T is None:
            T = self.T
        else:
            self.T = T
        start = time.time()
        self._logger.info('Starting cross covariance estimate.')
        if isinstance(X, list) or X.ndim == 3:
            self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        else:
            self.mean_ = X.mean(axis=0, keepdims=True)

        # Estimate only T + 1 autocovariances instead of 2T like in DCA
        cross_covs = calc_cross_cov_mats_from_data(X, self.T + 1, mean=self.mean_,
                                                   chunks=self.chunk_cov_estimate,
                                                   stride=self.stride,
                                                   rng=self.rng,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops,
                                                   logger=self._logger)
        self.cross_covs = torch.tensor(cross_covs, device=self.device, dtype=self.dtype)

        delta_time = round((time.time() - start) / 60., 1)
        self._logger.info('Cross covariance estimate took {:0.1f} minutes.'.format(delta_time))

        return self

    def _fit_projection(self, d=None, T=None, record_V=False):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional). Default is `self.T`. If `T` is set here
            it must be less than or equal to `self.T` or self.estimate_cross_covariance() must
            be called with a larger `T`.
        record_V : bool
            If True, saves a copy of V at each optimization step. Default is False.
        """
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError
        self.d_fit = d
        if T is None:
            T = self.T
        if T < 1:
            raise ValueError
        if T > self.cross_covs.shape[0] - 1:
            raise ValueError('T must less than or equal to the value when ' +
                             '`estimate_cross_covariance()` was called.')
        self.T_fit = T

        if self.cross_covs is None:
            raise ValueError('Call `estimate_cross_covariance()` first.')

        c = self.cross_covs[:T + 1]
        N = c.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)

        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device, dtype=self.dtype)

        def f_params(v_flat, requires_grad=True):
            v_flat_torch = torch.tensor(v_flat,
                                        requires_grad=requires_grad,
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            loss = build_loss(c, d, self.ortho_lambda)(v_torch)
            return loss, v_flat_torch
        objective = ObjectiveWrapper(f_params)

        def null_callback(*args, **kwargs):
            pass

        if self.verbose or record_V:
            if record_V:
                self.V_seq = [V_init]

            def callback(v_flat, objective):
                if record_V:
                    self.V_seq.append(v_flat.reshape(N, d))
                if self.verbose:
                    loss, v_flat_torch = objective.core_computations(v_flat,
                                                                     requires_grad=False)
                    v_torch = v_flat_torch.reshape(N, d)
                    loss = build_loss(c, d, self.ortho_lambda)(v_torch)
                    reg_val = ortho_reg_fn(self.ortho_lambda, v_torch)
                    loss = loss.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    PI = -(loss - reg_val)
                    string = "Loss {}, PI: {} nats, reg: {}"
                    self._logger.info(string.format(str(np.round(loss, 4)),
                                                    str(np.round(PI, 4)),
                                                    str(np.round(reg_val, 4))))

            callback(V_init, objective)
        else:
            callback = null_callback

        opt = minimize(objective.func, V_init.ravel(), method='L-BFGS-B', jac=objective.grad,
                       options={'disp': self.verbose, 'ftol': self.tol},
                       callback=lambda x: callback(x, objective))
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)

        # Return the negative as singleprojectionanalysis takes argmax of scores across initializations
        final_score = -1*calc_mmse_from_cross_cov_mats(c, torch.tensor(V_opt)).detach().cpu().numpy()

        return V_opt, final_score

    def score(self, X=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            cross_covs = self.cross_covs
        else:
            cross_covs = calc_cross_cov_mats_from_data(X, T=self.T + 1)
        return calc_mmse_from_cross_cov_mats(torch.tensor(cross_covs), torch.tensor(self.coef_)).detach().cpu().numpy()
