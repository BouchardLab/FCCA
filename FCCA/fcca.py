import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from .base import SingleProjectionComponentsAnalysis, ortho_reg_fn, init_coef, ObjectiveWrapper
from .cov_util import calc_mmse_from_cross_cov_mats, calc_cross_cov_mats_from_data

logging.basicConfig()
    
def build_loss(ccm_fwd, ccm_rev, d, ortho_lambda=1., project_mmse=False):

    def loss(V_flat):

        N = ccm_fwd.shape[1]
        V = V_flat.reshape(N, d)
        ortho_reg_val = ortho_reg_fn(ortho_lambda, V)
        mmse_fwd = calc_mmse_from_cross_cov_mats(ccm_fwd, V, project_mmse=project_mmse)    

        # In the reverse time direction, the readout is taken to be y = C Pi x_a = C x. 
        # This is implemented here by scaling V by ccm_fwd[0]
        V_rev = torch.matmul(ccm_fwd[0], V)
        mmse_rev = calc_mmse_from_cross_cov_mats(ccm_rev, V_rev, project_mmse=project_mmse)

        return torch.trace(torch.matmul(mmse_fwd, mmse_rev)) + ortho_reg_val

    return loss

class LQGComponentsAnalysis(SingleProjectionComponentsAnalysis):
    """LQG Components Analysis 

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
    def __init__(self, d=None, T=None,  
                 init="random_ortho", n_init=1, stride=1,
                 chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
                 device="cpu", dtype=torch.float64, rng_or_seed=None):

        super(LQGComponentsAnalysis,
              self).__init__(d=d, T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol, verbose=verbose,
                             device=device, dtype=dtype, rng_or_seed=rng_or_seed)


        self.ortho_lambda = ortho_lambda
        self.cross_covs = None

        # These should not be changed
        self.project_mmse = False
        self.normalize_reverse = True
        # Whether to only operate on the marginal autocorrelations (no cross-correlations)
        self.marginal_only = False

    def _estimate_data_statistics(self, X, T, regularization=None, reg_ops=None):

        if isinstance(X, list) or X.ndim == 3:
            self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        else:
            self.mean_ = X.mean(axis=0, keepdims=True)


        cross_covs = calc_cross_cov_mats_from_data(X, self.T + 1, mean=self.mean_,
                                                   chunks=self.chunk_cov_estimate,
                                                   stride=self.stride,
                                                   rng=self.rng,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops,
                                                   logger=self._logger)
        if self.normalize_reverse:
            # Normalize X by its variance an reverse the direction of time
            X = X @ np.linalg.inv(cross_covs[0])
            X = X[::-1, :]
            cross_covs_rev = calc_cross_cov_mats_from_data(X, self.T + 1, mean=self.mean_,
                                                       chunks=self.chunk_cov_estimate,
                                                       stride=self.stride,
                                                       rng=self.rng,
                                                       regularization=regularization,
                                                       reg_ops=reg_ops,
                                                       logger=self._logger)
        else:
            cross_covs_rev = np.transpose(cross_covs, 1, 2)

        # Set each matrix to be diagonal-
        if self.marginal_only:
            cross_covs = np.array([np.diag(np.diag(c)) for c in cross_covs])
            cross_covs_rev = np.array([np.diag(np.diag(c)) for c in cross_covs_rev])

        cross_covs = torch.tensor(cross_covs, device=self.device, dtype=self.dtype)
        cross_covs_rev = torch.tensor(cross_covs_rev, device=self.device, dtype=self.dtype)

        return cross_covs, cross_covs_rev

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

        cross_covs, cross_covs_rev = self._estimate_data_statistics(X, T, regularization, reg_ops)
        self.cross_covs = cross_covs
        self.cross_covs_rev = cross_covs_rev

        delta_time = round((time.time() - start) / 60., 1)
        self._logger.info('Cross covariance estimate took {:0.1f} minutes.'.format(delta_time))

        return self

    def fit_projection(self, d=None, T=None, n_init=None):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional). Default is `self.T`. If `T` is set here
            it must be less than or equal to `self.T` or self.estimate_cross_covariance() must
            be called with a larger `T`.
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        scores = []
        coefs = []
        for ii in range(n_init):
            start = time.time()
            self._logger.info('Starting projection fig {} of {}.'.format(ii + 1, n_init))
            coef, score = self._fit_projection(d=d, T=T)
            delta_time = round((time.time() - start) / 60., 1)
            self._logger.info('Projection fit {} of {} took {:0.1f} minutes.'.format(ii + 1,
                                                                                     n_init,
                                                                                     delta_time))
            scores.append(score)
            coefs.append(coef)
        idx = np.argmax(scores)
        self.scores = scores
        self.coef_ = coefs[idx]


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
        crev = self.cross_covs_rev[:T + 1]

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

            loss = build_loss(c, crev, d, ortho_lambda=self.ortho_lambda,
                              project_mmse=self.project_mmse)(v_torch)            
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
                    loss = build_loss(c, crev, d, ortho_lambda=self.ortho_lambda,
                                    project_mmse=self.project_mmse)(v_torch)            
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

        final_score = -1*self.score(V_opt)

        return V_opt, final_score

    def score(self, coef=None, X=None):
        """Calculate the controllability score associated with the projection

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            ccm_fwd = self.cross_covs
            ccm_rev = self.cross_covs_rev
        else:
            ccm_fwd, ccm_rev = self._estimate_data_statistics(X, self.T)

        if coef is None:
            coef = self.coef_
        coef = torch.tensor(coef)
        loss = build_loss(ccm_fwd, ccm_rev, coef.shape[1], ortho_lambda=0, project_mmse=self.project_mmse)(coef)

        return loss