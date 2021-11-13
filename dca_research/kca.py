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

def calc_mmse_from_cross_cov_mats(cross_cov_mats, proj=None, project_mmse=False, return_cov=False):

    T = cross_cov_mats.shape[0] - 1
    N = cross_cov_mats.shape[-1]

    # Due to differing conventions, we have to take the transpose of cross_cov_mats here, regardless
    # of whether the problem is causal or not
    cross_cov_mats = torch.transpose(cross_cov_mats, 1, 2)

    if proj is not None:
        # Two sided projections
        cross_cov_mats_proj = torch.matmul(proj.t().unsqueeze(0),
                                           torch.matmul(cross_cov_mats,
                                           proj.unsqueeze(0)))
        # One sided projections
        covpf = torch.cat([proj.t() @ cc for cc in cross_cov_mats[1:]])

    else:
        covpf = torch.cat([cc for cc in cross_cov_mats[1:]])

    covp = calc_cov_from_cross_cov_mats(cross_cov_mats_proj[:-1])
    mmse_cov = cross_cov_mats[0] - torch.t(covpf) @ torch.inverse(covp) @ covpf

    if project_mmse and proj is not None:
        mmse_cov = torch.chain_matmul(torch.t(proj), mmse_cov, proj)
    if return_cov:
        return torch.trace(mmse_cov), mmse_cov
    else:
        return torch.trace(mmse_cov)

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
    def __init__(self, causal=True, d=None, T=None, init="random_ortho", project_mmse=True, 
                 n_init=1, stride=1, chunk_cov_estimate=None, tol=1e-6, 
                 ortho_lambda=10., verbose=False, device="cpu", dtype=torch.float64, 
                rng_or_seed=None):

        super(KalmanComponentsAnalysis,
              self).__init__(d=d, T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol, verbose=verbose,
                             device=device, dtype=dtype, rng_or_seed=rng_or_seed)

        self.ortho_lambda = ortho_lambda
        self.cross_covs = None
        self.causal = causal
        self.project_mmse = project_mmse

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

        # Reverse the direction of time to solve an acausal filtering problem
        if not self.causal:
            self.cross_covs = torch.transpose(self.cross_covs, 1, 2)

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
            loss = self.build_loss(c, d, self.ortho_lambda)(v_torch)
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
                    loss = self.build_loss(c, d, self.ortho_lambda)(v_torch)
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
        # Both KCA and OCCA natively minimize a loss
        final_score = -1*self.score(coef=torch.tensor(V_opt))

        return V_opt, final_score

    def score(self, X=None, cross_covs=None, coef=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            if cross_covs is None:                 
                cross_covs = self.cross_covs
        else:
            cross_covs = calc_cross_cov_mats_from_data(X, T=self.T + 1)
            if not self.causal:
                self.cross_covs = torch.transpose(self.cross_covs, 1, 2)

        if coef is None:
            coef = self.coef_

        return calc_mmse_from_cross_cov_mats(torch.tensor(cross_covs), torch.tensor(coef), 
                                             project_mmse=self.project_mmse).detach().cpu().numpy()

    @staticmethod
    def build_loss(cross_cov_mats, d, ortho_lambda=1.):
        """ Constructs loss function for forward/reverse time filtering error

        Parameters
        ----------
        X : np.ndarray, shape (# time-steps, N)
            The multidimensional time series data from which the
            mmse is computed.
        d: int
            Number of basis vectors onto which the data X are projected.
        ortho_lambda : float
            Regularization hyperparameter.
        Returns
        -------
        loss : function
           Loss function which accepts a (flattened) N-by-d matrix, whose
           columns are basis vectors, and outputs the mmse
           corresponding to that projection (plus regularization term).
        """
        N = cross_cov_mats.shape[1]

        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(ortho_lambda, V)
            mmse_loss = calc_mmse_from_cross_cov_mats(cross_cov_mats, V, project_mmse=self.project_mmse)
            return mmse_loss + reg_val

        return loss

class ObserverControllerComponentsAnalysis(KalmanComponentsAnalysis):
    """Observer Controller Components Analysis

    Runs OCCA on multidimensional timeseries data X to discover a projection

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
    def __init__(self, loss_type='product', loss_reg_vals=None,
                 d=None, T=None, init="random_ortho", n_init=1, stride=1,
                 chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
                 device="cpu", dtype=torch.float64, rng_or_seed=None, project_mmse=True):

        super(ObserverControllerComponentsAnalysis,
              self).__init__(d=d, T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol, verbose=verbose,
                             device=device, dtype=dtype, rng_or_seed=rng_or_seed, 
                             ortho_lambda=ortho_lambda)

        self.loss_type = loss_type
        self.loss_reg_vals = loss_reg_vals

    def estimate_data_statistics(self, X, T=None, regularization=None, reg_ops=None):
        """Estimate the cross covariance matrix from data.
        Estimate both forward time and reverse time statistics and concatenate them

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

    def score(self, X=None, cross_covs=None, coef=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            if cross_covs is None:                 
                cross_covs = self.cross_covs
        else:

            # Estimate T cross covariance matrices in the forward direction
            cross_covs = calc_cross_cov_mats_from_data(X, self.T + 1)
            
        if coef is None:
            coef = self.coef_

        if self.loss_type == 'product':
            _, mmse_covf = calc_mmse_from_cross_cov_mats(cross_covs, torch.tensor(coef), project_mmse=self.project_mmse,
                                                         return_cov=True)
            _, mmse_covr = calc_mmse_from_cross_cov_mats(torch.transpose(torch.tensor(cross_covs), 1, 2), torch.tensor(coef),
                                                         project_mmse=self.project_mmse, return_cov=True)

            return torch.trace(torch.matmul(mmse_covf, mmse_covr)).detach().cpu().numpy()

        elif self.loss_type == 'sum':
            mmse_fwd = calc_mmse_from_cross_cov_mats(cross_covs, torch.tensor(coef), project_mmse=self.project_mmse)
            mmse_rev = calc_mmse_from_cross_cov_mats(torch.transpose(torch.tensor(cross_covs), 1, 2), torch.tensor(coef), 
                                                     project_mmse=self.project_mmse)

            return self.loss_reg_vals[0] * mmse_fwd.detach().cpu().numpy() + \
                   self.loss_reg_vals[1] * mmse_rev.detach().cpu().numpy()


    def build_loss(self, cross_cov_mats, d, ortho_lambda=1.):
        """ Constructs loss function for forward/reverse time filtering error

        Parameters
        ----------
        X : np.ndarray, shape (# time-steps, N)
            The multidimensional time series data from which the
            mmse is computed.
        d: int
            Number of basis vectors onto which the data X are projected.
        ortho_lambda : float
            Regularization hyperparameter.
        Returns
        -------
        loss : function
           Loss function which accepts a (flattened) N-by-d matrix, whose
           columns are basis vectors, and outputs the mmse
           corresponding to that projection (plus regularization term).
        """
        N = cross_cov_mats.shape[2]
        if self.loss_type == 'product':

            def loss(V_flat):
                V = V_flat.reshape(N, d)
                orth_reg_val = ortho_reg_fn(ortho_lambda, V)

                _, mmse_covf = calc_mmse_from_cross_cov_mats(cross_cov_mats, V, project_mmse=self.project_mmse,
                                                             return_cov=True)
                _, mmse_covr = calc_mmse_from_cross_cov_mats(torch.transpose(cross_cov_mats, 1, 2), V,
                                                             project_mmse=self.project_mmse, return_cov=True)

                return torch.trace(torch.matmul(mmse_covf, mmse_covr)) + orth_reg_val

        elif self.loss_type == 'sum':

            def loss(V_flat):
                V = V_flat.reshape(N, d)
                orth_reg_val = ortho_reg_fn(ortho_lambda, V)

                mmse_fwd = calc_mmse_from_cross_cov_mats(cross_cov_mats, V, project_mmse=self.project_mmse)
                mmse_rev = calc_mmse_from_cross_cov_mats(torch.transpose(cross_cov_mats, 1, 2), V,
                                                         project_mmse=self.project_mmse)

                return self.loss_reg_vals[0] * mmse_fwd  + self.loss_reg_vals[1] * mmse_rev + orth_reg_val

        return loss