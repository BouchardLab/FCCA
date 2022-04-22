import logging, time
import numpy as np 
from scipy.optimize import minimize

import torch

from .base import ortho_reg_fn, init_coef, ObjectiveWrapper
from dca.dca import DynamicalComponentsAnalysis

def build_loss(cross_cov_mats, d, ortho_lambda=1., block_toeplitz=False):
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

    if block_toeplitz:
        def loss(V, W):
            return -calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, V, W)
    else:
        def loss(V_flat, W_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(ortho_lambda, V)
            return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss


# Same as DCA, but allow the left and right projection matrices to be asymmetrical
class ObliqueDCA(DynamicalComponentsAnalysis):

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
        if (2 * T) > self.cross_covs.shape[0]:
            raise ValueError('T must less than or equal to the value when ' +
                             '`estimate_cross_covariance()` was called.')
        self.T_fit = T

        if self.cross_covs is None:
            raise ValueError('Call `estimate_cross_covariance()` first.')

        c = self.cross_covs[:2 * T]
        N = c.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)
        W_init = init_coef(N, d, self.rng, self.init)

        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device, dtype=self.dtype)

        def f_params(vw_flat, requires_grad=True):
        	vw_flat_torch = torch.tensor(wv_flat,
                                        requires_grad=requires_grad,
                                        device=self.device,
                                        dtype=self.dtype)

        	v_flat, w_flat = torch.splot(wv_flat_torch, 2)

            v_torch = v_flat_torch.reshape(N, d)
            w_torch = w_flat_torch.reshape(N, d)
            loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch, w_torch)

            return loss, vw_flat_torch

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
                    loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
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
        final_pi = calc_pi_from_cross_cov_mats(c, V_opt).detach().cpu().numpy()
        return V_opt, final_pi
