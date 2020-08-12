import numpy as np
import scipy.stats
from scipy.optimize import minimize

import torch
import torch.nn.functional as F

from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats
from dca.dca import DynamicalComponentsAnalysis, build_loss, ortho_reg_fn, init_coef
from .lbfgs import fmin_lbfgs

__all__ = ["L1DynamicalComponentsAnalysis"]

class L1DynamicalComponentsAnalysis(DynamicalComponentsAnalysis):
    """Dynamical Components Analysis.

    Runs CCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.
    Parameters
    ----------
    d: int
        Number of basis vectors onto which the data X are projected.
    T: int
        Size of time windows accross which to compute mutual information.
    init: string
        Options: "random", "PCA"
        Method for initializing the projection matrix.

    """
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False, l1_lambda=0., reestimate=False,
                 device="cpu", dtype=torch.float64, rng_or_seed=None):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose=verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None
        self.l1_lambda = l1_lambda
        self.reestimate = reestimate
        if rng_or_seed is None:
            self.rng = np.random
        elif isinstance(rng_or_seed, np.random.RandomState):
            self.rng = rng_or_seed
        else:
            self.rng = np.random.RandomState(rng_or_seed)

    def _fit_projection(self, d=None, record_V=False):
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError
        if self.cross_covs is None:
            raise ValueError('Call estimate_cross_covariance() first.')

        N = self.cross_covs.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)
        v = torch.tensor(V_init, requires_grad=True,
                         device=self.device, dtype=self.dtype)

        c = self.cross_covs
        if not isinstance(c, torch.Tensor):
        	c = torch.tensor(c, device=self.device, dtype=self.dtype)

        if self.verbose or record_V:
            if record_V:
                self.V_seq = [V_init]
            def callback(v_flat, g, fx, xnorm, gnorm, step, k, num_eval, *args):
                v_flat_torch = torch.tensor(v_flat,
                                            requires_grad=True,
                                            device=self.device,
                                            dtype=self.dtype)
                v_torch = v_flat_torch.reshape(N, d)
                #optimizer.zero_grad()
                L1 = np.sum(abs(v_flat)) * self.l1_lambda
                L0_frac = np.sum(np.equal(v_flat, 0.)) / float(v_flat.size)
                loss = build_loss(c, d)(v_torch)
                reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
                loss = loss.detach().cpu().numpy() + L1
                reg_val = reg_val.detach().cpu().numpy()
                PI = -(loss - reg_val)
                if record_V:
                    self.V_seq.append(v_flat.reshape(N, d))
                if self.verbose:
                    string = "Loss: {}, PI: {} nats, reg: {}, L1: {}, L0-frac: {}"
                    print(string.format(str(np.round(loss, 4)),
                                        str(np.round(PI, 4)),
                                        str(np.round(reg_val, 4)),
                                        str(np.round(L1, 4)),
                                        str(np.round(L0_frac, 4))))


            callback(V_init, None, None, None, None, None, None, None)
        else:
            callback = None
        def f_df(v_flat, g, *args):
            v_flat_torch = torch.tensor(v_flat,
                                        requires_grad=True,
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            #optimizer.zero_grad()
            loss = build_loss(c, d)(v_torch)
            loss.backward()
            grad = v_flat_torch.grad
            g[:] = grad.detach().cpu().numpy().astype(float)
            return loss.detach().cpu().numpy().astype(float)
        v = fmin_lbfgs(f_df, V_init.ravel(), orthantwise_c=self.l1_lambda,
                       progress=callback, epsilon=self.tol)
        v = v.reshape(N, d)
        if self.reestimate:
            mask = abs(v).sum(axis=0).astype(bool)
            model = DynamicalComponentsAnalysis(d=d, T=self.T)
            model.fit(X[:, mask])
            V_opt = np.zeros_like(v)
            V_opt[mask] = model.coef_
            final_pi = calc_pi_from_cross_cov_mats(c, V_opt).detach().cpu().numpy()
        else:

            # Orthonormalize the basis prior to returning it
            #V_opt = scipy.linalg.orth(v)
            V_opt = v / np.linalg.norm(v, axis=0, keepdims=True)
            final_pi = calc_pi_from_cross_cov_mats(c, V_opt).detach().cpu().numpy()
        return V_opt, final_pi
