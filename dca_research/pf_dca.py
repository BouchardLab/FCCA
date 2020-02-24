import numpy as np
import scipy.stats
from scipy.optimize import minimize

import torch
import torch.nn.functional as F

from dca.dca import DynamicalComponentsAnalysis, ortho_reg_fn

__all__ = ["PastFutureDynamicalComponentsAnalysis"]


def calc_pi_from_cov(cov_2_T_pi):
    """Calculates the mutual information ("predictive information"
    or "PI") between variables  {1,...,T_pi} and {T_pi+1,...,2*T_pi}, which
    are jointly Gaussian with covariance matrix cov_2_T_pi.
    Parameters
    ----------
    cov_2_T_pi : np.ndarray, shape (2*T_pi, 2*T_pi)
        Covariance matrix.
    Returns
    -------
    PI : float
        Mutual information in nats.
    """

    T_pi = cov_2_T_pi.shape[0] // 2
    use_torch = isinstance(cov_2_T_pi, torch.Tensor)

    cov_pp = cov_2_T_pi[:T_pi, :T_pi]
    cov_ff = cov_2_T_pi[T_pi:, T_pi:]
    if use_torch:
        logdet_pp_pi = torch.slogdet(cov_pp)[1]
        logdet_ff_pi = torch.slogdet(cov_ff)[1]
        logdet_2T_pi = torch.slogdet(cov_2_T_pi)[1]
    else:
        logdet_pp_pi = np.linalg.slogdet(cov_pp_pi)[1]
        logdet_ff_pi = np.linalg.slogdet(cov_ff_pi)[1]
        logdet_2T_pi = np.linalg.slogdet(cov_2_T_pi)[1]

    PI = .5 * (logdet_pp_pi + logdet_ff_pi - logdet_2T_pi)
    return PI


def calc_cov_from_cross_cov_mats(cc_pp, cc_pf, cc_ff):
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

    N = cc_pp.shape[1]
    T = len(cc_pp) + len(cc_ff)
    use_torch = isinstance(cc_pp, torch.Tensor)

    cross_cov_mats_repeated = []
    for ii in range(T):
        for jj in range(T):
            if ii < (T // 2) and jj < (T // 2):
                mat = cc_pp[abs(ii - jj)].T
            elif ii >= (T // 2) and jj >= (T // 2):
                mat = cc_ff[abs(ii - jj)].T
            else:
                mat = cc_pf[abs(ii - jj)-1]
            if ii > jj:
                if use_torch:
                    cross_cov_mats_repeated.append(mat.t())
                else:
                    cross_cov_mats_repeated.append(mat.T)
            else:
                cross_cov_mats_repeated.append(mat)

    if use_torch:
        cov_rows = [torch.cat(cross_cov_mats_repeated[ii*T:(ii+1)*T], dim=1) for ii in range(T)]
        cov = torch.cat(cov_rows, dim=0)
    else:
        cov_rows = [np.concatenate(cross_cov_mats_repeated[ii*T:(ii+1)*T], axis=1) for ii in range(T)]
        cov = np.concatenate(cov_rows, axis=0)

    return cov


def project_cross_cov_mats(cross_cov_mats, proj_past, proj_future):
    """Projects the cross covariance matrices.

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
    cross_cov_mats_proj : ndarray, shape (T, d, d)
        Mutual information in nats.
    """
    if isinstance(cross_cov_mats, torch.Tensor):
        use_torch = True
    elif isinstance(cross_cov_mats[0], torch.Tensor):
        cross_cov_mats = torch.stack(cross_cov_mats)
        use_torch = True
    else:
        use_torch = False

    if use_torch and isinstance(proj_past, np.ndarray):
        proj_past = torch.tensor(proj_past, device=cross_cov_mats.device, dtype=cross_cov_mats.dtype)
    if use_torch and isinstance(proj_future, np.ndarray):
        proj_future = torch.tensor(proj_future, device=cross_cov_mats.device, dtype=cross_cov_mats.dtype)

    T = cross_cov_mats.shape[0] // 2
    if use_torch:
        """
        #$if proj_past.shape[1] >= proj_future.shape[1]:
        cc_p = torch.matmul(proj_past.t().unsqueeze(0), cross_cov_mats)
        cc_pp = torch.matmul(cc_p[:T], proj_future.unsqueeze(0))
        cc_pf = torch.matmul(cc_p, proj_future.unsqueeze(0))
        """
        cc_pp = torch.matmul(proj_past.t().unsqueeze(0),
                             torch.matmul(cross_cov_mats[:T],
                                          proj_past.unsqueeze(0)))
        cc_pf = torch.matmul(proj_past.t().unsqueeze(0),
                             torch.matmul(torch.transpose(cross_cov_mats[1:], 1, 2),
                                          proj_future.unsqueeze(0)))
        cc_ff = torch.matmul(proj_future.t().unsqueeze(0),
                             torch.matmul(cross_cov_mats[:T],
                                          proj_future.unsqueeze(0)))
        """
        else:
            cc_f = torch.matmul(cross_cov_mats, proj_future.unsqueeze(0))
            cc_ff = torch.matmul(proj_future.t().unsqueeze(0), cc_f[:T])
            cc_pf = torch.matmul(proj_past.t().unsqueeze(0), cc_f)
            cc_pp = torch.matmul(proj_past.t().unsqueeze(0),
                                 torch.matmul(cross_cov_mats[:T],
                                              proj_past.unsqueeze(0)))
                                              """
    else:
        cc_pp = []
        cc_ff = []
        cc_pf = []
        for ii in range(T):
            cc = cross_cov_mats[ii]
            cc_pp.append(proj_past.T.dot(cc.dot(proj_past)))
            cc_ff.append(proj_future.T.dot(cc.dot(proj_future)))
        for ii in range(1, 2*T):
            cc = cross_cov_mats[ii]
            cc_pf.append(proj_past.T.dot(cc.T.dot(proj_future)))
        cc_pp = np.stack(cc_pp)
        cc_ff = np.stack(cc_ff)
        cc_pf = np.stack(cc_pf)

    return cc_pp, cc_pf, cc_ff


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj_past=None, proj_future=None):
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
    PI : float
        Mutual information in nats.
    """
    if proj_past is None or proj_future is None:
        raise NotImplementedError
    else:
        cc_pp, cc_pf, cc_ff = project_cross_cov_mats(cross_cov_mats, proj_past, proj_future)
    cov_2_T_pi = calc_cov_from_cross_cov_mats(cc_pp, cc_pf, cc_ff)
    PI = calc_pi_from_cov(cov_2_T_pi)

    return PI


def build_loss(cross_cov_mats, d, ortho_lambda=1.):
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

    def loss(V_past, V_future):
        reg_val = ortho_reg_fn(V_past, ortho_lambda) + ortho_reg_fn(V_future, ortho_lambda)
        return -calc_pi_from_cross_cov_mats(cross_cov_mats, V_past, V_future) + reg_val

    return loss


class PastFutureDynamicalComponentsAnalysis(DynamicalComponentsAnalysis):
    """Past-Future Dynamical Components Analysis.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.
    Parameters
    ----------
    d_past: int
        Number of basis vectors onto which the past data X are projected.
    d_future: int
        Number of basis vectors onto which the future data X are projected.
    T: int
        Size of time windows accross which to compute mutual information.
    init: string
        Options: "random", "PCA"
        Method for initializing the projection matrix.

    """
    def __init__(self, d=None, d_past=None, d_future=None, T=None, init="random_ortho",
                 n_init=1, tol=1e-6, ortho_lambda=10., verbose=False, use_scipy=True,
                 device="cpu", dtype=torch.float64):
        self.d_past = d_past
        if d_past is None:
            self.d_past = d
        self.d_futured = d_future
        if d_future is None:
            self.d_future = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.use_scipy = use_scipy
        self.cross_covs = None

    def _fit_projection(self, d=None, d_past=None, d_future=None, record_V=False):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        record_V : bool
            If True, saves a copy of V at each optimization step. Default is False.
        """
        if d_past is None:
            if d is not None:
                d_past = d
            else:
                d_past = self.d_past
        if d_future is None:
            if d is not None:
                d_future = d
            else:
                d_future = self.d_future

        if self.cross_covs is None:
            raise ValueError('Call estimate_cross_covariance() first.')

        N = self.cross_covs.shape[1]
        if type(self.init) == str:
            if self.init == "random":
                V_past = np.random.normal(0, 1, (N, d_past))
                V_future = np.random.normal(0, 1, (N, d_future))
            elif self.init == "random_ortho":
                V_past = scipy.stats.ortho_group.rvs(N)[:, :d_past]
                V_future = scipy.stats.ortho_group.rvs(N)[:, :d_future]
            elif self.init == "uniform":
                V_past = np.ones((N, d_past)) / np.sqrt(N)
                V_past += np.random.normal(0, 1e-3, V_init.shape)
                V_future = np.ones((N, d_future)) / np.sqrt(N)
                V_future += np.random.normal(0, 1e-3, V_init.shape)
            else:
                raise ValueError
        else:
            raise ValueError
        V_past /= np.linalg.norm(V_past, axis=0, keepdims=True)
        V_future /= np.linalg.norm(V_past, axis=0, keepdims=True)


        c = self.cross_covs
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device, dtype=self.dtype)

        if self.use_scipy:
            if self.verbose or record_V:
                if record_V:
                    self.V_seq = [(V_past, V_future)]

                def callback(v_flat):
                    v_flat_torch = torch.tensor(v_flat,
                                                requires_grad=True,
                                                device=self.device,
                                                dtype=self.dtype)
                    v_past = v_flat_torch[:N*d_past].reshape(N, d_past)
                    v_future = v_flat_torch[N*d_past:].reshape(N, d_future)
                    loss = build_loss(c, d, self.ortho_lambda)(v_past, v_future)
                    reg_val = ortho_reg_fn(v_past, self.ortho_lambda)
                    reg_val = reg_val + ortho_reg_fn(v_future, self.ortho_lambda)
                    loss = loss.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    if record_V:
                        self.V_seq.append((v_past, v_future))
                    if self.verbose:
                        print("PI: {} nats, reg: {}".format(str(np.round(-loss, 4)),
                                                            str(np.round(reg_val, 4))))

                callback(np.concatenate([V_past.ravel(), V_future.ravel()]))
            else:
                callback = None

            def f_df(v_flat):
                v_flat_torch = torch.tensor(v_flat,
                                            requires_grad=True,
                                            device=self.device,
                                            dtype=self.dtype)
                v_past = v_flat_torch[:N*d_past].reshape(N, d_past)
                v_future = v_flat_torch[N*d_past:].reshape(N, d_future)
                loss = build_loss(c, d, self.ortho_lambda)(v_past, v_future)
                loss.backward()
                grad = v_flat_torch.grad
                return (loss.detach().cpu().numpy().astype(float),
                        grad.detach().cpu().numpy().astype(float))
            opt = minimize(f_df, np.concatenate([V_past.ravel(), V_future.ravel()]),
                           method='L-BFGS-B', jac=True,
                           options={'disp': self.verbose, 'ftol': self.tol},
                           callback=callback)
            v_past = opt.x[:N*d_past].reshape(N, d_past)
            v_future = opt.x[N*d_past:].reshape(N, d_future)
        else:
            v = torch.tensor(V_init, requires_grad=True,
                             device=self.device, dtype=self.dtype)
            optimizer = torch.optim.LBFGS([v], max_eval=15000, max_iter=15000,
                                          tolerance_change=self.tol, history_size=10,
                                          line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                loss = build_loss(c, d, self.ortho_lambda)(v)
                loss.backward()
                if self.verbose:
                    reg_val = ortho_reg_fn(v, self.ortho_lambda)
                    loss_no_reg = loss - reg_val
                    pi = -loss_no_reg.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    print("PI: {} nats, reg: {}".format(str(np.round(pi, 4)),
                                                        str(np.round(reg_val, 4))))
                return loss

            optimizer.step(closure)
            v = v.detach().cpu().numpy()

        # Orthonormalize the basis prior to returning it
        V_past = scipy.linalg.orth(v_past)
        V_future = scipy.linalg.orth(v_future)
        final_pi = calc_pi_from_cross_cov_mats(c, V_past, V_future).detach().cpu().numpy()
        return (V_past, V_future), final_pi

    def transform(self, X):
        """Project the data onto the DCA components.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        """
        if isinstance(X, list):
            y = [(Xi - Xi.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X]
        elif X.ndim == 3:
            y = np.stack([(Xi - Xi.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X])
        else:
            y = (X - X.mean(axis=0, keepdims=True)).dot(self.coef_)
        return y

    def score(self):
        """Calculate the PI of the training data for the DCA projection.
        """
        return calc_pi_from_cross_cov_mats(self.cross_covs, self.coef_[0], self.coef_[1])
