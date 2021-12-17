import warnings

import numpy as np
from scipy.linalg import pinv2
from . import _pls
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_consistent_length


class SparsePLSRegression(_pls._PLS):
    """Sparse PLS regression

        PLSRegression that tries to finds sparse weights [1]_.
        It is the same as PLSRegression, but with an additional step of
        *sparsification*. This class inherits from sklearn _PLS, but
        overrides its fit method very close to the base class.

        Parameters
        ----------
        n_components : int, (default 2)
            Number of components to keep.

        sparsity: float or `(n_components,) ndarray` (default 0.3), optional
            Numbers from 0 to 1 that define sparsity of each component.

        sign_stable_weights: bool, (default True)
            If True, then returns weights with deterministic sign

        scale : boolean, (default True)
            whether to scale the data

        max_iter : an integer, (default 500)
            the maximum number of iterations of the NIPALS inner loop (used
            only if algorithm="nipals")

        tol : non-negative real
            Tolerance used in the iterative algorithm default 1e-06.

        copy : boolean, default True
            Whether the deflation should be done on a copy. Let the default
            value to True unless you don't care about side effect

        Attributes
        ----------
        x_weights_ : array, [p, n_components]
            X block weights vectors.

        y_weights_ : array, [q, n_components]
            Y block weights vectors.

        x_loadings_ : array, [p, n_components]
            X block loadings vectors.

        y_loadings_ : array, [q, n_components]
            Y block loadings vectors.

        x_scores_ : array, [n_samples, n_components]
            X scores.

        y_scores_ : array, [n_samples, n_components]
            Y scores.

        x_rotations_ : array, [p, n_components]
            X block to latents rotations.

        y_rotations_ : array, [q, n_components]
            Y block to latents rotations.

        coef_ : array, [p, q]
            The coefficients of the linear model: ``Y = X coef_ + Err``

        n_iter_ : array-like
            Number of iterations of the NIPALS inner loop for each
            component.

        Methods
        -------
        beta(n_components=None) :
            calculates coeffiticients of the linear model taking into
            account number of components. If n_components is None,
            then returns self.coef_ that is coefficients for all
            n_components.

        coefs(n_components=None) : same as beta

        Notes
        -----
        Matrices::

            T: x_scores_
            U: y_scores_
            W: x_weights_
            C: y_weights_
            P: x_loadings_
            Q: y_loadings_

        Are computed such that::

            X = T P.T + Err and Y = U Q.T + Err
            T[:, k] = Xk W[:, k] for k in range(n_components)
            U[:, k] = Yk C[:, k] for k in range(n_components)
            x_rotations_ = W (P.T W)^(-1)
            y_rotations_ = C (Q.T C)^(-1)

        where Xk and Yk are residual matrices at iteration k.

        For each component k, find weights u, v that optimizes:
        ``max corr(Xk u, Yk v) * std(Xk u) std(Yk u)``, such that ``|u| = 1``

        Note that it maximizes both the correlations between the scores and the
        intra-block variances.

        The residual matrix of X (Xk+1) block is obtained by the deflation on
        the current X score: x_score.

        The residual matrix of Y (Yk+1) block is obtained by deflation on the
        current X score. This performs the PLS regression known as PLS2. This
        mode is prediction oriented.

        References
        ----------

        .. [1] Karaman, İbrahim, et al. *Sparse multi-block PLSR
               for biomarker discovery when integrating data from
               LC–MS and NMR metabolomics.*
               Metabolomics 11.2 (2015): 367-379.

        .. [2] Karaman, İbrahim, et al. *Comparison of Sparse and
               Jack-knife partial least squares regression methods
               for variable selection.* Chemometrics and Intelligent
               Laboratory Systems 122 (2013): 65-77.

        See also
        --------
        sklearn.cross_decomposition.PLSRegression
        """

    def __init__(self, n_components=2, sparsity=0.3, sign_stable_weights=True,
                 scale=False, max_iter=500, tol=1e-6, copy=True):
        super().__init__(
            n_components, scale=scale, algorithm='svd',
            deflation_mode="regression", max_iter=max_iter, tol=tol, copy=copy)

        if isinstance(sparsity, float):
            self.sparsity = np.full(n_components, sparsity)
        else:
            self.sparsity = np.asarray(sparsity)
        assert self.sparsity.ndim == 1 and len(self.sparsity) == n_components, \
            'sparsity must be 1D array of shape (n_components,)'

        self.sign_stable_weights = sign_stable_weights
        self._is_fitted = False
        self.coefs = self.beta

    def beta(self, n_components: int = None):
        """
        The coefficients of the linear model: ``Y = X coef_ + Err``

        Parameters
        ----------
        n_components : `int`, optional
        number of components to take into account

        Returns
        -------
        beta coefficients: `(p, q) ndarray`
        """
        if not self._is_fitted:
            raise RuntimeError('Sparse PLS model must be fitter to access beta')

        if n_components is None:
            return self.coef_

        if n_components <= 0:
            raise ValueError('n_components must be > 0')

        beta = np.dot(self.x_rotations_[:, :n_components],
                      self.y_loadings_[:, :n_components].T)
        beta *= self.y_std_
        return beta

    def fit(self, X, Y):
        """Fit model to data.

        This is just a copy of original fit method of sklearn's PLS with added
        sparsity option.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        """
        X, Y = self._validate_and_clean_state(X, Y)
        n, p = X.shape
        q = Y.shape[1]

        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _pls._center_scale_xy(X, Y, self.scale))
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
        Y_eps = np.finfo(Yk.dtype).eps
        double_eps = np.finfo(np.double).eps
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break

            # 1) weights estimation (inner loop)
            # -----------------------------------
            x_weights, y_weights = self._estimate_weights(Xk, Yk, Y_eps)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            if self.sign_stable_weights:
                x_weights, y_weights = _pls.svd_flip(x_weights, y_weights.T)
                y_weights = y_weights.T

            if self.sparsity is not None:
                M = np.dot(Xk.T, Yk)
                x_weights, y_weights = self._find_sparse_weights(
                    M, self.sparsity[k], p, x_weights, y_weights)

            x_scores, y_scores = self._compute_scores(
                Xk, Yk, x_weights, y_weights)
            null_variance = np.dot(x_scores.T, x_scores) < double_eps
            if null_variance:
                warnings.warn('X scores are null at iteration %s' % k)
                break

            # 2) Deflation (in place)
            # ----------------------
            Xk, x_loadings = self._deflate_in_place(Xk, x_scores)
            if self.deflation_mode == "canonical":
                Yk, y_loadings = self._deflate_in_place(Yk, y_scores)
            elif self.deflation_mode == "regression":
                Yk, y_loadings = self._deflate_in_place(Yk, x_scores)

            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                      check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        if True or self.deflation_mode == "regression":
            # FIXME what's with the if?
            # Estimate regression coefficient
            # Regress Y on T
            # Y = TQ' + Err,
            # Then express in function of X
            # Y = X W(P'W)^-1Q' + Err = XB + Err
            # => B = W*Q' (p x q)
            self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
            self.coef_ = self.coef_ * self.y_std_
        self._is_fitted = True
        return self

    def _soft_thresholding(self, loadings: np.ndarray, lmbd):
        return np.sign(loadings) * (np.abs(loadings) - lmbd).clip(min=0)

    def _find_sparse_weights(self, M, sparsity, p, w_old, c_old):
        ite = 0
        maxiter = 5000
        converged = False
        while not converged and ite < maxiter:
            tmp = M @ c_old
            temp = np.sort(np.abs(tmp), axis=0)
            index = int(round(sparsity * p))
            if index == 0:
                lmbd = 0
            else:
                lmbd = (temp[index - 1]+temp[index]) / 2
            w_new = self._soft_thresholding(tmp, lmbd)
            w_new = w_new / np.linalg.norm(w_new)

            c_new = M.T @ w_new
            c_new = c_new / np.linalg.norm(c_new)

            converged = max(abs(w_old-w_new)) < 1e-12
            w_old, c_old = w_new, c_new
            ite += 1

        if ite == maxiter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)

        return w_new, c_new

    def _validate_and_clean_state(self, X, Y):
        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = self._validate_data(X, dtype=np.float64, copy=self.copy,
                                ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if self.n_components < 1 or self.n_components > X.shape[1]:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        if self.algorithm not in ("svd", "nipals"):
            raise ValueError("Got algorithm %s when only 'svd' "
                             "and 'nipals' are known" % self.algorithm)
        if self.algorithm == "svd" and self.mode == "B":
            raise ValueError('Incompatible configuration: mode B is not '
                             'implemented with svd algorithm')
        if self.deflation_mode not in ["canonical", "regression"]:
            raise ValueError('The deflation mode is unknown')

        return X, Y

    def _estimate_weights(self, Xk, Yk, Y_eps):
        if self.algorithm == "nipals":
            # Replace columns that are all close to zero with zeros
            Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
            Yk[:, Yk_mask] = 0.0

            x_weights, y_weights, n_iter_ = \
                _pls._nipals_twoblocks_inner_loop(
                    X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                    tol=self.tol, norm_y_weights=self.norm_y_weights)
            self.n_iter_.append(n_iter_)
        elif self.algorithm == "svd":
            x_weights, y_weights = _pls._svd_cross_product(X=Xk, Y=Yk)

        return x_weights, y_weights

    def _compute_scores(self, Xk, Yk, x_weights, y_weights):
        x_scores = np.dot(Xk, x_weights)
        if self.norm_y_weights:
            y_ss = 1
        else:
            y_ss = np.dot(y_weights.T, y_weights)
        y_scores = np.dot(Yk, y_weights) / y_ss
        return x_scores, y_scores

    def _deflate_in_place(self, residuals, scores):
        # Possible memory footprint reduction may done here: in order to
        # avoid the allocation of a data chunk for the rank-one
        # approximations matrix which is then subtracted to Xk, we suggest
        # to perform a column-wise deflation.
        #
        # - regress residuals on scores
        loadings = np.dot(residuals.T, scores) / np.dot(scores.T, scores)
        # - subtract rank-one approximations to obtain remainder matrix
        residuals -= np.dot(scores, loadings.T)
        return residuals, loadings
