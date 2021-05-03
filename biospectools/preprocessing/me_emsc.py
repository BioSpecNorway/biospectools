from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD

from biospectools.physics.misc import calculate_complex_n
from biospectools.preprocessing import EMSC
from . import criterions


def calculate_qext_curves(
    nprs: np.ndarray,
    nkks: np.ndarray,
    alpha0: np.ndarray,
    gamma: np.ndarray,
    wavenumbers: np.ndarray,
) -> np.ndarray:
    gamma_nkks = (1 + np.multiply.outer(gamma, nkks)) * (wavenumbers * 100)
    tanbeta = nprs / np.add.outer((1 / gamma.T), nkks)

    beta0 = np.arctan(tanbeta)
    cosB = np.cos(beta0)
    cos2B = np.cos(2.0 * beta0)

    n_alpha = len(alpha0)
    n_gamma = len(gamma)

    q_matrix = np.zeros((n_alpha * n_gamma, len(wavenumbers)))

    for i in range(n_alpha):
        rho = alpha0[i] * gamma_nkks
        rhocosB = cosB / rho
        q = 2.0 + (4 * rhocosB) * (
            -np.exp(-(rho) * (tanbeta))
            * (np.sin((rho) - (beta0)) + np.cos((rho - 2 * beta0)) * rhocosB)
            + cos2B * rhocosB
        )
        q_matrix[i * n_alpha : (i + 1) * n_alpha, :] = q
    return q_matrix


def orthogonalize_qext(qext: np.ndarray, reference: np.ndarray):
    m = np.dot(reference, reference)
    norm = np.sqrt(m)
    rnorm = reference / norm
    s = np.dot(qext, rnorm)
    qext_orthogonalized = qext - s[:, np.newaxis] * rnorm[np.newaxis, :]
    return qext_orthogonalized


def compress_mie_curves(qext_orthogonalized: np.ndarray, num_comp: int) -> np.ndarray:
    svd = TruncatedSVD(
        n_components=num_comp, n_iter=7, random_state=42
    )  # Self.ncomp needs to be specified
    svd.fit(qext_orthogonalized)
    return svd.components_


class ME_EMSC:
    def __init__(
        self,
        reference: np.ndarray = None,
        wavenumbers: np.ndarray = None,
        weights: np.ndarray = None,
        n_components: Optional[int] = None,
        n0: np.ndarray = None,
        a: np.ndarray = None,
        h: float = 0.25,
        max_iter: int = 30,
        tol: float = 1e-4,
        precision: Optional[int] = None,
        verbose: bool = False,
        positive_ref: bool = True,
        resonant: bool = True
    ):

        if reference is None:
            raise ValueError("reference spectrum must be defined")

        if (wavenumbers[1] - wavenumbers[0]) < 0:
            raise ValueError("wn_reference must be ascending")

        self.reference = reference
        self.wavenumbers = wavenumbers
        self.positive_ref = positive_ref
        self.tol = tol
        self.precision = precision
        if self.precision is not None:
            self.tol = np.finfo(float).eps
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(self.reference))
        self.n_components = n_components
        self.verbose = verbose
        self.resonant = resonant
        self.max_iter = max_iter

        self.n0 = n0
        if self.n0 is None:
            self.n0 = np.linspace(1.1, 1.4, 10)
        self.a = a
        if self.a is None:
            self.a = np.linspace(2, 7.1, 10)
        self.h = h

        self.alpha0 = (4 * np.pi * self.a * (self.n0 - 1)) * 1e-6
        self.gamma = (
            self.h
            * np.log(10)
            / (4 * np.pi * 0.5 * np.pi * (self.n0 - 1) * self.a * 1e-6)
        )

        if self.n_components is None:
            self.n_components = self._estimate_n_components()

    def transform(self, X: np.ndarray) -> np.ndarray:
        # wavenumber have to be input as sorted
        # compute average spectrum from the reference
        ref_x = self.reference * self.weights
        if self.positive_ref:
            ref_x[ref_x < 0] = 0

        # For the first iteration, make basic EMSC model
        basic_emsc = EMSC(
            ref_x, self.wavenumbers,
            validate_state=False, rebuild_model=False)

        new_spectra = []
        self.coefs_ = []
        self.residuals_ = []
        self.rmse_ = []
        self.n_iterations_ = []
        for spectrum in X:
            pure_guess = ref_x
            stop_criterion = self._build_stop_cretarion()
            while not stop_criterion:
                try:
                    pure_guess, coefs, res = self._iteration_step(
                        spectrum, pure_guess, basic_emsc)
                    rmse = np.sqrt((1 / len(res[0, :])) * np.sum(res ** 2))
                    stop_criterion.add(rmse, [pure_guess, coefs, res])
                except np.linalg.LinAlgError:
                    stop_criterion.add(np.nan, [np.nan, np.nan, np.nan])
                    stop_criterion.best_idx = -1
                    break
            new_spectra.append(stop_criterion.best_value[0])
            self.coefs_.append(stop_criterion.best_value[1])
            self.residuals_.append(stop_criterion.best_value[2])
            self.rmse_.append(stop_criterion.best_score)
            self.n_iterations_.append(stop_criterion.best_iter)

        self.coefs_ = np.stack(self.coefs_)
        self.residuals_ = np.stack(self.residuals_)
        self.rmse_ = np.stack(self.rmse_)
        self.n_iterations_ = np.stack(self.n_iterations_)

        return np.stack(new_spectra)

    def _iteration_step(
            self,
            spectrum: np.ndarray,
            reference: np.ndarray,
            basic_emsc: EMSC,
    ) -> tuple:
        # scale with basic EMSC:
        reference = basic_emsc.transform(reference[None])[0]
        if np.all(np.isnan(reference)):
            raise np.linalg.LinAlgError()

        # Apply weights
        reference = reference * self.weights

        if self.positive_ref:
            reference[reference < 0] = 0

        # calculate Qext-curves
        if self.resonant:
            # if this should be any point, we need to terminate after
            # 1 iteration for the non-resonant one
            nprs, nkks = calculate_complex_n(reference, self.wavenumbers)
        else:
            npr = np.zeros(len(self.wavenumbers))
            nprs = npr / (self.wavenumbers * 100)
            nkks = np.zeros(len(self.wavenumbers))
        qext = calculate_qext_curves(
            nprs, nkks, self.alpha0, self.gamma, self.wavenumbers)
        qext = orthogonalize_qext(qext, reference)

        badspectra = compress_mie_curves(qext, self.n_components)

        emsc = EMSC(reference=reference, poly_order=0, constituents=badspectra)
        new_spectrum = emsc.transform(spectrum[None])[0]
        # adapt EMSC results to code
        res = emsc.residuals_
        coefs = emsc.coefs_[0, [-1, *range(1, len(badspectra) + 1), 0]]

        return new_spectrum, coefs, res

    def _build_stop_cretarion(self) -> criterions.BaseStopCriterion:
        if self.precision is not None:
            stop_criterion = criterions.MatlabStopCriterion(
                self.max_iter, self.precision)
        else:
            stop_criterion = criterions.TolStopCriterion(
                self.max_iter, self.tol, 1)
        return stop_criterion

    def _estimate_n_components(self):
        nprs, nkks = calculate_complex_n(self.reference, self.wavenumbers)
        qext = calculate_qext_curves(
            nprs, nkks, self.alpha0, self.gamma, self.wavenumbers)
        qext_orthogonalized = orthogonalize_qext(qext, self.reference)
        max_ncomp = len(self.reference) - 1
        svd = TruncatedSVD(n_components=min(max_ncomp, 30), n_iter=7)
        svd.fit(qext_orthogonalized)
        lda = np.array(
            [
                (sing_val ** 2) / (qext_orthogonalized.shape[0] - 1)
                for sing_val in svd.singular_values_
            ]
        )

        explained_var = 100 * lda / np.sum(lda)
        explained_var = np.cumsum(explained_var)
        explained_var_thresh = 99.96
        num_comp = np.argmax(explained_var > explained_var_thresh) + 1
        return num_comp
