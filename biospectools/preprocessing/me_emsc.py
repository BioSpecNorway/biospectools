from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD

from biospectools.physics.misc import calculate_complex_n
from biospectools.preprocessing import EMSC


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
    badspectra = svd.components_[0:num_comp, :]
    return badspectra


def cal_ncomp(
    reference: np.ndarray,
    wavenumbers: np.ndarray,
    explained_var_lim: float,
    alpha0: np.ndarray,
    gamma: np.ndarray,
) -> int:
    nprs, nkks = calculate_complex_n(reference, wavenumbers)
    qext = calculate_qext_curves(nprs, nkks, alpha0, gamma, wavenumbers)
    qext_orthogonalized = orthogonalize_qext(qext, reference)
    max_ncomp = reference.shape[0] - 1
    svd = TruncatedSVD(n_components=min(max_ncomp, 30), n_iter=7, random_state=42)
    svd.fit(qext_orthogonalized)
    lda = np.array(
        [
            (sing_val ** 2) / (qext_orthogonalized.shape[0] - 1)
            for sing_val in svd.singular_values_
        ]
    )

    explained_var = 100 * lda / np.sum(lda)
    explained_var = np.cumsum(explained_var)
    num_comp = np.argmax(explained_var > explained_var_lim) + 1
    return num_comp


class ME_EMSC:
    def __init__(
        self,
        reference: np.ndarray = None,
        wavenumbers: np.ndarray = None,
        weights: np.ndarray = None,
        ncomp: int = 0,
        n0: np.ndarray = np.linspace(1.1, 1.4, 10),
        a: np.ndarray = np.linspace(2, 7.1, 10),
        h: float = 0.25,
        max_iter: int = 30,
        tol: float = 1e-4,
        precision: Optional[int] = None,
        verbose: bool = False,
        positive_ref: bool = True,
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
            self.precision = precision
            self.tol = np.finfo(float).eps
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(self.reference))
        self.ncomp = ncomp
        self.verbose = verbose
        self.max_iter = max_iter

        self.n0 = n0
        self.a = a
        self.h = h

        self.alpha0 = (4 * np.pi * self.a * (self.n0 - 1)) * 1e-6
        self.gamma = (
            self.h
            * np.log(10)
            / (4 * np.pi * 0.5 * np.pi * (self.n0 - 1) * self.a * 1e-6)
        )

        explained_variance = 99.96
        if self.ncomp == 0:
            self.ncomp = cal_ncomp(
                self.reference,
                self.wavenumbers,
                explained_variance,
                self.alpha0,
                self.gamma,
            )
        else:
            self.explained_variance = False

    def transform(self, X: np.ndarray) -> tuple:
        # wavenumber have to be input as sorted
        # compute average spectrum from the reference
        ref_x = self.reference * self.weights
        if self.positive_ref:
            ref_x[ref_x < 0] = 0

        resonant = True  # Possibility for using the 2008 version

        if resonant:
            # if this should be any point, we need to terminate after
            # 1 iteration for the non-resonant one
            nprs, nkks = calculate_complex_n(ref_x, self.wavenumbers)
        else:
            npr = np.zeros(len(self.wavenumbers))
            nprs = npr / (self.wavenumbers * 100)
            nkks = np.zeros(len(self.wavenumbers))

        # For the first iteration, make basic EMSC model
        basic_emsc = EMSC(
            ref_x, self.wavenumbers,
            validate_state=False, rebuild_model=False)

        # Calculate scattering curves for ME-EMSC
        qext = calculate_qext_curves(
            nprs, nkks, self.alpha0, self.gamma, self.wavenumbers
        )
        qext = orthogonalize_qext(qext, ref_x)
        badspectra = compress_mie_curves(qext, self.ncomp)

        emsc = EMSC(reference=ref_x, poly_order=0, constituents=badspectra)
        new_spectra = emsc.transform(X)
        # adapt EMSC results to code
        res = emsc.residuals_
        coefs = emsc.coefs_[:, [-1, *range(1, len(badspectra) + 1), 0]]

        if self.max_iter == 1:
            number_of_iterations = np.ones([1, new_spectra.shape[0]])
            rmse_all = np.sqrt((res ** 2).sum(axis=-1) / res.shape[1])
            return new_spectra, coefs, res, rmse_all, number_of_iterations

        # Iterate
        new_spectra, coefs, residuals, rmse_all, number_of_iterations = self._iterate(
            X, new_spectra, coefs, res, basic_emsc, self.alpha0, self.gamma
        )
        return new_spectra, coefs, residuals, rmse_all, number_of_iterations

    def _iterate(
            self,
            spectra: np.ndarray,
            corrected_first_iter: np.ndarray,
            coefs_first_iter: np.ndarray,
            residual_first_iter: np.ndarray,
            basic_emsc: EMSC,
            alpha0: np.ndarray,
            gamma: np.ndarray,
    ) -> tuple:
        new_spectra = np.full(corrected_first_iter.shape, np.nan)
        number_of_iterations = np.full(spectra.shape[0], np.nan)
        coefs = np.full((spectra.shape[0], self.ncomp + 2), np.nan)
        residuals = np.full(spectra.shape, np.nan)
        rmse_all = np.full([spectra.shape[0]], np.nan)
        N = corrected_first_iter.shape[0]
        for i in range(N):
            if self.verbose:
                print(
                    "Corrected spectra ["
                    + int((i / N) * 20) * "#"
                    + int(((N - i - 1) / N) * 20) * " "
                    + f"] [{i}/{N}]",
                    end="\r",
                )
            corr_spec = corrected_first_iter[i]
            prev_spec = corr_spec
            prev_coefs = coefs_first_iter[i]
            raw_spec = spectra[i, :]
            raw_spec = raw_spec.reshape(1, -1)
            rmse_list = [
                np.sqrt(
                    (1 / len(residual_first_iter[i, :]))
                    * np.sum(residual_first_iter[i, :] ** 2)
                )
            ]
            for iter_number in range(2, self.max_iter + 1):
                try:
                    new_spec, new_coefs, res = self._iteration_step(
                        raw_spec,
                        corr_spec,
                        basic_emsc,
                        alpha0,
                        gamma,
                    )
                except np.linalg.LinAlgError:
                    new_spectra[i, :] = np.full(
                        [raw_spec.shape[1]], np.nan
                    )
                    coefs[i] = np.full(self.ncomp + 2, np.nan)
                    residuals[i, :] = np.full(raw_spec.shape, np.nan)
                    rmse_all[i] = np.nan
                    break
                corr_spec = new_spec[0, :]
                rmse = np.sqrt((1 / len(res[0, :])) * np.sum(res ** 2))
                rmse_list.append(rmse)

                # Stop criterion
                if iter_number == self.max_iter:
                    new_spectra[i, :] = corr_spec
                    number_of_iterations[i] = iter_number
                    coefs[i] = new_coefs
                    residuals[i, :] = res
                    rmse_all[i] = rmse_list[-1]
                    break
                elif iter_number > 2:
                    if self.precision is None:
                        pp_rmse, p_rmse, rmse = rmse_list[-3:]
                    else:
                        pp_rmse, p_rmse, rmse = [round(r, self.precision)
                                                 for r in rmse_list[-3:]]

                    if rmse > p_rmse:
                        new_spectra[i, :] = prev_spec
                        coefs[i] = prev_coefs
                        number_of_iterations[i] = iter_number - 1
                        rmse_all[i] = rmse_list[-2]
                        break
                    elif (p_rmse - rmse <= self.tol
                          and pp_rmse - rmse <= self.tol):
                        new_spectra[i, :] = corr_spec
                        number_of_iterations[i] = iter_number
                        coefs[i] = new_coefs
                        residuals[i, :] = res
                        rmse_all[i] = rmse_list[-1]
                        break

        if self.verbose:
            print(f"\n ----- Finished correcting {N} spectra ----- \n")
        return new_spectra, coefs, residuals, rmse_all, number_of_iterations

    def _iteration_step(
            self,
            spectrum: np.ndarray,
            reference: np.ndarray,
            basic_emsc: EMSC,
            alpha0: np.ndarray,
            gamma: np.ndarray,
    ) -> tuple:
        # scale with basic EMSC:
        reference = basic_emsc.transform(reference[None])[0]
        if np.all(np.isnan(reference)):
            raise np.linalg.LinAlgError()

        # Apply weights
        reference = reference * self.weights

        # set negative parts to zero
        nonzero_reference = reference.copy()
        nonzero_reference[nonzero_reference < 0] = 0

        if self.positive_ref:
            reference = nonzero_reference

        # calculate Qext-curves
        nprs, nkks = calculate_complex_n(nonzero_reference, self.wavenumbers)
        qext = calculate_qext_curves(nprs, nkks, alpha0, gamma, self.wavenumbers)
        qext = orthogonalize_qext(qext, reference)

        badspectra = compress_mie_curves(qext, self.ncomp)

        emsc = EMSC(reference=reference, poly_order=0, constituents=badspectra)
        new_spectrum = emsc.transform(spectrum)
        # adapt EMSC results to code
        res = emsc.residuals_
        coefs = emsc.coefs_[:, [-1, *range(1, len(badspectra) + 1), 0]]

        return new_spectrum, coefs, res
