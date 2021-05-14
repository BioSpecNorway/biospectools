from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import numexpr as ne

from biospectools.preprocessing import EMSC
from biospectools.preprocessing.criterions import \
    BaseStopCriterion, TolStopCriterion


class MeEMSC:
    def __init__(
            self,
            reference: np.ndarray,
            wavenumbers: np.ndarray,
            weights: np.ndarray = None,
            n_components: Optional[int] = None,
            n0: np.ndarray = None,
            a: np.ndarray = None,
            h: float = 0.25,
            max_iter: int = 30,
            tol: float = 1e-4,
            patience: int = 1,
            stop_criterion: Optional[BaseStopCriterion] = None,
            positive_ref: bool = True,
            verbose: bool = False):
        self.reference = reference
        self.wavenumbers = wavenumbers
        self.weights = weights if weights is not None else 1

        self.mie_generator = MatlabMieCurvesGenerator(n0, a, h)
        self.mie_decomposer = MatlabMieCurvesDecomposer(n_components)

        self.stop_criterion = stop_criterion
        if self.stop_criterion is None:
            self.stop_criterion = TolStopCriterion(max_iter, tol, patience)

        self.positive_ref = positive_ref
        self.verbose = verbose

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        # wavenumber have to be input as sorted
        # compute average spectrum from the reference
        ref_x = self.reference
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
        for spectrum in spectra:
            pure_guess = ref_x
            self.stop_criterion.reset()
            while not self.stop_criterion:
                try:
                    pure_guess, coefs, res = self._iteration_step(
                        spectrum, pure_guess, basic_emsc)
                    rmse = np.sqrt(np.sum(res ** 2) / len(res))
                    self.stop_criterion.add(rmse, [pure_guess, coefs, res])
                except np.linalg.LinAlgError:
                    self.stop_criterion.add(np.nan, [np.nan, np.nan, np.nan])
                    self.stop_criterion.best_idx = -1
                    break
            new_spectra.append(self.stop_criterion.best_value[0])
            self.coefs_.append(self.stop_criterion.best_value[1])
            self.residuals_.append(self.stop_criterion.best_value[2])
            self.rmse_.append(self.stop_criterion.best_score)
            self.n_iterations_.append(self.stop_criterion.best_iter)

        self.coefs_ = np.stack(self.coefs_)
        self.residuals_ = np.stack(self.residuals_)
        self.rmse_ = np.stack(self.rmse_)
        self.n_iterations_ = np.stack(self.n_iterations_)

        return np.stack(new_spectra)

    def _iteration_step(self, spectrum, reference, basic_emsc: EMSC) -> tuple:
        # scale with basic EMSC:
        reference = basic_emsc.transform(reference[None])[0]
        if np.all(np.isnan(reference)):
            raise np.linalg.LinAlgError()

        reference = reference * self.weights
        if self.positive_ref:
            reference[reference < 0] = 0

        qexts = self.mie_generator.generate(reference, self.wavenumbers)
        qexts = self._orthogonalize(qexts, reference)
        components = self.mie_decomposer.get_orthogonal_components(qexts)

        emsc = EMSC(
            reference=reference, poly_order=0, constituents=components)
        new_spectrum = emsc.transform(spectrum[None])[0]
        # adapt EMSC results to code
        res = emsc.residuals_[0]
        coefs = emsc.coefs_[0, [-1, *range(1, len(components) + 1), 0]]

        return new_spectrum, coefs, res

    def _orthogonalize(self, qext: np.ndarray, reference: np.ndarray):
        rnorm = reference / np.linalg.norm(reference)
        s = np.dot(qext, rnorm)[:, None]
        qext_orthogonalized = qext - s * rnorm
        return qext_orthogonalized


class MatlabMieCurvesGenerator:
    def __init__(self, n0s=None, rs=None, h=0.25):
        self.rs = rs if rs is not None else np.linspace(2, 7.1, 10)
        self.n0s = n0s if n0s is not None else np.linspace(1.1, 1.4, 10)
        self.h = h

        self.rs = self.rs * 1e-6
        self.alpha0s = 4 * np.pi * self.rs * (self.n0s - 1)

        optical_depths = 0.5 * np.pi * self.rs
        fs = self.h * np.log(10) / (4 * np.pi * optical_depths)
        self.gammas = fs / (self.n0s - 1)

        # prepare for broadcasting (alpha, gamma, wns)
        self.alpha0s = self.alpha0s[:, None, None]
        self.gammas = self.gammas[None, :, None]

    def generate(self, pure_absorbance, wavenumbers):
        wavenumbers = wavenumbers * 100
        nprs, nkks = self._get_refractive_index(pure_absorbance, wavenumbers)
        qexts = self._calculate_qext_curves(nprs, nkks, wavenumbers)
        return qexts

    def _calculate_qext_curves(self, nprs, nkks, wavenumbers):
        rho = self.alpha0s * (1 + self.gammas*nkks) * wavenumbers
        tanbeta = nprs / (1 / self.gammas + nkks)
        beta = np.arctan(tanbeta)
        qexts = ne.evaluate(
            '2 - 4 * exp(-rho * tanbeta) * cos(beta) / rho * sin(rho - beta)'
            '- 4 * exp(-rho * tanbeta) * (cos(beta) / rho) ** 2 * cos(rho - 2 * beta)'
            '+ 4 * (cos(beta) / rho) ** 2 * cos(2 * beta)')
        return qexts.reshape(-1, len(wavenumbers))

    def _get_refractive_index(self, pure_absorbance, wavenumbers):
        pad_size = 200
        # Extend absorbance spectrum
        wns_ext = self._extrapolate_wns(wavenumbers, pad_size)
        pure_ext = np.pad(pure_absorbance, pad_size, mode='edge')

        # Calculate refractive index
        nprs_ext = pure_ext / wns_ext
        nkks_ext = hilbert(nprs_ext).imag
        if wns_ext[0] < wns_ext[1]:
            nkks_ext *= -1

        nprs = nprs_ext[pad_size:-pad_size]
        nkks = nkks_ext[pad_size:-pad_size]
        return nprs, nkks

    def _extrapolate_wns(self, wns, pad_size):
        f = interp1d(np.arange(len(wns)), wns, fill_value='extrapolate')
        idxs_ext = np.arange(-pad_size, len(wns) + pad_size)
        wns_ext = f(idxs_ext)
        return wns_ext


class MatlabMieCurvesDecomposer:
    def __init__(self, n_components: Optional[int]):
        self.max_components = 30
        self.explained_thresh = 99.96
        self.svd = TruncatedSVD(n_components, n_iter=7)

    def get_orthogonal_components(self, qexts: np.ndarray):
        if self.svd.n_components is None:
            n_comp = self._estimate_n_components(qexts)
            self.svd.n_components = n_comp
            # do not refit svd, since it was fitted during _estimation
            return self.svd.components_[:n_comp]

        self.svd.fit(qexts)
        return self.svd.components_

    def _estimate_n_components(self, qexts: np.ndarray):
        self.svd.n_components = min(self.max_components, qexts.shape[-1] - 1)
        self.svd.fit(qexts)

        # svd.explained_variance_ is not used since
        # it is not consistent with matlab code
        lda = self.svd.singular_values_ ** 2
        explained_var = np.cumsum(lda / np.sum(lda)) * 100
        n_comp = np.argmax(explained_var > self.explained_thresh) + 1
        return n_comp
