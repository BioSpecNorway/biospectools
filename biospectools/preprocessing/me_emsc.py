from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import numexpr as ne

from biospectools.preprocessing import EMSC
from biospectools.preprocessing.criterions import \
    BaseStopCriterion, TolStopCriterion


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
        nprs_ext = pure_ext / wns_ext

        # Calculate Hilbert transform
        nkks_ext = -hilbert(nprs_ext).imag

        nkks = nkks_ext[pad_size:-pad_size]
        nprs = nprs_ext[pad_size:-pad_size]
        return nprs, nkks

    def _extrapolate_wns(self, wns, pad_size):
        f = interp1d(np.arange(len(wns)), wns, fill_value='extrapolate')
        idxs_ext = np.arange(-pad_size, len(wns) + pad_size)
        wns_ext = f(idxs_ext)
        return wns_ext


def orthogonalize_qext(qext: np.ndarray, reference: np.ndarray):
    m = np.dot(reference, reference)
    norm = np.sqrt(m)
    rnorm = reference / norm
    s = np.dot(qext, rnorm)
    qext_orthogonalized = qext - s[:, np.newaxis] * rnorm[np.newaxis, :]
    return qext_orthogonalized


class MeEMSC:
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
        patience: int = 1,
        stop_criterion: Optional[BaseStopCriterion] = None,
        verbose: bool = False,
        positive_ref: bool = True
    ):

        if reference is None:
            raise ValueError("reference spectrum must be defined")

        if (wavenumbers[1] - wavenumbers[0]) < 0:
            raise ValueError("wn_reference must be ascending")

        self.reference = reference
        self.wavenumbers = wavenumbers
        self.positive_ref = positive_ref
        self.stop_criterion = stop_criterion
        if self.stop_criterion is None:
            self.stop_criterion = TolStopCriterion(max_iter, tol, patience)
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(self.reference))
        self.n_components = n_components
        self.verbose = verbose

        self.mie_generator = MatlabMieCurvesGenerator(n0, a, h)

        if self.n_components is None:
            self.n_components = self._estimate_n_components()

    def transform(self, X: np.ndarray) -> np.ndarray:
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
        for spectrum in X:
            pure_guess = ref_x
            self.stop_criterion.reset()
            while not self.stop_criterion:
                try:
                    pure_guess, coefs, res = self._iteration_step(
                        spectrum, pure_guess, basic_emsc)
                    rmse = np.sqrt((1 / len(res[0, :])) * np.sum(res ** 2))
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
        svd = self._generate_mie_curves_and_fit_svd(
            reference, self.n_components)

        emsc = EMSC(
            reference=reference, poly_order=0, constituents=svd.components_)
        new_spectrum = emsc.transform(spectrum[None])[0]
        # adapt EMSC results to code
        res = emsc.residuals_
        coefs = emsc.coefs_[0, [-1, *range(1, self.n_components + 1), 0]]

        return new_spectrum, coefs, res

    def _generate_mie_curves_and_fit_svd(self, reference, n_components):
        qext = self.mie_generator.generate(reference, self.wavenumbers)
        qext_orthogonal = orthogonalize_qext(qext, reference)
        self._n_generated = len(qext)

        svd = TruncatedSVD(n_components, n_iter=7)
        svd.fit(qext_orthogonal)
        return svd

    def _estimate_n_components(self):
        svd = self._generate_mie_curves_and_fit_svd(
            self.reference, n_components=min(30, len(self.reference) - 1))
        lda = np.array(
            [
                (sing_val ** 2) / (self._n_generated - 1)
                for sing_val in svd.singular_values_
            ]
        )

        explained_var = 100 * lda / np.sum(lda)
        explained_var = np.cumsum(explained_var)
        explained_var_thresh = 99.96
        num_comp = np.argmax(explained_var > explained_var_thresh) + 1
        return num_comp
