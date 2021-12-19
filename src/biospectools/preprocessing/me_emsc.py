from typing import Optional, List, Union as U, Tuple as T
import copy

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import numexpr as ne

from biospectools.preprocessing import EMSC
from biospectools.preprocessing.emsc import EMSCInternals
from biospectools.preprocessing.criterions import \
    BaseStopCriterion, TolStopCriterion, EmptyCriterionError


class MeEMSCInternals:
    coefs: np.ndarray
    residuals: np.ndarray
    emscs: List[Optional[EMSC]]
    criterions: List[BaseStopCriterion]
    rmses: np.ndarray
    n_iterations: np.ndarray
    n_mie_components: int

    def __init__(
            self,
            criterions: List[BaseStopCriterion],
            n_mie_components: int):
        self.criterions = criterions
        self.n_mie_components = n_mie_components
        if self.n_mie_components <= 0:
            raise ValueError('n_components must be greater than 0')

        self._extract_from_criterions()

    def _extract_from_criterions(self):
        self.emscs = []
        np_arrs = [[] for _ in range(4)]
        rmses, iters, coefs, resds = np_arrs
        for c in self.criterions:
            try:
                self.emscs.append(c.best_value['emsc'])
                emsc_inns: EMSCInternals = c.best_value['internals']
                coefs.append(emsc_inns.coefs[0])
                resds.append(emsc_inns.residuals[0])
                rmses.append(c.best_score)
                iters.append(c.best_iter)
            except EmptyCriterionError:
                self.emscs.append(None)
                coefs.append(np.nan)
                resds.append(np.nan)
                rmses.append(np.nan)
                iters.append(0)

        self.rmses, self.n_iterations, self.coefs, self.residuals = \
            [np.array(np.broadcast_arrays(*arr)) for arr in np_arrs]

    @property
    def scaling_coefs(self) -> np.ndarray:
        return self.coefs[:, 0]

    @property
    def mie_components_coefs(self) -> np.ndarray:
        assert self.n_mie_components > 0, \
            'Number of mie components must be greater than zero'
        return self.coefs[:, 1:1 + self.n_mie_components]

    @property
    def polynomial_coefs(self) -> np.ndarray:
        return self.coefs[:, -1:]


class MeEMSC:
    def __init__(
            self,
            reference: np.ndarray,
            wavenumbers: np.ndarray,
            n_components: Optional[int] = None,
            n0s: np.ndarray = None,
            radiuses: np.ndarray = None,
            h: float = 0.25,
            weights: np.ndarray = None,
            max_iter: int = 30,
            tol: float = 1e-4,
            patience: int = 1,
            positive_ref: bool = True,
            verbose: bool = False):
        self.reference = reference
        self.wavenumbers = wavenumbers
        self.weights = weights

        self.mie_generator = MatlabMieCurvesGenerator(n0s, radiuses, h)
        self.mie_decomposer = MatlabMieCurvesDecomposer(n_components)
        self.stop_criterion = TolStopCriterion(max_iter, tol, patience)

        self.positive_ref = positive_ref
        self.verbose = verbose

    def transform(self, spectra: np.ndarray, internals=False) \
            -> U[np.ndarray, T[np.ndarray, MeEMSCInternals]]:
        ref_x = self.reference
        if self.positive_ref:
            ref_x[ref_x < 0] = 0
        basic_emsc = EMSC(ref_x, self.wavenumbers, rebuild_model=False)

        correcteds = []
        criterions = []
        for spectrum in spectra:
            try:
                result = self._correct_spectrum(basic_emsc, ref_x, spectrum)
            except np.linalg.LinAlgError:
                result = np.full_like(self.wavenumbers, np.nan)
            correcteds.append(result)
            if internals:
                criterions.append(copy.copy(self.stop_criterion))

        if internals:
            inns = MeEMSCInternals(criterions, self.mie_decomposer.n_components)
            return np.array(correcteds), inns
        return np.array(correcteds)

    def _correct_spectrum(self, basic_emsc, pure_guess, spectrum):
        self.stop_criterion.reset()
        while not self.stop_criterion:
            emsc = self._build_emsc(pure_guess, basic_emsc)
            pure_guess, inn = emsc.transform(
                spectrum[None], internals=True, check_correlation=False)
            pure_guess = pure_guess[0]
            rmse = np.sqrt(np.mean(inn.residuals ** 2))
            iter_result = \
                {'corrected': pure_guess, 'internals': inn, 'emsc': emsc}
            self.stop_criterion.add(rmse, iter_result)
        return self.stop_criterion.best_value['corrected']

    def _build_emsc(self, reference, basic_emsc: EMSC) -> EMSC:
        # scale with basic EMSC:
        reference = basic_emsc.transform(
            reference[None], check_correlation=False)[0]
        if np.all(np.isnan(reference)):
            raise np.linalg.LinAlgError()

        if self.weights is not None:
            reference *= self.weights
        if self.positive_ref:
            reference[reference < 0] = 0

        qexts = self.mie_generator.generate(reference, self.wavenumbers)
        qexts = self._orthogonalize(qexts, reference)
        components = self.mie_decomposer.find_orthogonal_components(qexts)

        emsc = EMSC(reference, poly_order=0, constituents=components)
        return emsc

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
        rho = self.alpha0s * (1 + self.gammas*nkks) * wavenumbers  # noqa: F841
        tanbeta = nprs / (1 / self.gammas + nkks)
        beta = np.arctan(tanbeta)  # noqa: F841
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
        self.n_components = n_components
        self.svd = TruncatedSVD(self.n_components, n_iter=7)

        self.max_components = 30
        self.explained_thresh = 99.96

    def find_orthogonal_components(self, qexts: np.ndarray):
        if self.n_components is None:
            self.n_components = self._estimate_n_components(qexts)
            self.svd.n_components = self.n_components
            # do not refit svd, since it was fitted during _estimation
            return self.svd.components_[:self.n_components]

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
