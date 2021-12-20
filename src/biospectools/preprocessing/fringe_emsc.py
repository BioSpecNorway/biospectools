from typing import Tuple as T, List, Union as U

import numpy as np
import scipy
from scipy.signal import windows

from biospectools.preprocessing import EMSC
from biospectools.preprocessing.emsc import EMSCInternals


class FringeEMSCInternals:
    """Contains intermediate results of FringeEMSC algorithm.

    Parameters
    ----------
    coefs : `(N_samples, 1 + N_constituents + (poly_order + 1) ndarray`
        All coefficients for each transformed sample. First column is a
        scaling parameter followed by constituent and polynomial coefs.
        This is a transposed solution of equation
        _model @ coefs.T = spectrum.
    scaling_coefs : `(N_samples,) ndarray`
        Scaling coefficients (reference to the first column of coefs_).
    polynomial_coefs : `(N_samples, poly_order + 1) ndarray`
        Coefficients for each polynomial order.
    constituents_coefs : `(N_samples, N_constituents) ndarray`
        Coefficients for each constituent.
    freqs_coefs: `(N_samples, n_freqs, 2) ndarray`
        Coefficients for sin and cos components corresponding
        to frequency in freqs.
    freqs: `(N_samples, n_freqs) ndarray`
        List of frequencies sorted by their prominance in fringes.
    residuals : `(N_samples, K_channels) ndarray`
         Chemical residuals that were not fitted by EMSC model.

    Raises
    ------
    AttributeError
        When polynomial's or constituents' coeffs are not available.
    """
    def __init__(self, emsc_internals: List[EMSCInternals], freqs):
        self.freqs = np.array(freqs)
        self._gather_emsc_attributes(emsc_internals)
        self._sort_freqs_by_contribution()
        pass

    def _gather_emsc_attributes(self, emscs: List[EMSCInternals]):
        self.coefs = np.array([e.coefs[0] for e in emscs])
        self.scaling_coefs = np.array([e.scaling_coefs[0] for e in emscs])
        self.residuals = np.array([e.residuals[0] for e in emscs])

        try:
            self.polynomial_coefs = np.array(
                [e.polynomial_coefs[0] for e in emscs])
        except AttributeError:
            pass

        self.freqs_coefs = self._extract_frequencies(emscs)

        n_freq_coefs = self.freqs.shape[1] * 2
        if emscs[0].constituents_coefs.shape[1] > n_freq_coefs:
            self.constituents_coefs = np.array(
                [e.constituents_coefs[0, n_freq_coefs:] for e in emscs])

    def _extract_frequencies(self, emscs: List[EMSCInternals]):
        n = self.freqs.shape[1]
        # each freq has sine and cosine component
        freq_coefs = np.array([e.constituents_coefs[0, :n * 2] for e in emscs])
        return freq_coefs.reshape((-1, n, 2))

    def _sort_freqs_by_contribution(self):
        freq_scores = np.abs(self.freqs_coefs).sum(axis=-1)
        idxs = np.argsort(-freq_scores, axis=-1)  # descendent
        idxs = np.unravel_index(idxs, self.freqs_coefs.shape[:2])

        self.freqs = self.freqs[idxs]
        self.freqs_coefs = self.freqs_coefs[idxs]

        # fix order in coefs_
        # take into account that freq's sine and cosine components
        # are flattened in coefs_ and we want to move sin and cos
        # together
        n = self.freqs.shape[1]
        freq_coefs = self.coefs[:, 1: n * 2 + 1]
        reordered = freq_coefs.reshape(-1, n, 2)[idxs].reshape(-1, n*2)
        self.coefs[:, 1: n * 2 + 1] = reordered


class FringeEMSC:
    """FringeEMSC [1]_ removes fringe effects from spectra that often
    occur in thin film samples. To remove fringes, it estimates
    a frequency of fringes in the given region with FFT and adds
    sin-cos components to the EMSC model. The EMSC model is built
    individually for each spectrum.

    Parameters
    ----------
    reference : `(K_channels,) ndarray`
        Reference spectrum.
    wavenumbers: `(K_channels,) ndarray`
        Wavenumbers must be ordered in ascending or descending order.
    fringe_wn_location: `Tuple[float, float]`
        Left and right wavenumbers of region with undisturbed fringes,
        i.e. the silent region. Longer region will give better
        estimation for fringes.
    n_freq: `int`, optional (default 2)
        Number of frequencies that will be used to fit fringes
    poly_order : `int`, optional (default 2)
        Order of polynomial to be used in regression model. If None
        then polynomial will be not used.
    weights : `(K_channels,) ndarray`, optional
        Weights of spectra used in the EMSC model.
    constituents : `(N_constituents, K_channels) np.ndarray`, optional
        Chemical constituents for the ESMC model.
    scale : `bool`, default True
        If True then spectra will be scaled to reference spectrum.
    pad_length_multiplier: `int`, (default 5)
        Padding for FFT is calculated relatively to the length of
        fringe region. So padding is len(region) times pad_length_multiplier.
    double_freq: `bool`, default True
        Whether to include neighbouring frequence to the model. Usually,
        the fringe frequency cannot be retrieved exactly, due to the
        discrete Fourier domain. Including neighbouring frequencies
        help mitigate that problem.
    window_function: `Callable`, (default scipy.signal.windows.bartlett)
        The window function for FFT transform.

    References
    ----------
    .. [1] Solheim J. H. et al. *An automated approach for
           fringe frequency estimation and removal in infrared
           spectroscopy and hyperspectral imaging of biological
           samples.* Journal of Biophotonics: e202100148
    """
    def __init__(
            self,
            reference,
            wavenumbers,
            fringe_wn_location: T[float, float],
            n_freq: int = 2,
            poly_order: int = 2,
            weights=None,
            constituents=None,
            scale: bool = True,
            pad_length_multiplier: int = 5,
            double_freq: bool = True,
            window_function=windows.bartlett
    ):
        self.reference = np.asarray(reference)
        self.wavenumbers = np.asarray(wavenumbers)
        self.fringe_wn_location = fringe_wn_location
        self.n_freq = n_freq
        self.poly_order = poly_order
        self.weights = weights
        self.constituents = constituents
        self.scale = scale
        self.pad_length_multiplier = pad_length_multiplier
        self.double_freq = double_freq
        self.window_function = window_function

    def transform(
            self,
            spectra,
            internals=False) \
            -> U[np.ndarray, T[np.ndarray, FringeEMSCInternals]]:
        spectra = np.asarray(spectra)

        corrected = []
        emscs_internals = []
        all_freqs = []
        for spec in spectra:
            freqs = self._find_fringe_frequencies(spec)
            emsc = self._build_emsc(freqs)
            corr, inns = emsc.transform(
                spec[None], internals=True, check_correlation=False)

            corrected.append(corr[0])
            emscs_internals.append(inns)
            all_freqs.append(freqs)
        corrected = np.array(corrected)

        if internals:
            inn = FringeEMSCInternals(emscs_internals, all_freqs)
            return corrected, inn
        return corrected

    def _find_fringe_frequencies(self, raw_spectrum):
        region = self._select_fringe_region(raw_spectrum)
        region = region - region.mean()
        region *= self.window_function(len(region))

        f_transform, freqs = self._apply_fft(region)
        freq_idxs, _ = scipy.signal.find_peaks(f_transform)

        # use only N highest frequencies
        max_idxs = f_transform[freq_idxs].argsort()[-self.n_freq:]
        freq_idxs = freq_idxs[max_idxs]

        if self.double_freq:
            ft = f_transform
            # FIXME: out of bounds?
            neighbors = [i + 1 if ft[i + 1] > ft[i - 1] else i - 1
                         for i in freq_idxs]
            freq_idxs = np.concatenate((freq_idxs, neighbors))

        return freqs[freq_idxs]

    def _apply_fft(self, region):
        k = self._padded_region_length(region)
        dw = np.abs(np.diff(self.wavenumbers).mean())
        freqs = 2 * np.pi * scipy.fft.fftfreq(k, dw)[0:k // 2]
        f_transform = scipy.fft.fft(region, k)[0:k // 2]
        f_transform = np.abs(f_transform)
        return f_transform, freqs

    def _build_emsc(self, freqs):
        fringe_comps = np.array([sin_then_cos(freq * self.wavenumbers)
                                 for freq in freqs
                                 for sin_then_cos in [np.sin, np.cos]])
        if self.constituents is not None:
            constituents = np.concatenate((fringe_comps, self.constituents))
        else:
            constituents = fringe_comps
        emsc = EMSC(
            self.reference, self.wavenumbers, self.poly_order,
            constituents, self.weights, self.scale)
        return emsc

    def _padded_region_length(self, region):
        k = region.shape[-1]
        pad = int(k * self.pad_length_multiplier)
        length = k + pad
        if length % 2 == 1:
            length += 1
        return length

    def _select_fringe_region(self, spectra):
        """
        Assumes that spectra lies along last axis
        """
        wns = self.wavenumbers
        idx_lower = np.argmin(abs(wns - self.fringe_wn_location[0]))
        idx_upper = np.argmin(abs(wns - self.fringe_wn_location[1]))
        if idx_lower > idx_upper:
            idx_lower, idx_upper = idx_upper, idx_lower
        region = spectra[..., idx_lower: idx_upper]
        return region
