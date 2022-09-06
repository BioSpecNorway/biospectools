from typing import Tuple as T, List, Union as U

import numpy as np
import scipy
from scipy.signal import windows

from biospectools.preprocessing import EMSC
from biospectools.preprocessing.emsc import EMSCDetails
from biospectools.utils.deprecated import deprecated_alias


class FringeEMSCDetails:
    """Contains intermediate results of FringeEMSC algorithm. All
    coefficients preserve the spatial shape of input spectra. For example,
    if the input spectra had a shape (128, 128, 3000), then all coefficients
    will have shape starting with (128, 128). The spatial shape will
    be denoted as ...

    Parameters
    ----------
    coefs : `(..., 1 + N_constituents + (poly_order + 1) ndarray`
        All coefficients for each transformed sample. First column is a
        scaling parameter followed by constituent and polynomial coefs.
        This is a transposed solution of equation
        _model @ coefs.T = spectrum.
    scaling_coefs : `(...,) ndarray`
        Scaling coefficients (reference to the first column of coefs_).
    polynomial_coefs : `(..., poly_order + 1) ndarray`
        Coefficients for each polynomial order.
    interferents_coefs : `(..., N_interferents) ndarray`
        Coefficients for each interferent.
    analytes_coefs : `(..., N_analytes) ndarray`
        Coefficients for each analyte.
    freqs_coefs: `(..., n_freqs, 2) ndarray`
        Coefficients for sin and cos components corresponding
        to frequency in freqs.
    freqs: `(..., n_freqs) ndarray`
        List of frequencies sorted by their prominance in fringes.
    residuals : `(..., K_channels) ndarray`
         Chemical residuals that were not fitted by EMSC model.

    Raises
    ------
    AttributeError
        When polynomial's or interferents' coeffs are not available.
    """
    def __init__(self, emsc_internals: List[EMSCDetails], freqs, input_shape):
        self.freqs = np.array(freqs)
        self.input_shape = input_shape
        self._gather_emsc_attributes(emsc_internals)
        self._sort_freqs_by_contribution()

    def _gather_emsc_attributes(self, emscs: List[EMSCDetails]):
        shp = self.input_shape[:-1] + (-1,)
        self.coefs = np.array(
            [e.coefs for e in emscs]).reshape(shp)
        self.scaling_coefs = np.array(
            [e.scaling_coefs for e in emscs]).reshape(shp[:-1])
        self.residuals = np.array(
            [e.residuals for e in emscs]).reshape(shp)

        try:
            self.polynomial_coefs = np.array(
                [e.polynomial_coefs for e in emscs]).reshape(shp)
        except AttributeError:
            pass

        self.freqs_coefs = self._extract_frequencies(emscs)

        n_freq_coefs = self.freqs.shape[-1] * 2
        if emscs[0].interferents_coefs.shape[-1] > n_freq_coefs:
            self.interferents_coefs = np.array(
                [e.interferents_coefs[n_freq_coefs:] for e in emscs])
            self.interferents_coefs = self.interferents_coefs.reshape(shp)
        try:
            self.analytes_coefs = np.array(
                [e.analytes_coefs for e in emscs]).reshape(shp)
        except AttributeError:
            pass

    def _extract_frequencies(self, emscs: List[EMSCDetails]):
        n = self.freqs.shape[-1]
        # each freq has sine and cosine component
        freq_coefs = np.array([e.interferents_coefs[:n * 2] for e in emscs])
        return freq_coefs.reshape((*self.input_shape[:-1], n, 2))

    def _sort_freqs_by_contribution(self):
        freq_scores = np.abs(self.freqs_coefs).sum(axis=-1)
        idxs = np.argsort(-freq_scores, axis=-1)  # descendent
        idxs = np.unravel_index(idxs, self.freqs_coefs.shape[:-1])

        self.freqs = self.freqs[idxs]
        self.freqs_coefs = self.freqs_coefs[idxs]

        # fix order in coefs_
        # take into account that freq's sine and cosine components
        # are flattened in coefs_ and we want to move sin and cos
        # together
        n = self.freqs.shape[-1]
        freq_coefs = self.coefs[..., 1: n * 2 + 1]
        reordered = freq_coefs.reshape(*self.input_shape[:-1], n, 2)[idxs]
        reordered = reordered.reshape(*self.input_shape[:-1], n*2)
        self.coefs[..., 1: n * 2 + 1] = reordered


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
    interferents : `(N_interferents, K_channels) np.ndarray`, optional
        Chemical interferents for the ESMC model.
    analytes : `(N_analytes, K_channels) np.ndarray`, optional
        Chemical analytes for the ESMC model.
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
    @deprecated_alias(constituents='interferents')
    def __init__(
            self,
            reference,
            wavenumbers,
            fringe_wn_location: T[float, float],
            n_freq: int = 2,
            poly_order: int = 2,
            weights=None,
            interferents=None,
            analytes=None,
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
        self.interferents = interferents
        self.analytes = analytes
        self.scale = scale
        self.pad_length_multiplier = pad_length_multiplier
        self.double_freq = double_freq
        self.window_function = window_function

    @deprecated_alias(internals='details')
    def transform(
            self,
            spectra,
            details=False) \
            -> U[np.ndarray, T[np.ndarray, FringeEMSCDetails]]:
        spectra = np.asarray(spectra)
        input_shape = spectra.shape
        spectra = spectra.reshape(-1, input_shape[-1])

        corrected = []
        emscs_internals = []
        all_freqs = []
        for spec in spectra:
            freqs = self._find_fringe_frequencies(spec)
            emsc = self._build_emsc(freqs)
            corr, inns = emsc.transform(
                spec, details=True, check_correlation=False)

            corrected.append(corr)
            emscs_internals.append(inns)
            all_freqs.append(freqs)
        corrected = np.array(corrected).reshape(input_shape)
        all_freqs = np.array(all_freqs).reshape(input_shape[:-1] + (-1,))

        if details:
            inn = FringeEMSCDetails(emscs_internals, all_freqs, input_shape)
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
        if self.interferents is not None:
            interferents = np.concatenate((fringe_comps, self.interferents))
        else:
            interferents = fringe_comps
        emsc = EMSC(
            self.reference, self.wavenumbers, self.poly_order,
            interferents, self.analytes, self.weights, self.scale)
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
