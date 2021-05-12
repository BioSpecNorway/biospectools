from typing import Tuple as T, List

import numpy as np
import scipy
from scipy.signal import windows

from biospectools.preprocessing import EMSC


class FringeEMSC:
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
            pad_length_multiplier: float = 5,
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

    def transform(self, spectra):
        spectra = np.asarray(spectra)
        corrected = []
        emsc_models = []
        self.freqs_ = []
        for spec in spectra:
            freqs = self._find_fringe_frequencies(spec)
            emsc = self._build_emsc(freqs)
            corr = emsc.transform(spec[None])

            corrected.append(corr[0])
            emsc_models.append(emsc)
            self.freqs_.append(freqs)

        self.freqs_ = np.array(self.freqs_)
        self._gather_emsc_attributes(emsc_models)
        self._sort_freqs_by_contribution()
        return np.array(corrected)

    def clear_state(self):
        del self.coefs_
        del self.scaling_coefs_
        del self.constituents_coefs_
        del self.polynomial_coefs_
        del self.residuals_
        del self.freqs_coefs_
        del self.freqs_

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
            self.weights, constituents, self.scale)
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

    def _gather_emsc_attributes(self, emscs: List[EMSC]):
        self.coefs_ = np.array([e.coefs_[0] for e in emscs])
        self.scaling_coefs_ = np.array([e.scaling_coefs_[0] for e in emscs])
        self.residuals_ = np.array([e.residuals_[0] for e in emscs])

        if self.poly_order is not None:
            self.polynomial_coefs_ = np.array([e.polynomial_coefs_[0]
                                               for e in emscs])
        else:
            self.polynomial_coefs_ = None

        if self.constituents is not None:
            n = len(self.constituents)
            self.constituents_coefs_ = np.array([e.constituents_coefs_[0, -n:]
                                                 for e in emscs])
        else:
            self.constituents_coefs_ = None

        self.freqs_coefs_ = self._extract_frequencies(emscs)

    def _extract_frequencies(self, emscs: List[EMSC]):
        n = self.n_freq
        if self.double_freq:
            n *= 2

        # each freq has sine and cosine component
        freq_coefs = np.array([e.constituents_coefs_[0, :n * 2] for e in emscs])
        return freq_coefs.reshape((-1, n, 2))

    def _sort_freqs_by_contribution(self):
        freq_scores = np.abs(self.freqs_coefs_).sum(axis=-1)
        idxs = np.argsort(-freq_scores, axis=-1)  # descendent
        idxs = np.unravel_index(idxs, self.freqs_coefs_.shape[:2])

        self.freqs_ = self.freqs_[idxs]
        self.freqs_coefs_ = self.freqs_coefs_[idxs]

        # fix order in coefs_
        n = self.n_freq
        if self.double_freq:
            n *= 2
        # take into account that freq's sine and cosine components
        # are flattened in coefs_ and we want to move sin and cos
        # together
        freq_coefs = self.coefs_[:, 1: n * 2 + 1]
        reordered = freq_coefs.reshape(-1, n, 2)[idxs].reshape(-1, n*2)
        self.coefs_[:, 1: n * 2 + 1] = reordered
