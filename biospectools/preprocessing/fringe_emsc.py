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
        wns = self.wavenumbers
        corrected = []
        emsc_models = []
        self.freqs_ = []
        for spec in spectra:
            freqs = self._find_fringe_frequencies(spec, wns)
            fringe_comps = np.array([sin_then_cos(freq * wns)
                                     for freq in freqs
                                     for sin_then_cos in [np.sin, np.cos]])
            if self.constituents is not None:
                constituents = np.concatenate((fringe_comps, self.constituents))
            else:
                constituents = fringe_comps
            emsc = EMSC(
                self.reference, wns, self.poly_order,
                self.weights, constituents, self.scale)
            corr = emsc.transform(spec[None])

            corrected.append(corr[0])
            emsc_models.append(emsc)
            self.freqs_.append(freqs)

        self.freqs_ = np.array(self.freqs_)
        self._gather_emsc_attributes(emsc_models)
        return np.array(corrected)

    def clear_state(self):
        del self.coefs_
        del self.scaling_coefs_
        del self.constituents_coefs_
        del self.polynomial_coefs_
        del self.residuals_
        del self.freq_coefs_
        del self.freqs_

    def _find_fringe_frequencies(self, raw_spectrum, wavenumbers):
        region = self._select_fringe_region(raw_spectrum, wavenumbers)
        region = region - region.mean()
        region *= self.window_function(len(region))

        f_transform, freqs = self._apply_fft(region, wavenumbers)
        freq_idxs, _ = scipy.signal.find_peaks(f_transform)

        # use only N highest frequencies
        max_idxs = f_transform[freq_idxs].argsort()[-self.n_freq:]
        freq_idxs = freq_idxs[max_idxs]

        if self.double_freq:
            ft = f_transform
            neighbors = [i + 1 if ft[i + 1] > ft[i - 1] else i - 1
                         for i in freq_idxs]
            freq_idxs = np.concatenate((freq_idxs, neighbors))

        freq_idxs.sort()
        return freqs[freq_idxs]

    def _apply_fft(self, region, wavenumbers):
        k = self._padded_region_length(region)
        dw = np.abs(np.diff(wavenumbers).mean())
        freqs = 2 * np.pi * scipy.fft.fftfreq(k, dw)[0:k // 2]
        f_transform = scipy.fft.fft(region, k)[0:k // 2]
        f_transform = np.abs(f_transform)
        return f_transform, freqs

    def _padded_region_length(self, region):
        k = region.shape[-1]
        pad = int(k * self.pad_length_multiplier)
        length = k + pad
        if length % 2 == 1:
            length += 1
        return length

    def _select_fringe_region(self, spectra, wavenumbers):
        """
        Assumes that spectra lies along last axis
        """
        idx_lower = np.argmin(abs(wavenumbers - self.fringe_wn_location[0]))
        idx_upper = np.argmin(abs(wavenumbers - self.fringe_wn_location[1]))
        if idx_lower > idx_upper:
            idx_lower, idx_upper = idx_upper, idx_lower
        region = spectra[..., idx_lower: idx_upper]
        return region

    def _gather_emsc_attributes(self, emsc_models: List[EMSC]):
        self.coefs_ = np.array(
            [m.coefs_[0] for m in emsc_models])
        self.scaling_coefs_ = np.array(
            [m.scaling_coefs_[0] for m in emsc_models])
        self.residuals_ = np.array(
            [m.residuals_[0] for m in emsc_models])

        if self.poly_order is not None:
            all_poly = [m.polynomial_coefs_[0] for m in emsc_models]
            self.polynomial_coefs_ = np.array(all_poly)
        else:
            self.polynomial_coefs_ = None

        if self.constituents is not None:
            n = len(self.constituents)
            all_csts = [m.constituents_coefs_[0, -n:] for m in emsc_models]
            self.constituents_coefs_ = np.array(all_csts)
        else:
            self.constituents_coefs_ = None

        self.freq_coefs_ = self._extract_frequencies(emsc_models)

    def _extract_frequencies(self, emsc_models: List[EMSC]):
        n = self.n_freq
        if self.double_freq:
            n *= 2

        # each freq has sine and cosine component
        freq_coefs = [m.constituents_coefs_[0, :n * 2] for m in emsc_models]
        freq_coefs = np.array(freq_coefs)
        return freq_coefs.reshape((-1, n, 2))
