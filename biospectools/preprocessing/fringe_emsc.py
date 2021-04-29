from typing import Tuple as T

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
            emsc_weights=None,
            scale: bool = True,
            pad_length_multiplier: float = 5,
            double_freq: bool = False,
            window_function=windows.bartlett
    ):
        self.reference = np.asarray(reference)
        self.wavenumbers = np.asarray(wavenumbers)
        self.fringe_wn_location = fringe_wn_location
        self.n_freq = n_freq
        self.poly_order = poly_order
        self.emsc_weights = emsc_weights
        self.scale = scale
        self.pad_length_multiplier = pad_length_multiplier
        self.double_freq = double_freq
        self.window_function = window_function

    def transform(self, spectra):
        spectra = np.asarray(spectra)
        wns = self.wavenumbers
        corrected = []
        self.emsc_models_ = []
        self.freqs_ = []
        for spec in spectra:
            freqs = self._find_fringe_frequencies(spec, wns)
            constituents = np.stack([sin_then_cos(freq * wns)
                                     for freq in freqs
                                     for sin_then_cos in [np.sin, np.cos]])
            emsc = EMSC(
                self.reference, wns, self.poly_order,
                self.emsc_weights, constituents, self.scale)
            corr = emsc.transform(spec[None])

            corrected.append(corr[0])
            self.emsc_models_.append(emsc)
            self.freqs_.append(freqs)

        self.freqs_ = np.stack(self.freqs_)
        return np.stack(corrected)

    def clear_state(self):
        del self.emsc_models_
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
