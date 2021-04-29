import numpy as np
from scipy.fftpack import fft

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time
from scipy.signal import hilbert
from scipy.signal import blackman, blackmanharris, bartlett
from scipy.signal import find_peaks
from scipy import signal
from biospectools.preprocessing.emsc import *

class FringeEMSC:
    def __init__(self, refSpec, wnref, wnLower, wnUpper, nFreq, scaling=True, weights=False, polyorder=2,
                 Npad=2.5, double_freq=False, returnmodel=False):
        self.refSpec = refSpec
        self.wnref = wnref
        self.wnUpper = wnUpper
        self.wnLower = wnLower
        self.nFreq = nFreq
        self.scaling = scaling
        self.weights = weights
        self.polyorder = polyorder
        self.Npad = Npad
        self.flipWN = False
        self.flipWeigths = False
        self.double_freq = double_freq
        self.returnmodel = returnmodel

        if self.wnref[1]<self.wnref[0]:
            self.wnref = np.flip(self.wnref)
            self.refSpec = np.flip(self.refSpec)

    def flipWeights(self, weights):  #FIXME should be more elegant
        if not self.flipWeigths:
            if isinstance(weights, bool):
                return weights
            else:
                weights = np.flip(weights)
                self.flipWeigths = True
                return weights
        else:
            return weights

    def frequency_from_spectrum(self, rawspectrum, wn):
        indLower = np.argmin(abs(wn-self.wnLower))
        indUpper = np.argmin(abs(wn-self.wnUpper))

        region_frequency = rawspectrum[indLower:indUpper]

        # Zero pad signal
        nPad = region_frequency.shape[0]
        region_frequency = region_frequency - np.mean(region_frequency)

        # apply window
        N = len(region_frequency)
        w = signal.bartlett(N)
        region_frequency = region_frequency * w

        region_frequency = np.hstack((np.zeros([int(np.ceil(nPad*self.Npad))]), region_frequency, np.zeros([int(np.ceil(nPad*self.Npad))])))
        if len(region_frequency) % 2:
            region_frequency = np.hstack((region_frequency, region_frequency[-1]))

        # Fourier transform signal
        fTransform = fft(region_frequency)

        # Calculate the frequency axis in the Fourier domain
        N = region_frequency.shape[0]

        dw = wn[1]-wn[0]
        x = 2*np.pi*np.fft.fftfreq(N, dw)

        fTransform = np.abs(fTransform[0:N//2])
        x = x[0:N//2]

        peaks_ind, _ = find_peaks(fTransform)


        n_max = fTransform[peaks_ind].argsort()[-self.nFreq:]

        if self.double_freq:
            peakpos_double = []

            for peakpos in n_max:
                upper = fTransform[peaks_ind[peakpos] + 1]
                lower = fTransform[peaks_ind[peakpos] - 1]
                if upper > lower:
                    peakpos_double.append(peaks_ind[peakpos] + 1)
                else:
                    peakpos_double.append(peaks_ind[peakpos] - 1)
                peakpos_double.append(peaks_ind[peakpos])

            freqMax = x[np.flip(peakpos_double)]
        else:
            freqMax = x[np.flip(peaks_ind[n_max])]

        return freqMax

    def setup_emsc(self, freqMax, wn):
        half_rng = np.abs(wn[0] - wn[-1]) / 2
        normalized_wns = (wn - np.mean(wn)) / half_rng

        polynomial_columns = [np.ones(len(wn))]
        for j in range(1, self.polyorder + 1):
            if j == 1:
                polynomial_columns.append(-1 * normalized_wns ** j)
            elif j == 3:
                polynomial_columns.append(0.5 * normalized_wns ** j)
            else:
                polynomial_columns.append(normalized_wns ** j)
        M = np.stack(polynomial_columns, axis=1)

        for i in range(0, self.nFreq + self.double_freq*self.nFreq):
            sinspec = np.sin(freqMax[i]*wn)
            sinspec = sinspec.reshape(-1, 1)
            cosspec = np.cos(freqMax[i]*wn)
            cosspec = cosspec.reshape(-1, 1)
            M = np.hstack((M, sinspec, cosspec))
        M = np.hstack((M, self.refSpec.reshape(-1, 1)))

        return M

    def solve_emsc(self, rawspectrum, M, nbad):
        if isinstance(self.weights, bool):
            weights = np.ones((rawspectrum.shape))
        else:
            weights = self.weights  # FIXME interpolate the weights to the data

        weights=weights.reshape(-1, 1).T

        MW = M*weights.T
        rawspectrumW = rawspectrum*weights[0]

        params = np.linalg.lstsq(MW, rawspectrumW, rcond=-1)[0]
        corrected = rawspectrum

        for x in range(0, nbad):
            corrected = (corrected - (params[x] * M[:, x]))
        if self.scaling:
            corrected = corrected/params[-1]

        residuals = rawspectrum - np.dot(params, M.T)
        return corrected, params, residuals

    def correct_spectra(self, rawspectra, wn):
        nbad = 1 + self.polyorder + self.nFreq*2 + self.double_freq*self.nFreq*2
        newspectra = np.full(rawspectra.shape, np.nan)
        residuals = np.full(rawspectra.shape, np.nan)
        parameters = np.full([rawspectra.shape[0], self.nFreq*2 + self.double_freq*self.nFreq*2 + 2 + self.polyorder], np.nan)
        allmodels = dict()
        freq_list = np.empty((0, self.nFreq + self.double_freq * self.nFreq))
        for i in range(0, rawspectra.shape[0]):
            freq = self.frequency_from_spectrum(rawspectra[i, :], wn)
            emsc_mod = self.setup_emsc(freq, wn)
            corr, par, res = self.solve_emsc(rawspectra[i, :], emsc_mod, nbad)
            newspectra[i, :] = corr
            residuals[i, :] = res
            parameters[i, :] = par.T
            allmodels[i] = emsc_mod
            freq_list = np.append(freq_list, freq[np.newaxis], axis=0)
        return newspectra, parameters, residuals, allmodels, freq_list

    def transform(self, spectra, wn):
        self.flipWN = False
        if wn[1]<wn[0]:
            wn = np.flip(wn)
            spectra = np.fliplr(spectra)
            self.flipWN = True
            self.weights = self.flipWeights(self.weights)

        newspectra, parameters, residuals, modelspectra, freq_list = self.correct_spectra(spectra, wn)

        if self.flipWN:
            newspectra = np.fliplr(newspectra)
            residuals = np.fliplr(residuals)
            for i in range(len(modelspectra)):
                modelspectra[i] = np.flipud(modelspectra[i])

        if self.returnmodel:
            return newspectra, parameters, residuals, modelspectra, freq_list
        else:
            return newspectra, parameters, residuals, freq_list


if __name__ == '__main__':
    from biospectools.physics.peak_shapes import *
    from biospectools.physics.misc import *
    from biospectools.physics.fresnel_equations import *
    import matplotlib.pyplot as plt

    wn = np.linspace(1000, 2500, 1500)*100
    wl = 1/wn

    lorentz_peak = lorentzian(wn, 1600*100, 0.05, 500)
    nkk = get_nkk(lorentz_peak, wl)
    n0 = 1.3

    n = n0 + nkk + 1j*lorentz_peak

    l = 15e-6

    t = transmission_amp(n, wl, l)
    T = np.abs(t)**2


    fringe_spectrum = -np.log10(T)

    wn = wn/100
    fringeEMSCmodelEMSC = FringeEMSC(refSpec=lorentz_peak*10, wnref=wn, wnLower=1800, wnUpper=2320, nFreq=2,
                                     scaling=True, polyorder=1, Npad=2.5, double_freq=True)

    lol = np.vstack((fringe_spectrum, fringe_spectrum))
    corr, par, res, freqList = fringeEMSCmodelEMSC.transform(lol, wn)

    plt.figure()
    plt.plot(wn, fringe_spectrum)
    plt.plot(wn, corr[0,:])

    fringeEMSCmodelEMSC2 = FringeEMSC2(refSpec=lorentz_peak*10, wnref=wn, wnLower=1800, wnUpper=2320, nFreq=1,
                                       scaling=True, polyorder=1, Npad=2.5, double_freq=True)

    corr, par, res, freqList2 = fringeEMSCmodelEMSC2.transform(fringe_spectrum[np.newaxis, :], wn)

    plt.figure()
    plt.plot(wn, fringe_spectrum)
    plt.plot(wn, corr[0,:])
    plt.show()

    print('lol')