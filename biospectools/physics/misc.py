import logging

import numpy as np
from scipy.signal import hilbert

from biospectools.physics import van_de_hulst as vdh


def get_imagpart(pure_absorbance, wavelength, radius, factor=1):
    """
    Computes imaginary part from pure absorbance

    :param pure_absorbance:
    :param wavelength: micrometers
    :param radius: micrometers
    :param factor: ?
    :return:
    """
    if np.any(wavelength > 500):
        logging.warning('Big wavelength occured: '
                        'Starting from the version 0.3.0 this method takes'
                        'wavelength instead of wavenumbers')
    deff = np.pi / 2 * radius * factor
    imagpart = (pure_absorbance * np.log(10)) / \
               (4 * np.pi * deff / wavelength)
    return imagpart


def get_nkk(imag_part, wavelengths: np.ndarray, pad_size=200):
    """
    Calculates kramers-kronig relation across last dimension
    i.e. assumes that frequency dimension is last one.
    :param imag_part: np.ndarray
    :param wavelengths: array of wavelengths 1D or same shape as imag_part
    :param pad_size: int
    :return:
    """
    pad_last_axis = [(0, 0)] * imag_part.ndim
    pad_last_axis[-1] = (pad_size, pad_size)
    nkk = np.imag(hilbert(np.pad(imag_part, pad_last_axis, mode='edge')))
    nkk = nkk[..., pad_size:-pad_size]

    wls_increase = wavelengths[..., 0] < wavelengths[..., -1]
    if wls_increase:
        # return in both cases C-Contiguous array
        return nkk.copy()
    else:
        return -nkk


def to_wavelength(wavenumbers):
    return 10e+3 / wavenumbers


def calculate_complex_n(ref_X, wavenumbers):
    """
    Calculates the scaled imaginary part and scaled fluctuating
    real part of the refractive index.
    """

    npr = ref_X
    nprs = npr / (wavenumbers * 100)

    # Extend absorbance spectrum
    dw = wavenumbers[1] - wavenumbers[0]
    wavenumbers_extended = np.hstack(
        (
            dw * np.linspace(1, 200, 200) + (wavenumbers[0] - dw * 201),
            wavenumbers,
            dw * np.linspace(1, 200, 200) + (wavenumbers[-1]),
        )
    )
    extension1 = npr[0] * np.ones(200)
    extension2 = npr[-1] * np.ones(200)
    npr_extended = np.hstack((extension1, npr, extension2))

    # Calculate Hilbert transform
    nkks_extended = -hilbert(npr_extended / (wavenumbers_extended * 100)).imag

    # Cut extended spectrum
    nkks = nkks_extended[200:-200]
    return nprs, nkks


def to_wavenumbers(wavelength):
    return 10e+3 / wavelength


def get_qext(wavelength: np.ndarray,
             pure_absorbance: np.ndarray,
             n0: float, r: float, mie_func=vdh.mie_complex):
    imagpart = get_imagpart(pure_absorbance, wavelength, r)
    nkk = get_nkk(imagpart, wavelength)
    ms = n0 + nkk + 1j * imagpart

    return mie_func(ms, wavelength, r)
