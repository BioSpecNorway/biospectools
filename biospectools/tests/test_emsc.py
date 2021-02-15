import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pandas as pd

from biospectools.preprocessing.emsc import emsc


def test_multiplicative_correction():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5,
    ])
    wavenumbers = np.array(range(n_channels))
    corrected_spectra = emsc(spectra, wavenumbers)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum * 3)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum * 3)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum * 3)


def test_reference_spectrum():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5,
    ])
    wavenumbers = np.array(range(n_channels))

    corrected_spectra = emsc(spectra, wavenumbers, reference=base_spectrum)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum)


def test_linear_correction():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    linear_coef = -0.6135

    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5 + np.linspace(-1, 1, 100) * linear_coef,
    ])
    wavenumbers = np.array(range(n_channels))
    corrected_spectra, coefs = emsc(spectra, wavenumbers,
                                    reference=base_spectrum,
                                    return_coefs=True)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum)

    # check linear coefs
    assert_almost_equal(coefs[0, 2], 0)
    assert_almost_equal(coefs[1, 2], 0)
    assert_almost_equal(coefs[2, 2], linear_coef)


def test_constituents():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    constituent = np.random.uniform(0, 2, n_channels)

    spectra = np.array([
        base_spectrum,
        base_spectrum * 3 + constituent * 2 + np.linspace(-1, 1, n_channels) * 4
    ])
    constituents = np.array([
        constituent
    ])
    wavenumbers = np.array(range(n_channels))

    corrected_spectra, coefs = emsc(spectra, wavenumbers,
                                    reference=base_spectrum,
                                    constituents=constituents,
                                    return_coefs=True)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)

    # check coefs
    assert_almost_equal(coefs[1, 0], 3)
    assert_almost_equal(coefs[1, 1], 2)
    assert_almost_equal(coefs[1, 2], 0)
    assert_almost_equal(coefs[1, 3], 4)
    assert_almost_equal(coefs[1, 4], 0)


def test_emsc_parameters():
    """
    Test against gold standard: EMSC implementation in Matlab
    from Achim Kohler (06.02.2020)
    """
    data_path = os.path.join(DATA_PATH, 'emsc_testdata.xlsx')
    params_path = os.path.join(DATA_PATH, 'emsc_testdata_parameters.xlsx')
    emsc_data = pd.read_excel(data_path, index_col=0, header=None)
    emsc_data_params = pd.read_excel(params_path, index_col=0, header=None)

    wns = emsc_data.iloc[0].values
    raw_spectra = emsc_data.iloc[2:4].values
    weights = emsc_data.iloc[1].values

    quartic_params_standard = emsc_data_params.iloc[0:2].values
    weigths_params_standard = emsc_data_params.iloc[2:4].values

    _, quartic_params = emsc(raw_spectra, wns,
                             poly_order=4, return_coefs=True)
    _, weights_params = emsc(raw_spectra, wns,
                             weights=weights, return_coefs=True)

    # scale coefficients to Achim's implementation
    quartic_params[:, 2] *= -1
    quartic_params[:, 4] *= 2
    weights_params[:, 2] *= -1

    assert_array_almost_equal(quartic_params_standard[:, :-1], quartic_params[:, 1:])
    assert_array_almost_equal(quartic_params_standard[:, -1], quartic_params[:, 0])
    assert_array_almost_equal(weigths_params_standard[:, :-3], weights_params[:, 1:])
    assert_array_almost_equal(weigths_params_standard[:, -3], weights_params[:, 0])


def test_quartic_correction():
    """
    Test against gold standard: EMSC implementation in Matlab
    from Achim Kohler (06.02.2020)
    """
    emsc_data = pd.read_excel(os.path.join(DATA_PATH, 'emsc_testdata.xlsx'),
                              index_col=0, header=None)

    wns = emsc_data.iloc[0].values
    raw_spectra = emsc_data.iloc[2:4].values

    corrected_standard = emsc_data.iloc[4:6].values
    residuals_standard = emsc_data.iloc[6:8].values

    corrected, residuals = emsc(raw_spectra, wns,
                                poly_order=4, return_residuals=True)

    assert_array_almost_equal(corrected, corrected_standard)
    assert_array_almost_equal(residuals, residuals_standard)


def test_weighting():
    """
    Test against gold standard: EMSC implementation in Matlab
    from Achim Kohler (06.02.2020)
    """
    emsc_data = pd.read_excel(os.path.join(DATA_PATH, 'emsc_testdata.xlsx'),
                              index_col=0, header=None)

    wns = emsc_data.iloc[0].values
    raw_spectra = emsc_data.iloc[2:4].values
    weights = emsc_data.iloc[1].values

    corrected_standard = emsc_data.iloc[8:10].values
    residuals_standard = emsc_data.iloc[10:12].values

    corrected, residuals = emsc(raw_spectra, wns,
                                weights=weights, return_residuals=True)

    assert_array_almost_equal(corrected, corrected_standard)
    assert_array_almost_equal(residuals, residuals_standard)