import os

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from biospectools.preprocessing import FringeEMSC


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def lorentzian(wn, center, amplitude, width):
    return amplitude * width**2 / (width**2 + 4*(wn - center)**2)


@pytest.fixture
def wavenumbers():
    return np.arange(500, 3500)


@pytest.fixture
def freqs():
    return np.array([0.01, 0.02]) * 2 * np.pi


@pytest.fixture
def amps():
    return np.array([[0.3, 0.2], [0.1, 0]])


@pytest.fixture
def pure_spectrum(wavenumbers):
    return lorentzian(wavenumbers, 700, 1, 20)


@pytest.fixture
def constituent(wavenumbers):
    p1 = lorentzian(wavenumbers, 1200, 0.3, 20)
    p2 = lorentzian(wavenumbers, 1300, 0.4, 10)
    return p1 + p2


@pytest.fixture
def spectrum_with_fringe(wavenumbers, pure_spectrum, constituent, freqs, amps):
    base = pure_spectrum / 10 + constituent * 3
    sines = np.sin(wavenumbers * freqs[:, None]) * amps[:, :1]
    coses = np.cos(wavenumbers * freqs[:, None]) * amps[:, 1:]
    return base + sines.sum(axis=0) + coses.sum(axis=0)


def test_fring_emsc(
        wavenumbers, pure_spectrum, constituent,
        spectrum_with_fringe, freqs, amps):
    fringe_emsc = FringeEMSC(
        pure_spectrum, wavenumbers, fringe_wn_location=(1724, 1924),
        interferents=constituent[None],
        n_freq=2, double_freq=True)
    corrected, inn = fringe_emsc.transform(
        spectrum_with_fringe[None], details=True)
    corrected = corrected[0]

    np.testing.assert_almost_equal(corrected, pure_spectrum)
    assert len(inn.freqs[0]) == 4
    np.testing.assert_almost_equal(inn.freqs[0, :2], freqs)
    np.testing.assert_almost_equal(inn.freqs_coefs[0, :2], amps)
    np.testing.assert_almost_equal(inn.freqs_coefs[0, 2:], 0)
    np.testing.assert_almost_equal(inn.scaling_coefs[0], 0.1)
    np.testing.assert_almost_equal(inn.interferents_coefs[0, 0], 3)

    coefs = np.concatenate((
        inn.scaling_coefs[:, None],
        inn.freqs_coefs.reshape(1, -1),
        inn.interferents_coefs,
        inn.polynomial_coefs
    ), axis=1)
    np.testing.assert_almost_equal(inn.coefs, coefs)


def test_reshaping(
        wavenumbers, pure_spectrum, constituent,
        spectrum_with_fringe, freqs, amps):

    fringe_emsc = FringeEMSC(
        pure_spectrum, wavenumbers, fringe_wn_location=(1724, 1924),
        interferents=constituent[None],
        n_freq=2, double_freq=True)
    n_coefs = 1 + 4*2 + 1 + 3

    # 1D-array
    corrected, inn = fringe_emsc.transform(
        spectrum_with_fringe, details=True)
    assert corrected.shape == pure_spectrum.shape
    assert inn.coefs.shape == (n_coefs,)
    assert_array_almost_equal(inn.freqs[:2], freqs)
    assert_array_almost_equal(inn.freqs_coefs[:2], amps)
    assert_almost_equal(inn.freqs_coefs[2:], 0)
    assert_almost_equal(inn.scaling_coefs, 0.1)
    assert_almost_equal(inn.interferents_coefs, 3)
    assert_array_almost_equal(corrected, pure_spectrum)

    # 3D-array
    shape_3d = (3, 5, len(pure_spectrum))
    raw_spectra = np.broadcast_to(spectrum_with_fringe, shape_3d)
    gt = np.broadcast_to(pure_spectrum, shape_3d)

    corrected, inn = fringe_emsc.transform(raw_spectra, details=True)
    assert corrected.shape == raw_spectra.shape
    assert inn.coefs.shape == shape_3d[:2] + (n_coefs,)
    assert inn.scaling_coefs.shape == shape_3d[:2]
    assert_array_almost_equal(
        inn.freqs[..., :2],
        np.broadcast_to(freqs, inn.freqs[..., :2].shape))
    assert_array_almost_equal(
        inn.freqs_coefs[..., :2, :],
        np.broadcast_to(amps, inn.freqs_coefs[..., :2, :].shape))
    assert_array_almost_equal(
        inn.freqs_coefs[..., 2:, :],
        np.broadcast_to(0, inn.freqs_coefs[..., 2:, :].shape))
    assert_array_almost_equal(
        inn.scaling_coefs,
        np.broadcast_to(0.1, inn.scaling_coefs.shape))
    assert_array_almost_equal(
        inn.interferents_coefs,
        np.broadcast_to(3, inn.interferents_coefs.shape))
    assert_array_almost_equal(corrected, gt)
