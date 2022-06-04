import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from biospectools.preprocessing import DSAE


pytest.importorskip('tensorflow', minversion='2.3.4',
                    reason='Tensorflow is not installed')


@pytest.fixture
def wavenumbers():
    return np.linspace(500, 3500, 1408)


@pytest.fixture
def base_spectrum(wavenumbers):
    return np.random.uniform(0, 2, len(wavenumbers))


def test_reshaping(wavenumbers, base_spectrum):
    dsae = DSAE(wavenumbers)

    single_spectrum = dsae.transform(base_spectrum, wns=None)
    assert single_spectrum.shape == (len(wavenumbers),)

    spectra_list = dsae.transform(base_spectrum[None], wns=None)
    assert spectra_list.shape == (1, len(wavenumbers))
    assert np.all(spectra_list[0] == single_spectrum)

    complex_shape = (3, 9, len(wavenumbers), 4)
    complex_spectra = dsae.transform(
        np.broadcast_to(base_spectrum[:, None], complex_shape),
        wns=None, axis=2)
    assert complex_spectra.shape == complex_shape
    assert_array_almost_equal(
        complex_spectra,
        np.broadcast_to(single_spectrum[:, None], complex_shape))


def test_interpolation(wavenumbers, base_spectrum):
    extended_wns = np.concatenate((wavenumbers, [3501, 3502]))
    extended_spec = np.concatenate((base_spectrum, [1, 2]))

    dsae = DSAE(wavenumbers)
    gt = dsae.transform(base_spectrum, wns=None)

    with pytest.raises(ValueError):
        dsae.transform(extended_spec, wns=None)
    with pytest.raises(ValueError):
        dsae.transform(extended_spec, extended_wns)
    with pytest.raises(ValueError):
        dsae.transform(extended_spec, extended_wns, interpolate='intersect')

    result = dsae.transform(extended_spec, extended_wns, interpolate=True)
    assert np.all(result == gt)
