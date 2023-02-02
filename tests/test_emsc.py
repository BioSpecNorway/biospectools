import os
import collections

import pytest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pandas as pd

from biospectools.preprocessing.emsc import emsc
from biospectools.preprocessing import EMSC


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def emsc_data():
    return pd.read_excel(
        os.path.join(DATA_PATH, 'emsc_testdata.xlsx'),
        index_col=0, header=None)


@pytest.fixture
def emsc_data_params():
    return pd.read_excel(
        os.path.join(DATA_PATH, 'emsc_testdata_parameters.xlsx'),
        index_col=0, header=None)


@pytest.fixture
def emsc_quartic_params(emsc_data_params):
    return emsc_data_params.iloc[0:2].values


@pytest.fixture
def emsc_weights_params(emsc_data_params):
    return emsc_data_params.iloc[2:4].values


@pytest.fixture
def wavenumbers():
    return np.linspace(500, 3500, 100)


@pytest.fixture
def norm_wns(wavenumbers):
    half_rng = np.abs(wavenumbers[0] - wavenumbers[-1]) / 2
    mid_point = (wavenumbers.min() + wavenumbers.max()) / 2
    return (wavenumbers - mid_point) / half_rng


@pytest.fixture
def base_spectrum(wavenumbers):
    return np.random.uniform(0, 2, len(wavenumbers))


@pytest.fixture
def constituent(wavenumbers):
    return np.random.uniform(0, 2, len(wavenumbers))


@pytest.fixture
def spectra(base_spectrum):
    return np.array([base_spectrum, base_spectrum, base_spectrum])


@pytest.fixture
def mult_coefs():
    return np.array([1, 3, 5])[:, None]


@pytest.fixture
def linear_coefs():
    return np.array([0, -0.616, 2])[:, None]


@pytest.fixture
def constituent_coefs():
    return np.array([0, -1.5, 3])[:, None]


@pytest.fixture
def multiplied_spectra(spectra, mult_coefs):
    return spectra * mult_coefs


@pytest.fixture
def spectra_linear_effect(norm_wns, spectra, linear_coefs):
    return spectra + norm_wns * linear_coefs


@pytest.fixture
def spectra_with_constituent(spectra, constituent, constituent_coefs):
    return spectra + constituent * constituent_coefs


class TestEmscFunction:
    EmscResult = collections.namedtuple('EmscResult', ['corrs', 'coefs'])

    @pytest.fixture
    def mult_corrected(self, wavenumbers, multiplied_spectra):
        return self.EmscResult(*emsc(
            multiplied_spectra, wavenumbers, return_coefs=True))

    @pytest.fixture
    def mult_corrected_with_reference(
            self, wavenumbers, base_spectrum, multiplied_spectra):
        return self.EmscResult(*emsc(
            multiplied_spectra, wavenumbers,
            reference=base_spectrum, return_coefs=True))

    @pytest.fixture
    def linear_corrected(
            self, wavenumbers, base_spectrum, spectra_linear_effect):
        return self.EmscResult(*emsc(
            spectra_linear_effect, wavenumbers,
            reference=base_spectrum, return_coefs=True))

    @pytest.fixture
    def constituent_corrected(
            self, wavenumbers, base_spectrum,
            spectra_with_constituent, constituent):
        return self.EmscResult(*emsc(
            spectra_with_constituent, wavenumbers, poly_order=None,
            reference=base_spectrum, interferents=constituent[None],
            return_coefs=True))

    def test_multiplicative_correction(
            self, spectra, mult_corrected, mult_coefs):
        mean = mult_coefs.mean()
        assert_array_almost_equal(mult_corrected.corrs, spectra * mean)
        assert_array_almost_equal(mult_corrected.coefs[:, [0]] * mean, mult_coefs)

    def test_multiplicative_correction_with_reference(
            self, spectra, mult_corrected_with_reference, mult_coefs):
        assert_array_almost_equal(mult_corrected_with_reference.corrs, spectra)
        assert_array_almost_equal(
            mult_corrected_with_reference.coefs[:, [0]], mult_coefs)

    def test_linear_correction(
            self, spectra, linear_corrected, linear_coefs):
        assert_array_almost_equal(linear_corrected.corrs, spectra)
        assert_array_almost_equal(linear_corrected.coefs[:, [2]], linear_coefs)

    def test_constituents(
            self, spectra, constituent_corrected, constituent_coefs):
        assert_array_almost_equal(constituent_corrected.corrs, spectra)
        assert_array_almost_equal(
            constituent_corrected.coefs[:, [1]], constituent_coefs)

    def test_quartic_correction(self, emsc_data, emsc_quartic_params):
        """
        Test against gold standard: EMSC implementation in Matlab
        from Achim Kohler (06.02.2020)
        """
        wns = emsc_data.iloc[0].values
        raw_spectra = emsc_data.iloc[2:4].values

        corrected_standard = emsc_data.iloc[4:6].values
        residuals_standard = emsc_data.iloc[6:8].values

        corrected, coefs, residuals = emsc(
            raw_spectra, wns, poly_order=4,
            return_coefs=True, return_residuals=True)

        # scale coefficients to Achim's implementation
        coefs[:, 2] *= -1
        coefs[:, 4] *= 2

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(residuals, residuals_standard)
        assert_array_almost_equal(emsc_quartic_params[:, :-1], coefs[:, 1:])
        assert_array_almost_equal(emsc_quartic_params[:, -1], coefs[:, 0])

    def test_weighting(self, emsc_data, emsc_weights_params):
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

        corrected, coefs, residuals = emsc(
            raw_spectra, wns, weights=weights,
            return_coefs=True, return_residuals=True)

        # scale coefficients to Achim's implementation
        coefs[:, 2] *= -1

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(residuals, residuals_standard)
        assert_array_almost_equal(emsc_weights_params[:, :-3], coefs[:, 1:])
        assert_array_almost_equal(emsc_weights_params[:, -3], coefs[:, 0])


class TestEmscClass:

    def test_multiplicative_correction(
            self, wavenumbers, multiplied_spectra,
            spectra, mult_coefs):
        emsc = EMSC(multiplied_spectra.mean(axis=0), wavenumbers)
        corrected, inn = emsc.transform(multiplied_spectra, details=True)

        mean = mult_coefs.mean()
        assert_array_almost_equal(corrected, spectra * mean)
        assert_array_almost_equal(inn.scaling_coefs * mean, mult_coefs[:, 0])
        assert_array_almost_equal(inn.coefs[:, 0] * mean, mult_coefs[:, 0])

    def test_multiplicative_correction_with_reference(
            self, wavenumbers, base_spectrum, multiplied_spectra,
            spectra, mult_coefs):
        emsc = EMSC(base_spectrum, poly_order=0)
        corrected, inn = emsc.transform(multiplied_spectra, details=True)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(inn.scaling_coefs, mult_coefs[:, 0])
        assert_array_almost_equal(inn.coefs[:, 0], mult_coefs[:, 0])

    def test_linear_correction(
            self, wavenumbers, base_spectrum, spectra_linear_effect,
            spectra, linear_coefs):
        emsc = EMSC(base_spectrum, wavenumbers)
        corrected, inn = emsc.transform(spectra_linear_effect, details=True)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(
            inn.polynomial_coefs[:, 1], linear_coefs[:, 0])
        assert_array_almost_equal(inn.coefs[:, 2], linear_coefs[:, 0])
        with pytest.raises(AttributeError):
            inn.interferents_coefs

    def test_constituents(
            self, wavenumbers, base_spectrum,
            spectra_with_constituent, constituent,
            spectra, constituent_coefs):
        emsc = EMSC(base_spectrum, wavenumbers,
                    poly_order=None, interferents=constituent[None])
        corrected, inn = emsc.transform(
            spectra_with_constituent, details=True)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(
            inn.interferents_coefs[:, 0], constituent_coefs[:, 0])
        assert_array_almost_equal(inn.coefs[:, 1], constituent_coefs[:, 0])
        with pytest.raises(AttributeError):
            inn.polynomial_coefs

    def test_analytes(
            self, wavenumbers, base_spectrum,
            spectra_with_constituent, constituent,
            spectra, constituent_coefs):
        emsc = EMSC(base_spectrum, wavenumbers,
                    poly_order=None, analytes=constituent[None])
        corrected, inn = emsc.transform(
            spectra_with_constituent, details=True)

        assert_array_almost_equal(corrected, spectra_with_constituent)
        assert_array_almost_equal(
            inn.analytes_coefs[:, 0], constituent_coefs[:, 0])
        assert_array_almost_equal(inn.coefs[:, 1], constituent_coefs[:, 0])
        with pytest.raises(AttributeError):
            inn.polynomial_coefs

    def test_emsc_parameters(self, emsc_data, emsc_quartic_params):
        """
        Test against gold standard: EMSC implementation in Matlab
        from Achim Kohler (06.02.2020)
        """
        wns = emsc_data.iloc[0].values
        raw_spectra = emsc_data.iloc[2:4].values
        weights = emsc_data.iloc[1].values

        _, quartic_params = emsc(raw_spectra, wns,
                                 poly_order=4, return_coefs=True)
        _, weights_params = emsc(raw_spectra, wns,
                                 weights=weights, return_coefs=True)

    def test_quartic_correction(self, emsc_data, emsc_quartic_params):
        """
        Test against gold standard: EMSC implementation in Matlab
        from Achim Kohler (06.02.2020)
        """
        wns = emsc_data.iloc[0].values
        raw_spectra = emsc_data.iloc[2:4].values

        corrected_standard = emsc_data.iloc[4:6].values
        residuals_standard = emsc_data.iloc[6:8].values

        emsc = EMSC(raw_spectra.mean(axis=0), wns, poly_order=4)
        corrected, inn = emsc.transform(raw_spectra, details=True)

        # scale coefficients to Achim's implementation
        inn.coefs[:, 2] *= -1
        inn.coefs[:, 4] *= 2

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(inn.residuals, residuals_standard)
        assert_array_almost_equal(
            emsc_quartic_params[:, :-1], inn.polynomial_coefs)
        assert_array_almost_equal(
            emsc_quartic_params[:, -1], inn.scaling_coefs)

    def test_weighting(self, emsc_data, emsc_weights_params):
        """
        Test against gold standard: EMSC implementation in Matlab
        from Achim Kohler (06.02.2020)
        """
        wns = emsc_data.iloc[0].values
        raw_spectra = emsc_data.iloc[2:4].values
        weights = emsc_data.iloc[1].values

        corrected_standard = emsc_data.iloc[8:10].values
        residuals_standard = emsc_data.iloc[10:12].values

        emsc = EMSC(raw_spectra.mean(axis=0), wns, weights=weights)
        corrected, inn = emsc.transform(raw_spectra, details=True)

        # scale coefficients to Achim's implementation
        inn.coefs[:, 2] *= -1

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(inn.residuals, residuals_standard)
        assert_array_almost_equal(
            emsc_weights_params[:, :-3], inn.polynomial_coefs)
        assert_array_almost_equal(
            emsc_weights_params[:, -3], inn.scaling_coefs)

    def test_reshaping(
            self, base_spectrum, spectra_with_constituent, constituent,
            spectra, constituent_coefs):
        emsc = EMSC(base_spectrum, poly_order=None,
                    interferents=constituent[None])

        # 1d input array
        raw_spectrum = spectra_with_constituent[0]
        corrected, inn = emsc.transform(raw_spectrum, details=True)
        assert corrected.shape == raw_spectrum.shape
        assert inn.coefs.shape == (2,)
        assert_almost_equal(inn.scaling_coefs, 1)
        assert_almost_equal(inn.interferents_coefs, constituent_coefs[0, 0])
        assert_array_almost_equal(corrected, spectra[0])
        with pytest.raises(AttributeError):
            inn.polynomial_coefs

        # 3d input array
        shape = spectra_with_constituent.shape
        shape_3d = (shape[0], 5, shape[1])
        raw_spectra = np.broadcast_to(
            spectra_with_constituent[:, None], shape_3d)
        const_coefs_3d = np.broadcast_to(
            constituent_coefs[:, None],
            shape_3d[:2] + constituent_coefs.shape[-1:])
        gt = np.broadcast_to(spectra[:, None], shape_3d)
        corrected, inn = emsc.transform(raw_spectra, details=True)
        assert corrected.shape == raw_spectra.shape
        assert inn.coefs.shape == shape_3d[:2] + (2,)
        assert_almost_equal(inn.scaling_coefs, np.ones(shape_3d[:2]))
        assert_almost_equal(inn.interferents_coefs, const_coefs_3d)
        assert_array_almost_equal(corrected, gt)
        with pytest.raises(AttributeError):
            inn.polynomial_coefs

    def test_rebuild_model(self, base_spectrum, multiplied_spectra):
        emsc = EMSC(base_spectrum, poly_order=0, rebuild_model=False)
        with patch.object(EMSC, '_build_model', wraps=emsc._build_model) as mock:
            emsc.transform(multiplied_spectra)
            assert mock.call_count == 1
            emsc.transform(multiplied_spectra)
            assert mock.call_count == 1
            emsc.rebuild_model = True
            emsc.transform(multiplied_spectra)
            assert mock.call_count == 2
