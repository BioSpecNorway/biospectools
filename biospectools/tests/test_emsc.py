import os
import collections

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
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
    return (wavenumbers - np.mean(wavenumbers)) / half_rng


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
            reference=base_spectrum, constituents=constituent[None],
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
        corrected = emsc.transform(multiplied_spectra)

        mean = mult_coefs.mean()
        assert_array_almost_equal(corrected, spectra * mean)
        assert_array_almost_equal(emsc.scaling_coefs_ * mean, mult_coefs[:, 0])
        assert_array_almost_equal(emsc.coefs_[:, 0] * mean, mult_coefs[:, 0])

    def test_multiplicative_correction_with_reference(
            self, wavenumbers, base_spectrum, multiplied_spectra,
            spectra, mult_coefs):
        emsc = EMSC(base_spectrum, poly_order=0)
        corrected = emsc.transform(multiplied_spectra)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(emsc.scaling_coefs_, mult_coefs[:, 0])
        assert_array_almost_equal(emsc.coefs_[:, 0], mult_coefs[:, 0])

    def test_linear_correction(
            self, wavenumbers, base_spectrum, spectra_linear_effect,
            spectra, linear_coefs):
        emsc = EMSC(base_spectrum, wavenumbers)
        corrected = emsc.transform(spectra_linear_effect)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(
            emsc.polynomial_coefs_[:, 1], linear_coefs[:, 0])
        assert_array_almost_equal(emsc.coefs_[:, 2], linear_coefs[:, 0])
        assert emsc.constituents_coefs_ is None

    def test_constituents(
            self, wavenumbers, base_spectrum,
            spectra_with_constituent, constituent,
            spectra, constituent_coefs):
        emsc = EMSC(base_spectrum, wavenumbers,
                    poly_order=None, constituents=constituent[None])
        corrected = emsc.transform(spectra_with_constituent)

        assert_array_almost_equal(corrected, spectra)
        assert_array_almost_equal(
            emsc.constituents_coefs_[:, 0], constituent_coefs[:, 0])
        assert_array_almost_equal(emsc.coefs_[:, 1], constituent_coefs[:, 0])
        assert emsc.polynomial_coefs_ is None

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
        corrected = emsc.transform(raw_spectra)

        # scale coefficients to Achim's implementation
        emsc.coefs_[:, 2] *= -1
        emsc.coefs_[:, 4] *= 2

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(emsc.residuals_, residuals_standard)
        assert_array_almost_equal(
            emsc_quartic_params[:, :-1], emsc.polynomial_coefs_)
        assert_array_almost_equal(
            emsc_quartic_params[:, -1], emsc.scaling_coefs_)

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
        corrected = emsc.transform(raw_spectra)

        # scale coefficients to Achim's implementation
        emsc.coefs_[:, 2] *= -1

        assert_array_almost_equal(corrected, corrected_standard)
        assert_array_almost_equal(emsc.residuals_, residuals_standard)
        assert_array_almost_equal(
            emsc_weights_params[:, :-3], emsc.polynomial_coefs_)
        assert_array_almost_equal(
            emsc_weights_params[:, -3], emsc.scaling_coefs_)
