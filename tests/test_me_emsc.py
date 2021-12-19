import os
from typing import Optional

import pytest
import numpy as np
from scipy.interpolate import interp1d
from biospectools.preprocessing.me_emsc import MeEMSC, MeEMSCInternals
from biospectools.preprocessing.criterions import \
    MatlabStopCriterion, TolStopCriterion


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def at_wavenumbers(
        from_wavenumbers: np.ndarray,
        to_wavenumbers: np.ndarray,
        spectra: np.ndarray,
        extrapolation_mode: Optional[str] = None,
        extrapolation_value: int = 0) -> np.ndarray:
    """
    Interpolates spectrum at another wavenumbers
    :param from_wavenumbers: initial wavenumbers
    :param to_wavenumbers: to what wavenumbers interpolate
    :param spectra: spectra
    :param extrapolation_mode: 'constant' or 'boundary' or None (then raise error)
    :param extrapolation_value: value to which interpolate in case of
    constant extrapolation
    """
    to_max = to_wavenumbers.max()
    to_min = to_wavenumbers.min()
    if from_wavenumbers[0] > from_wavenumbers[-1]:
        from_wavenumbers = from_wavenumbers[::-1]
        spectra = spectra[..., ::-1]
    if to_max > from_wavenumbers.max():
        if extrapolation_mode is None:
            raise ValueError('Range of to_wavenumbers exceeds '
                             'boundaries of from_wavenumbers')
        from_wavenumbers = np.append(from_wavenumbers, to_max)
        if extrapolation_mode == 'constant':
            spectra = np.append(spectra, [extrapolation_value], axis=-1)
        elif extrapolation_mode == 'boundary':
            spectra = np.append(spectra, [spectra[..., -1]], axis=-1)
        else:
            raise ValueError(f'Unknown extrapolation_mode {extrapolation_mode}')
    if to_min < from_wavenumbers.min():
        if extrapolation_mode is None:
            raise ValueError('Range of to_wavenumbers exceeds '
                             'boundaries of from_wavenumbers')
        from_wavenumbers = np.insert(from_wavenumbers, 0, to_min)
        if extrapolation_mode == 'constant':
            spectra = np.insert(spectra, 0, [extrapolation_value], axis=-1)
        elif extrapolation_mode == 'boundary':
            spectra = np.insert(spectra, 0, [spectra[..., 0]], axis=-1)
        else:
            raise ValueError(f'Unknown extrapolation_mode {extrapolation_mode}')

    return interp1d(from_wavenumbers, spectra)(to_wavenumbers)


@pytest.fixture()
def criterion_empty():
    return TolStopCriterion(3, 0, 0)


@pytest.fixture()
def emsc_internals_mock():
    from unittest import mock
    inn_mock = mock.Mock()
    inn_mock.coefs = np.random.rand(1, 10)
    inn_mock.residuals = np.random.rand(1, 100)
    return inn_mock


@pytest.fixture()
def criterion_unfinished(emsc_internals_mock):
    criterion = TolStopCriterion(3, 0, 0)
    v = {'corrected': 1, 'internals': emsc_internals_mock, 'emsc': None}
    criterion.add(score=0.9, value=v)
    assert not bool(criterion)
    return criterion


@pytest.fixture()
def criterion_finished(emsc_internals_mock):
    criterion = TolStopCriterion(3, 0, 0)
    v = {'corrected': 1, 'internals': emsc_internals_mock, 'emsc': None}
    criterion.add(score=0.9, value=v)
    criterion.add(score=0.6, value=v)
    criterion.add(score=0.6, value=v)
    assert bool(criterion)
    return criterion


def test_me_emsc_internals_only_invalid_criterions(criterion_empty):
    inn = MeEMSCInternals(
        [criterion_empty, criterion_empty],
        n_mie_components=2)
    assert inn.coefs.shape == (2,)
    assert np.all(np.isnan(inn.coefs[0]))
    assert np.all(np.isnan(inn.coefs[1]))


def test_me_emsc_internals_with_invalid_criterions(
        criterion_empty, criterion_unfinished, criterion_finished):
    inn = MeEMSCInternals(
        [criterion_empty, criterion_unfinished, criterion_finished],
        n_mie_components=3)
    assert inn.coefs.shape == (3, 10)
    assert np.all(np.isnan(inn.coefs[0]))
    assert np.all(~np.isnan(inn.coefs[1]))
    assert np.all(~np.isnan(inn.coefs[2]))


@pytest.fixture
def matlab_reference_spectra():
    wns, spectrum = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd1_rawSpec.csv"),
        usecols=np.arange(1, 779), delimiter=",")
    spectra = spectrum[None]

    wns_ref, reference = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd2_refSpec.csv"),
        usecols=np.arange(1, 752), delimiter=",")
    reference = at_wavenumbers(wns_ref, wns, reference.reshape(1, -1))[0]
    return wns, spectra, reference


@pytest.fixture
def matlab_results():
    # to save space saved only 20th wavenumbers values
    default_spec, ncomp14_spec, fixed_iter3_spec = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd3_corr.csv"),
        usecols=np.arange(1, 40), delimiter=",")

    default_coefs, ncomp14_coefs, fixed_iter3_coefs = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd4_param.csv"),
        usecols=np.arange(1, 17), delimiter=",")
    default_coefs = default_coefs[~np.isnan(default_coefs)]
    fixed_iter3_coefs = fixed_iter3_coefs[~np.isnan(fixed_iter3_coefs)]

    default_resid, ncomp14_resid, fixed_iter3_resid = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd5_residuals.csv"),
        usecols=np.arange(1, 40), delimiter=",")

    d_niter, n_niter, fixed_niter = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd6_niter.csv"),
        usecols=(1,), delimiter=",", dtype=np.int64)
    d_rmse, n_rmse, fixed_rmse = np.loadtxt(
        os.path.join(DATA_PATH, "memsc_test_data/MieStd7_RMSE.csv"),
        usecols=(1,), delimiter=",", dtype=float)

    return {
        'default': (default_spec, default_coefs, default_resid,
                    d_niter, d_rmse),
        'ncomp14': (ncomp14_spec, ncomp14_coefs, ncomp14_resid,
                    n_niter, n_rmse),
        'fixed_iter3': (fixed_iter3_spec, fixed_iter3_coefs, fixed_iter3_resid,
                        fixed_niter, fixed_rmse),
    }


@pytest.fixture
def default_result(matlab_reference_spectra, matlab_results):
    wns, spectra, reference = matlab_reference_spectra

    me_emsc = MeEMSC(reference=reference, wavenumbers=wns)
    me_emsc.stop_criterion = MatlabStopCriterion(max_iter=45, precision=4)
    preproc, internals = me_emsc.transform(spectra, internals=True)
    return me_emsc, preproc[0, ::20].T, internals


@pytest.fixture
def inverse_wns_result(matlab_reference_spectra, matlab_results):
    wns, spectra, reference = matlab_reference_spectra

    idxs = np.arange(len(wns))[::-1]
    me_emsc = MeEMSC(reference=reference[idxs], wavenumbers=wns[idxs])
    me_emsc.stop_criterion = MatlabStopCriterion(max_iter=45, precision=4)
    preproc, internals = me_emsc.transform(spectra[:, idxs], internals=True)
    unshuffled = preproc[:, idxs]
    return me_emsc, unshuffled[0, ::20].T, internals


@pytest.fixture
def ncomp14_result(matlab_reference_spectra, matlab_results):
    wns, spectra, reference = matlab_reference_spectra

    me_emsc = MeEMSC(reference=reference, wavenumbers=wns, n_components=14)
    me_emsc.stop_criterion = MatlabStopCriterion(max_iter=30, precision=4)
    preproc, internals = me_emsc.transform(spectra, internals=True)
    return me_emsc, preproc[0, ::20].T, internals


@pytest.fixture
def fixed_iter3_result(matlab_reference_spectra, matlab_results):
    wns, spectra, reference = matlab_reference_spectra

    me_emsc = MeEMSC(reference=reference, wavenumbers=wns, max_iter=1)
    preproc, internals = me_emsc.transform(spectra, internals=True)
    return me_emsc, preproc[0, ::20].T, internals


@pytest.mark.parametrize(
    "python_result,params_set",
    [
        ('default_result', 'default'),
        ('inverse_wns_result', 'default'),
        ('ncomp14_result', 'ncomp14'),
        ('fixed_iter3_result', 'fixed_iter3')
    ])
def test_compare_with_matlab(
        python_result, matlab_results, params_set, request):
    me_emsc, preproc, internals = request.getfixturevalue(python_result)
    coefs = _matlab_ordered_coefs(internals)[0]
    gt_spec, gt_coefs, gt_resid, gt_niter, gt_rmse = matlab_results[params_set]
    np.testing.assert_almost_equal(gt_spec, preproc)
    np.testing.assert_almost_equal(np.abs(gt_coefs), np.abs(coefs))
    np.testing.assert_equal(gt_niter, internals.n_iterations)
    np.testing.assert_equal(gt_rmse, np.round(internals.rmses, decimals=4))


def _matlab_ordered_coefs(inn: MeEMSCInternals):
    return np.concatenate((
        inn.polynomial_coefs,
        inn.mie_components_coefs,
        inn.scaling_coefs[:, None]), axis=1)
