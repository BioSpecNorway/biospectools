import pytest
import unittest
import numpy as np
from biospectools.preprocessing.me_emsc import MeEMSC, MeEMSCInternals
from biospectools.utils import at_wavenumbers
from biospectools.preprocessing.criterions import \
    MatlabStopCriterion, TolStopCriterion


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
    criterion.add(score=0.9, value=[1, emsc_internals_mock, 3])
    assert not bool(criterion)
    return criterion


@pytest.fixture()
def criterion_finished(emsc_internals_mock):
    criterion = TolStopCriterion(3, 0, 0)
    criterion.add(score=0.9, value=[1, emsc_internals_mock, 3])
    criterion.add(score=0.5, value=[1, emsc_internals_mock, 3])
    criterion.add(score=0.6, value=[1, emsc_internals_mock, 3])
    assert bool(criterion)
    return criterion


def test_me_emsc_internals_only_invalid_criterions(criterion_empty):
    inn = MeEMSCInternals([criterion_empty, criterion_empty])
    assert inn.coefs.shape == (2,)
    assert np.all(np.isnan(inn.coefs[0]))
    assert np.all(np.isnan(inn.coefs[1]))


def test_me_emsc_internals_with_invalid_criterions(
        criterion_empty, criterion_unfinished, criterion_finished):
    inn = MeEMSCInternals(
        [criterion_empty, criterion_unfinished, criterion_finished])
    assert inn.coefs.shape == (3, 10)
    assert np.all(np.isnan(inn.coefs[0]))
    assert np.all(~np.isnan(inn.coefs[1]))
    assert np.all(~np.isnan(inn.coefs[2]))


class TestME_EMSC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        path2data1 = "data/memsc_test_data/MieStd1_rawSpec.csv"
        path2data2 = "data/memsc_test_data/MieStd2_refSpec.csv"
        path2data3 = "data/memsc_test_data/MieStd3_corr.csv"
        path2data4 = "data/memsc_test_data/MieStd4_param.csv"
        path2data5 = "data/memsc_test_data/MieStd5_residuals.csv"
        path2data6 = "data/memsc_test_data/MieStd6_niter.csv"
        path2data7 = "data/memsc_test_data/MieStd7_RMSE.csv"

        v = np.loadtxt(path2data1, usecols=np.arange(1, 779), delimiter=",")
        cls.wnS = v[0]
        cls.Spectra = np.vstack((v[1], v[1]))

        v = np.loadtxt(path2data2, usecols=np.arange(1, 752), delimiter=",")
        cls.wnM = v[0]
        cls.Matrigel = v[1].reshape(1, -1)
        cls.reference = cls.Matrigel
        cls.wn_ref = cls.wnM

        v = np.loadtxt(path2data3, usecols=np.arange(1, 40), delimiter=",")
        cls.corr_default_20th_elem = v[0]
        cls.corr_14ncomp_20th_elem = v[1]
        cls.corr_fixed_iter3_20th_elem = v[2]

        v = np.loadtxt(path2data4, usecols=np.arange(1, 17), delimiter=",")
        cls.param_default_20th_elem = v[0][~np.isnan(v[0])]
        cls.param_14ncomp_20th_elem = v[1]
        cls.param_fixed_iter3_20th_elem = v[2][~np.isnan(v[2])]

        v = np.loadtxt(path2data5, usecols=np.arange(1, 40), delimiter=",")
        cls.res_default_20th_elem = v[0]
        cls.res_14ncomp_20th_elem = v[1]
        cls.res_fixed_iter3_20th_elem = v[2]

        cls.numiter_std = np.loadtxt(
            path2data6, usecols=(1,), delimiter=",", dtype="int64"
        )
        cls.RMSE_std = np.loadtxt(
            path2data7, usecols=(1,), delimiter=",", dtype="float"
        )

        cls.numiter_std = np.array([cls.numiter_std, cls.numiter_std]).T
        cls.RMSE_std = np.array([cls.RMSE_std, cls.RMSE_std]).T

        cls.reference = at_wavenumbers(cls.wn_ref, cls.wnS, cls.reference)
        cls.reference = cls.reference[0]

        cls.f1 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            weights=None,
        )
        cls.f1.stop_criterion = MatlabStopCriterion(max_iter=45, precision=4)
        cls.f1data, cls.f1inn = cls.f1.transform(cls.Spectra, internals=True)
        cls.f1_inv = MeEMSC(
            reference=cls.reference[::-1],
            wavenumbers=cls.wnS[::-1],
            weights=None
        )
        cls.f1_inv.stop_criterion = MatlabStopCriterion(max_iter=45, precision=4)
        cls.f1data_inv, cls.f1inn_inv = cls.f1_inv.transform(
            cls.Spectra[:, ::-1], internals=True)

        cls.f2 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            n_components=14
        )  # With weights
        cls.f2.stop_criterion = MatlabStopCriterion(max_iter=30, precision=4)
        cls.f2data, cls.f2inn = cls.f2.transform(cls.Spectra, internals=True)

        cls.f3 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            max_iter=1
        )
        cls.f3data, cls.f3inn = cls.f3.transform(cls.Spectra, internals=True)

    def disabled_test_plotting(self):
        import matplotlib.pyplot as plt

        # Default parameters
        plt.figure()
        plt.plot(self.wnS[0::20], self.f1data[0, 0::20].T, label="python")
        plt.plot(self.wnS[0::20], self.corr_default_20th_elem, label="matlab")
        plt.plot(
            self.wnS[0::20],
            self.f1data[0, 0::20].T[:, 0] - self.corr_default_20th_elem,
            label="diff",
        )
        plt.legend()
        plt.title("Comparison Matlab/Python - default parameters")

        # 14 principal components
        plt.figure()
        plt.plot(self.wnS[0::20], self.f2data[0, 0::20].T, label="python")
        plt.plot(self.wnS[0::20], self.corr_14ncomp_20th_elem, label="matlab")
        plt.plot(
            self.wnS[0::20],
            self.f2data[0, 0::20].T[:, 0] - self.corr_14ncomp_20th_elem,
            label="diff",
        )
        plt.legend()
        plt.title("Comparison Matlab/Python - 14 principal components")

        # Fixed iteration number 3
        plt.figure()
        plt.plot(self.wnS[0::20], self.f3data[0, 0::20].T, label="python")
        plt.plot(self.wnS[0::20], self.corr_fixed_iter3_20th_elem, label="matlab")
        plt.plot(
            self.wnS[0::20],
            self.f3data[0, 0::20].T[:, 0] - self.corr_fixed_iter3_20th_elem,
            label="diff",
        )
        plt.legend()
        plt.title("Comparison Matlab/Python - fixed iterations 3")
        plt.show()

    def test_correction_output(self):
        print("Test Correction")
        np.testing.assert_almost_equal(self.corr_default_20th_elem, self.f1data[0, ::20].T)
        np.testing.assert_almost_equal(self.corr_default_20th_elem, self.f1data_inv[0, ::-20].T)
        np.testing.assert_almost_equal(self.corr_14ncomp_20th_elem, self.f2data[0, ::20].T)
        np.testing.assert_almost_equal(self.corr_fixed_iter3_20th_elem, self.f3data[0, ::20].T)

    def test_EMSC_parameters(self):
        print("Test Parameters")
        np.testing.assert_almost_equal(
            abs(self._matlab_ordered_coefs(self.f1inn)[0]),
            abs(self.param_default_20th_elem),
        )
        np.testing.assert_almost_equal(
            abs(self._matlab_ordered_coefs(self.f1inn_inv)[0]),
            abs(self.param_default_20th_elem),
        )
        np.testing.assert_almost_equal(
            abs(self._matlab_ordered_coefs(self.f2inn)[0]),
            abs(self.param_14ncomp_20th_elem),
        )
        np.testing.assert_almost_equal(
            abs(self._matlab_ordered_coefs(self.f3inn)[0]),
            abs(self.param_fixed_iter3_20th_elem),
        )

    def test_number_iterations(self):
        print("Test Iters")
        numiter = np.vstack((
            self.f1inn.n_iterations,
            self.f2inn.n_iterations,
            self.f3inn.n_iterations))
        np.testing.assert_equal(numiter, self.numiter_std)

    def test_RMSE(self):
        RMSE = np.array([self.f1inn.rmses, self.f2inn.rmses, self.f3inn.rmses])
        np.testing.assert_equal(np.round(RMSE, decimals=4), self.RMSE_std)

    def test_same_data_reference(self):
        # it was crashing before
        f = MeEMSC(reference=self.reference, wavenumbers=self.wnS)
        _ = f.transform(self.reference[None, :])

        # TODO fix for 1D input array

    def _matlab_ordered_coefs(self, inn: MeEMSCInternals):
        return np.concatenate((
            inn.polynomial_coefs,
            inn.mie_components_coefs,
            inn.scaling_coefs[:, None]), axis=1)


"""
    def test_short_reference(self):
        wnMshort = self.wnM[0::30]
        Matrigelshort = self.Matrigel[0, 0::30]
        Matrigelshort = Matrigelshort.reshape(-1, 1).T
        # it was crashing before
        f = ME_EMSC(reference=Matrigelshort, wavenumbers=wnMshort)
        _ = f.transform(self.Spectra)

"""

if __name__ == "__main__":
    unittest.main()
