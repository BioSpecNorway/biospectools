import unittest
import numpy as np
from biospectools.preprocessing.me_emsc import MeEMSC
from biospectools.utils import at_wavenumbers
from biospectools.preprocessing.criterions import \
    MatlabStopCriterion, TolStopCriterion


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

        stop_criterion = MatlabStopCriterion(max_iter=45, precision=4)
        cls.f1 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            weights=None,
            stop_criterion=stop_criterion
        )
        cls.f1data = cls.f1.transform(cls.Spectra)
        cls.f1_inv = MeEMSC(
            reference=cls.reference[::-1],
            wavenumbers=cls.wnS[::-1],
            weights=None,
            stop_criterion=stop_criterion
        )
        cls.f1data_inv = cls.f1_inv.transform(cls.Spectra[:, ::-1])

        stop_criterion = MatlabStopCriterion(max_iter=30, precision=4)
        cls.f2 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            n_components=14,
            stop_criterion=stop_criterion
        )  # With weights
        cls.f2data = cls.f2.transform(cls.Spectra)

        cls.f3 = MeEMSC(
            reference=cls.reference,
            wavenumbers=cls.wnS,
            max_iter=1
        )
        cls.f3data = cls.f3.transform(cls.Spectra)

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
        n_comp = self.f1.mie_decomposer.svd.n_components
        np.testing.assert_almost_equal(
            abs(self.f1.coefs_[0, [-1, *range(1, n_comp + 1), 0]]),
            abs(self.param_default_20th_elem),
        )
        n_comp = self.f1_inv.mie_decomposer.svd.n_components
        np.testing.assert_almost_equal(
            abs(self.f1_inv.coefs_[0, [-1, *range(1, n_comp + 1), 0]]),
            abs(self.param_default_20th_elem),
        )
        n_comp = self.f2.mie_decomposer.svd.n_components
        np.testing.assert_almost_equal(
            abs(self.f2.coefs_[0, [-1, *range(1, n_comp + 1), 0]]),
            abs(self.param_14ncomp_20th_elem),
        )
        n_comp = self.f3.mie_decomposer.svd.n_components
        np.testing.assert_almost_equal(
            abs(self.f3.coefs_[0, [-1, *range(1, n_comp + 1), 0]]),
            abs(self.param_fixed_iter3_20th_elem),
        )

    def test_number_iterations(self):
        print("Test Iters")
        numiter = np.vstack((
            self.f1.n_iterations_,
            self.f2.n_iterations_,
            self.f3.n_iterations_))
        np.testing.assert_equal(numiter, self.numiter_std)

    def test_RMSE(self):
        RMSE = np.array([self.f1.rmse_, self.f2.rmse_, self.f3.rmse_])
        np.testing.assert_equal(np.round(RMSE, decimals=4), self.RMSE_std)

    def test_same_data_reference(self):
        # it was crashing before
        f = MeEMSC(reference=self.reference, wavenumbers=self.wnS)
        _ = f.transform(self.reference[None, :])

        # TODO fix for 1D input array


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
