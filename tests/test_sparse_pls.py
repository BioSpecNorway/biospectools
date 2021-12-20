import os

from numpy.testing import assert_array_almost_equal
from scipy import io
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from biospectools.models import SparsePLSRegression


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_sparse_pls():
    mat = io.loadmat(os.path.join(DATA_PATH, 'test_spls_data.mat'))
    spectra = mat['spectra']
    labels = mat['labels']

    n_components = int(mat['num_LV'][0][0])
    sparsity = mat['sparsity'][0][0]

    numeric_labels = LabelEncoder().fit_transform(labels)
    dummy_labels = OneHotEncoder().fit_transform(numeric_labels.reshape(-1, 1))

    spls = SparsePLSRegression(
        n_components, sparsity, sign_stable_weights=False)
    spls.fit(spectra, dummy_labels.toarray())

    assert_array_almost_equal(mat['W'], spls.x_weights_)
    assert_array_almost_equal(mat['P'], spls.x_loadings_)
    assert_array_almost_equal(mat['Q'], spls.y_loadings_)
    assert_array_almost_equal(mat['T'], spls.x_scores_)

    for i in range(mat['B'].shape[-1]):
        assert_array_almost_equal(mat['B'][:, :, i], spls.beta(i + 1))
        assert_array_almost_equal(mat['B'][:, :, i], spls.coefs(i + 1))
