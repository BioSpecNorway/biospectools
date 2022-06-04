import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv1D, AveragePooling1D, MaxPooling1D, BatchNormalization,\
    Lambda, Conv2DTranspose, Activation, Layer, Input
from tensorflow.keras.regularizers import l1_l2

from biospectools.utils import downloads
from biospectools.utils.interpolate import interp2wns


__all__ = ['DSAE']


class DSAE:
    """
    Descattering Autoencoder (DSAE) [1]_. A neural network that can efficiently
    substitute ME-EMSC for Mie Scattering correction.

    Pre-trained models
    ------------------
    One pre-trained model is available that was trained on
    fungal data (Mucor circinelloides). It can be loaded using
    >>> dsae = DSAE.pretrained_on_fungi()

    Parameters
    ----------
    wavenumbers: `(N_wns,) array-like`
        A 1D-array with wavenumbers values that were used to train DSAE
    filters: `(N_layers,) integer array-like`
        List of number of filters for each convolutional layer
    kernel_sizes: `(N_layers,) integer array-like`
        List of kernel size for each convolutional layer
    strides: `(N_layers,) integer array-like`
        List of strides layers in each layer
    l2_reg: `float`, default 0
        l2 regularization for each kernel and bias convolutional layer.
    l1_reg: `float`, default 0
        l2 regularization for each kernel and bias convolutional layer.
    pooling: `str`, default 'adverage'
        Type of pooling layer. Can be 'average' or 'max'

    Other Parameters
    ----------------
    model : tf.keras.Model
        Built tensorflow model

    References
    ----------
    .. [1] Magnussen, Eirik Almklov, et al. *Deep convolutional neural
           network recovers pure absorbance spectra from highly
           scatter‐distorted spectra of cells.*
           Journal of Biophotonics 13.12 (2020): e202000204.
    """
    wavenumbers: np.ndarray
    model: tf.keras.Model

    @classmethod
    def pretrained_on_fungi(cls):
        """
        The model that was trained in the original article [1]_. The model was
        trained on fungal data (Mucor circinelloides).

        Returns
        -------
        dsae_model: DSAE
            DSAE model with loaded pre-trained weights.
        """
        weights_path, wns_path = _download_dsae_files()

        dsae = cls(wavenumbers=np.load(wns_path)['wn'].squeeze(), l2_reg=0.001)
        dsae.model.load_weights(weights_path)
        return dsae

    def __init__(
            self,
            wavenumbers,
            filters=None,
            kernel_sizes=None,
            strides=None,
            l2_reg=0.0,
            l1_reg=0.0,
            pooling='average'):
        if np.ndim(wavenumbers) != 1:
            raise ValueError(
                f'Wavenumbers must be 1d array (given {self.wavenumbers.ndim})')
        if pooling not in ['average', 'max']:
            raise ValueError(
                f'Unknown pooling type {pooling}. '
                f'Valid values are "average" or "max"')

        if filters is None:
            filters = [32, 64, 32, 16, 8, 4, 16, 32, 64, 32, 1]
        if kernel_sizes is None:
            kernel_sizes = [23, 17, 13, 7, 5, 3, 3, 5, 7, 11, 17]
        if strides is None:
            strides = [1, 2, 1, 1, 1, 2, 2, 2, 4, 2, 1]

        self.wavenumbers = np.array(wavenumbers)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg

        self.model = self._build_model(
            len(wavenumbers),
            filters, kernel_sizes, strides, l2_reg, l1_reg, pooling)

    def transform(self, spectra, wns, axis=-1, interpolate=False):
        """
        Corrects spectra with DSAE model. The spectra can be automatically
        interpolated to the required wavenumbers region. The array with spectra
        can be given in any shape.

        Parameters
        ----------
        spectra: `(..., N_wns, ...) array-like`
            A ND-array containing spectra
        wns: `(N_wns,) array-like, optional`
            A 1D-array of wavenumbers. If None then validation of spectral
            region and interpolation will not be performed.
        axis: `int, default -1`
            Axis along which lay spectra (spectral dimension)
        interpolate: `bool or str, optional`
            The value will be passed to `extrapolation` parameter of
            `biospectools.utils.interp2wns`. All values except False or None
            will perform interpolation. The difference is how boundary exceed
            is handled.

            Possible values:
            - False or None - will raise error in case of non-matching
              wavenumbers.
            - True - will use default extrapolation of interp1d.
              See `scipy.interpolate.interp1d`.
            - 'constant' will use 0 to fill values outside interpolation
              boundaries
            - 'bounds' will fill with bound values outside interp. region

        Returns
        -------
        corrected_spectra: `(..., N_wns, ...) array-like`
            A ND-array of input shape with corrected spectra
        """
        if not interpolate and np.shape(spectra)[axis] != len(self.wavenumbers):
            raise ValueError(
                f'Shape mismatch. Expected to have {len(self.wavenumbers)} '
                f'wavenumbers, given {np.shape(spectra)[axis]}')
        elif interpolate == 'intersect':
            raise ValueError(
                'intersect mode of interpolation is non-sense, '
                'since it may change required wavenumbers')

        spectra = np.array(spectra).swapaxes(axis, -1)
        spectra = self._interpolate_if_needed(spectra, wns, interpolate)

        reshaped = spectra.reshape(-1, len(self.wavenumbers), 1)
        corrected = self.model.predict(reshaped)
        return corrected.reshape(spectra.shape).swapaxes(axis, -1)

    def _interpolate_if_needed(self, spectra, wns, interpolate):
        if wns is not None and np.all(self.wavenumbers != wns):
            if not interpolate:
                raise ValueError(
                    'Wavenumbers mismatch. See wavenumbers parameter. '
                    'The values can be automatically interpolated '
                    'using argument `interpolate`')
            else:
                spectra, wns = interp2wns(
                    wns, self.wavenumbers, spectra, extrapolation=interpolate)
        return spectra

    @staticmethod
    def _build_model(
            n_wavenumbers, filters, kernel_sizes,
            strides, l2_reg, l1_reg, pooling):
        Pooling = AveragePooling1D if pooling == "average" else MaxPooling1D
        bn_idx = np.argmin(np.array(filters[:-1]))

        # Encoder
        input = Input(shape=(n_wavenumbers, 1), name="Input_Spectrum")
        encoder = input
        k, j = 1, 1
        for flter, krnl_size, strd in zip(
                filters[:bn_idx], kernel_sizes[:bn_idx], strides[:bn_idx]):
            encoder = Conv1D(filters=flter,
                             kernel_size=krnl_size,
                             strides=strd,
                             padding='same',
                             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                             bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                             name=f"Conv_{k}")(encoder)

            encoder = BatchNormalization(name=f"BatchNorm_{k}")(encoder)
            encoder = Activation("relu", name=f"ReLU_{k}")(encoder)
            encoder = Pooling(pool_size=2, name=f"Pooling_{k}")(encoder)
            k += 1

        # Decoder
        decoder = encoder
        for flter, krnl_size, strd in zip(
                filters[bn_idx:], kernel_sizes[bn_idx:], strides[bn_idx:]):
            decoder = _Conv1DTranspose(filters=flter,
                                       kernel_size=krnl_size,
                                       strides=strd,
                                       padding='same',
                                       kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                                       bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                                       name=f"TranspConv_{j}")(decoder)

            decoder = BatchNormalization(name=f"Batch_Norm_{k}")(decoder)
            decoder = Activation("relu", name=f"ReLU_{k}")(decoder)
            k += 1
            j += 1

        output_spectrum = Conv1D(filters=filters[-1],
                                 kernel_size=kernel_sizes[-1],
                                 strides=strides[-1],
                                 padding='same',
                                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                                 bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                                 name=f"Conv_{k - j + 1}")(decoder)

        return tf.keras.Model(inputs=input, outputs=output_spectrum)


class _Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides, padding,
                 kernel_regularizer, bias_regularizer, *args, **kwargs):
        super(_Conv1DTranspose, self).__init__(**kwargs)
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._padding = padding
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._args, self._kwargs = args, kwargs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self._filters,
            'kernel_size': self._kernel_size,
            'strides': self._strides,
            'padding': self._padding,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
        })
        return config

    def build(self, input_shape):
        self._model = tf.keras.Sequential(name=self.name)
        self._model.add(Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=1),
                               batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(filters=self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        padding=self._padding,
                                        kernel_regularizer=self._kernel_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:, 0]))

        super(_Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)


def _download_dsae_files():
    weights_url = ('https://github.com/eirikama/DSAE/raw/master'
                   '/weights/DSAE_weights.hdf5')
    weights_path = os.path.join(
        downloads.get_cache_directory(), 'dsae_article_weights.hdf5')
    downloads.download_http(weights_url, weights_path, overwrite=False)

    wns_url = 'https://github.com/eirikama/DSAE/raw/master/wns.npz'
    wns_path = os.path.join(
        downloads.get_cache_directory(), 'dsae_article_wns.npz')
    downloads.download_http(wns_url, wns_path, overwrite=False)

    return weights_path, wns_path
