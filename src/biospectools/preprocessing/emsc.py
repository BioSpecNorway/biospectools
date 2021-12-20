import logging
from typing import Union as U, Tuple as T, Optional

import numpy as np


class EMSCInternals:
    """Class that contains intermediate results of EMSC algorithm.

    Parameters
    ----------
    coefs : `(N_samples, 1 + N_constituents + (poly_order + 1) ndarray`
        All coefficients for each transformed sample. First column is a
        scaling parameter followed by constituent and polynomial coefs.
        This is a transposed solution of equation
        _model @ coefs.T = spectrum.
    scaling_coefs : `(N_samples,) ndarray`
        Scaling coefficients (reference to the first column of coefs_).
    polynomial_coefs : `(N_samples, poly_order + 1) ndarray`
        Coefficients for each polynomial order.
    constituents_coefs : `(N_samples, N_constituents) ndarray`
        Coefficients for each constituent.
    residuals : `(N_samples, K_channels) ndarray`
         Chemical residuals that were not fitted by EMSC model.

    Raises
    ------
    AttributeError
        When polynomial's or constituents' coeffs are not available.
    """

    def __init__(
            self,
            coefs: np.ndarray,
            residuals: np.ndarray,
            poly_order: Optional[int],
            constituents: Optional[np.ndarray]):
        assert len(coefs.T) == len(residuals), 'Inconsistent number of spectra'

        self.coefs = coefs.T
        self.residuals = residuals
        if constituents is not None:
            self._n_constituents = len(constituents)
        else:
            self._n_constituents = 0
        if poly_order is not None:
            self._n_polynomials = poly_order
        else:
            self._n_polynomials = 0

    @property
    def scaling_coefs(self) -> np.ndarray:
        return self.coefs[:, 0]

    @property
    def constituents_coefs(self) -> np.ndarray:
        if self._n_constituents == 0:
            raise AttributeError(
                'constituents were not set up. '
                'Did you forget to call transform?')
        return self.coefs[:, 1:1 + self._n_constituents]

    @property
    def polynomial_coefs(self) -> np.ndarray:
        if self._n_polynomials == 0:
            raise AttributeError(
                'poly_order was not set up. '
                'Did you forget to call transform?')
        return self.coefs[:, 1 + self._n_constituents:]


class EMSC:
    """Extended multiplicative signal correction (EMSC) [1]_.

    Parameters
    ----------
    reference : `(K_channels,) ndarray`
        Reference spectrum.
    wavenumbers : `(K_channels,) ndarray`, optional
        Wavenumbers must be passed if given polynomial order
        is greater than zero.
    poly_order : `int`, optional (default 2)
        Order of polynomial to be used in regression model. If None
        then polynomial will be not used.
    weights : `(K_channels,) ndarray`, optional
        Weights for spectra.
    constituents : `(N_constituents, K_channels) np.ndarray`, optional
        Chemical constituents for regression model [2]_. Can be used to add
        orthogonal vectors.
    scale : `bool`, default True
        If True then spectra will be scaled to reference spectrum.
    rebuild_model : `bool`, default True
         If True, then model will be built each time transform is called,
         this allows to dynamically change parameters of EMSC class.
         Otherwise model will be built once (for speed).

    Other Parameters
    ----------------
    _model : `(K_channels, 1 + N_constituents + (poly_order + 1) ndarray`
        Matrix that is used to solve least squares. First column is a
        reference spectrum followed by constituents and polynomial columns.
    _norm_wns : `(K_channels,) ndarray`
        Normalized wavenumbers to -1, 1 range

    References
    ----------
    .. [1] A. Kohler et al. *EMSC: Extended multiplicative
           signal correction as a tool for separation and
           characterization of physical and chemical
           information  in  fourier  transform  infrared
           microscopy  images  of  cryo-sections  of
           beef loin.* Applied spectroscopy, 59(6):707–716, 2005.
    """

    # TODO: Add numpy typing for array-like objects?
    def __init__(
            self,
            reference,
            wavenumbers=None,
            poly_order: Optional[int] = 2,
            constituents=None,
            weights=None,
            scale: bool = True,
            rebuild_model: bool = True,
    ):
        self.reference = np.asarray(reference)
        self.wavenumbers = wavenumbers
        if self.wavenumbers is not None:
            self.wavenumbers = np.asarray(wavenumbers)
        self.poly_order = poly_order
        self.weights = weights
        if self.weights is not None:
            self.weights = np.asarray(weights)
        self.constituents = constituents
        if self.constituents is not None:
            self.constituents = np.asarray(constituents)
        self.scale = scale
        self.rebuild_model = rebuild_model

        # lazy init during transform
        # allows to change dynamically EMSC's parameters
        self._model = None
        self._norm_wns = None

    def transform(
            self,
            spectra,
            internals: bool = False,
            check_correlation: bool = True) \
            -> U[np.ndarray, T[np.ndarray, EMSCInternals]]:
        spectra = np.asarray(spectra)
        self._validate_inputs()
        if check_correlation:
            self._check_high_correlation(spectra)

        if self.rebuild_model or self._model is None:
            self._norm_wns = self._normalize_wns()
            self._model = self._build_model()

        coefs = self._solve_lstsq(spectra)
        residuals = spectra - np.dot(self._model, coefs).T

        scaling = coefs[0]
        corr = self.reference + residuals / scaling[:, None]
        if not self.scale:
            corr *= scaling[:, None]

        if internals:
            internals_ = EMSCInternals(
                coefs, residuals, self.poly_order, self.constituents)
            return corr, internals_
        return corr

    def clear_state(self):
        del self._model
        del self._norm_wns

    def _build_model(self):
        columns = [self.reference]
        if self.constituents is not None:
            columns.extend(self.constituents)
        if self.poly_order is not None:
            columns.append(np.ones_like(self.reference))
            if self.poly_order > 0:
                n = self.poly_order + 1
                columns.extend(self._norm_wns ** pwr for pwr in range(1, n))
        return np.stack(columns, axis=1)

    def _solve_lstsq(self, spectra):
        if self.weights is None:
            return np.linalg.lstsq(self._model, spectra.T, rcond=None)[0]
        else:
            w = self.weights[:, None]
            return np.linalg.lstsq(self._model * w, spectra.T * w, rcond=None)[0]

    def _validate_inputs(self):
        if (self.poly_order is not None
                and self.poly_order < 0):
            raise ValueError(
                'poly_order must be equal or greater than 0')
        if (self.poly_order is not None
                and self.poly_order > 0
                and self.wavenumbers is None):
            raise ValueError(
                'wavenumbers must be specified when poly_order is given')
        if (self.wavenumbers is not None
                and len(self.wavenumbers) != len(self.reference)):
            raise ValueError(
                "Shape of wavenumbers doesn't match reference spectrum")

    def _check_high_correlation(self, spectra):
        mean = spectra.mean(axis=0)
        score = np.corrcoef(mean, self.reference)[0, 1]
        if score < 0.7:
            logging.warning(
                f'Low pearson score {score:.2f} between mean and reference '
                f'spectrum. Make sure that reference spectrum given in '
                f'the same order as spectra.')

    def _normalize_wns(self):
        if self.wavenumbers is None:
            return None
        half_rng = np.abs(self.wavenumbers[0] - self.wavenumbers[-1]) / 2
        return (self.wavenumbers - np.mean(self.wavenumbers)) / half_rng


def emsc(
        spectra: np.ndarray,
        wavenumbers: np.ndarray,
        poly_order: Optional[int] = 2,
        reference: np.ndarray = None,
        weights: np.ndarray = None,
        constituents: np.ndarray = None,
        return_coefs: bool = False,
        return_residuals: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray],
                                             T[np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess all spectra with EMSC algorithm [1]_.

    Parameters
    ----------
    spectra : `(N_samples, K_channels) np.ndarray`
        Spectra to be processed.
    wavenumbers : `(K_channels,) ndarray`
        Wavenumbers.
    poly_order : `int`, optional
        Order of polynomial to be used in regression model. If None
        then polynomial will be not used. (2, by default)
    reference : `(K_channels,) ndarray`, optional
        Reference spectrum. If None, then average will be computed.
    weights : `(K_channels,) ndarray`, optional
        Weights for spectra.
    constituents : `(N_constituents, K_channels) np.ndarray`, optional
        Chemical constituents for regression model [2]. Can be used to add
        orthogonal vectors.
    return_coefs : `bool`, optional
        Return coefficients.
    return_residuals : `bool`, optional
        Return residuals.

    Returns
    -------
    preprocessed_spectra : `(N_samples, K_channels) ndarray`
    coefficients : `(N_samples, 1 + N_constituents + (poly_order + 1) ndarray`, optional
        If ``return_coefs`` is true, then returns coefficients in the
        following order:
        #. Scaling parametes, b (related to reference spectrum)
        #. All constituents parameters in the same order as they given
        #. Polynomial coefficients (slope, quadratic effect and so on)
    residuals: `(N_samples, K_channels) ndarray`, optional
        If ``return_residuals`` is true, then returns residuals


    References
    ----------
    .. [1] A. Kohler et al. *EMSC: Extended multiplicative
           signal correction as a tool for separation and
           characterization of physical and chemical
           information  in  fourier  transform  infrared
           microscopy  images  of  cryo-sections  of
           beef loin.* Applied spectroscopy, 59(6):707–716, 2005.
    """
    assert poly_order is None or poly_order >= 0, 'poly_order must be >= 0'

    if reference is None:
        reference = np.mean(spectra, axis=0)

    # solve for coefs: X @ coefs = spectrum (column)
    # (1) build matrix X = [reference constituents polynomial]
    columns = [reference]
    if constituents is not None:
        columns.extend(constituents)
    if poly_order is not None:
        norm_wns = _normalize_wavenumbers(wavenumbers)
        columns.append(np.ones_like(norm_wns))
        for j in range(1, poly_order + 1):
            columns.append(norm_wns ** j)
    X = np.stack(columns, axis=1)

    # (2) Calculate coefs
    if weights is None:
        coefs = np.linalg.lstsq(X, spectra.T, rcond=None)[0]
    else:
        w = weights[:, None]
        coefs = np.linalg.lstsq(X * w, spectra.T * w, rcond=None)[0]

    # (3) Preprocessing
    residuals = spectra.T - np.dot(X, coefs)
    preprocessed_spectra = reference[:, None] + residuals / coefs[0]

    # (4) return results
    if return_residuals and return_coefs:
        return preprocessed_spectra.T, coefs.T, residuals.T
    elif return_coefs:
        return preprocessed_spectra.T, coefs.T
    elif return_residuals:
        return preprocessed_spectra.T, residuals.T

    return preprocessed_spectra.T


def _normalize_wavenumbers(wns: np.ndarray):
    half_rng = np.abs(wns[0] - wns[-1]) / 2
    return (wns - np.mean(wns)) / half_rng
