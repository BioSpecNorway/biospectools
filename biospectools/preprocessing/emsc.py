from typing import Union as U, Tuple as T, Optional as O

import numpy as np


def emsc(
        spectra: np.ndarray,
        wavenumbers: np.ndarray,
        poly_order: O[int] = 2,
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
