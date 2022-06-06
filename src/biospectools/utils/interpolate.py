import numpy as np
from scipy.interpolate import interp1d


def interp2wns(
        old_wns, new_wns, spectra,
        kind='linear', axis=-1, extrapolation=None, fill_value=0):
    """
    Adds functionality to `scipy.interpolate.interp1d`.

    Parameters
    ----------
    old_wns: `(N,) array-like`
        A 1-D array of old wavenumbers
    new_wns: `(N,) array-like`
        A 1-D array of new wavenumbers
    spectra: `(..., N,...) array-like`
        A N-D array of spectra values. The length of y along
        the interpolation axis must be equal to the length of x.
    kind: `str or int, optional`
        Most common values ‘linear’, ‘nearest’, ‘quadratic’, ‘cubic’, 'next',
        'previous'. If integer then specifying the order of spline.
        Default is "linear". For more details see `scipy.interpolate.interp1d`.
    axis: `int, optional`
        axis along which interpolation will be done
    extrapolation: `bool or str, optional`
        - False or None - will raise error in case of exceeding bounds
        - True - uses default extrapolation of interp1d.
                 See `scipy.interpolate.interp1d`.
        - 'constant' will use fill_value to fill values outside interpolation
          boundaries
        - 'bounds' will fill with bound values outside interp. region
        - 'intersect' will return new wavenumbers that are within old ones
    fill_value: `array-like or (array-like, array_like), optional`
        Value to use with 'constant' extrapolation mode.
        For more details see `scipy.interpolate.interp1d`.

    Returns
    -------
    interpolated_spectra: `ndarray`
    new_wns: `ndarray`
        The same new_wns as were given as input, except
        'intersect' extrapolation mode
    """
    if not extrapolation:
        f = interp1d(old_wns, spectra, kind, axis, bounds_error=True)
    elif extrapolation is True:
        f = interp1d(old_wns, spectra, kind, axis, fill_value='extrapolate')
    elif extrapolation == 'constant':
        f = interp1d(
            old_wns, spectra, kind, axis,
            bounds_error=False, fill_value=fill_value)
    elif extrapolation == 'bounds':
        min_idx, max_idx = np.argmin(old_wns), np.argmax(old_wns)
        lower_bound = np.take(spectra, min_idx, axis)
        upper_bound = np.take(spectra, max_idx, axis)
        f = interp1d(
            old_wns, spectra, kind, axis,
            bounds_error=False, fill_value=(lower_bound, upper_bound))
    elif extrapolation == 'intersect':
        interpolated = interp1d(
            old_wns, spectra, kind, axis,
            bounds_error=False, fill_value=np.nan)(new_wns)

        spatial_axes = list(range(np.ndim(spectra)))
        del spatial_axes[axis]
        wns_mask = ~np.any(np.isnan(interpolated), axis=tuple(spatial_axes))
        idxs = np.where(wns_mask)[0]

        return np.take(interpolated, idxs, axis), new_wns[wns_mask]
    else:
        raise ValueError(
            f'Unknown extrapolation parameter: {extrapolation}. '
            f"Must be one of [None, False, 'constant', 'bounds', 'intersect']")

    return f(new_wns), new_wns
