from dataclasses import dataclass
from itertools import chain
from typing import Optional as Opt, Tuple, List, Callable

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Layout


@dataclass
class HimgData:
    himg: np.ndarray
    wns: np.ndarray
    name: np.ndarray = None


def interactive_himg(
        himg: np.ndarray,
        wns: Opt[np.ndarray] = None,
        init_wn: Opt[float] = None,
        percentile: int = None,
        figsize: Tuple[float, float] = (9, 3),
        cmap='turbo'):
    """Interactive exploration of a hyperspectral image.

    The figure shows selected wavenumber slice of a
    hyperspectral image and a corresponding spectrum
    under the pointer.

    Controls
    --------

    - To fix a spectrum for comparison, click on the image
    - To select a wavenumber, click on the spectrum

    Notes
    -----
    Jupyter notebook should be turned into an interactive
    matplotlib regime. Two most popular backgrounds for it
    are `notebook` and `widget`.

    >>> # Run one of the lines below in jupyter notebook
    >>> %matplotlib notebook  # requires ipywidgets (for notebook)
    >>> %matplotlib widget    # requires ipympl (for jupyterlab)

    >>> # Comeback to the default inline regime
    >>> %matplotlib inline

    Parameters
    ----------
    himg: `(H, W, K_channels) np.ndarray`
        Hyperspectral cube for the visualization
    wns: `(K_channels,) np.ndarray`, optional
        Corresponding wavenumbers for the image. By default,
        will create a range of indices
    init_wn: `float`, optional
        wavenumber that will be first selected
    figsize: `(float, float)`, optional
        figsize parameter for the `plt.figure` func.
        Default (9, 3)
    percentile: float
       A value between 0 and 100 to suppress outliers. Will
       set color limits according to perc and 100-perc values.
       Default None.
    cmap: `str`, optional
        cmap parameter for the `plt.imshow` func.
        Default 'turbo'

    Returns
    -------
    Ipywidgets interactive layout
    """
    if wns is None:
        wns = np.arange(himg.shape[-1])
    if init_wn is None:
        init_wn = wns.mean()
    wn_slider = FloatSlider(init_wn, min=wns.min(), max=wns.max(), step=1,
                            layout=Layout(width='95%'))

    fig = plt.figure(figsize=figsize)
    gridspec = plt.GridSpec(1, 3)

    # Plot image
    img_ax = fig.add_subplot(gridspec[0, 0])
    img = img_ax.imshow(himg.mean(axis=-1), cmap=cmap)

    # Plot spectrum
    spec_ax = fig.add_subplot(gridspec[0, 1:])
    sp_line, = spec_ax.plot(wns, himg[0, 0], ls='--', lw=0.5, color='k')
    wn_line = spec_ax.axvline(init_wn, ls='--', lw=0.5, color='r')
    spec_ax.invert_xaxis()

    fig.tight_layout()

    # Moving mouse over himg
    def onmove(event):
        i, j = round(event.ydata), round(event.xdata)
        sp_line.set_ydata(himg[i, j])
        spec_ax.relim()
        spec_ax.autoscale_view()
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', onmove)

    # Clicking on himg and spectrum
    def onclick(event):
        if event.inaxes == img_ax:
            i, j = round(event.ydata), round(event.xdata)
            spec_ax.plot(wns, himg[i, j], label=f'{i}, {j}')
            spec_ax.legend()
            img_ax.scatter(j, i)
        elif event.inaxes == spec_ax:
            wn_slider.value = event.xdata
        else:
            return
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', onclick)

    def update(wn):
        i = np.argmin(np.abs(wns - wn))
        slice_ = himg[..., i]
        clim = _get_color_boundaries(
            himg_slices=[slice_], mode='relative', percentile=percentile)
        img.set_data(slice_)
        img.set_clim(*clim)
        wn_line.set_data([wn, wn], [0, 1])
        fig.canvas.draw_idle()

    return interact(update, wn=wn_slider)


def compare_himgs(
        *himgs: tuple,
        init_wn: Opt[float] = None,
        figsize: Tuple[float, float] = (9, 6),
        percentile: int = None,
        mode: str ='relative',
        normalizer: Callable = None,
        cmap='turbo'):
    """Interactive comparison of several hyperspectral images.

    The figure can display from 2 to 5 hyperspectral images with
    spectra taken from the same position of the corresponding
    image. Useful to compare preprocessing methods.

    Controls
    --------
    To fix/unfix spectra, click on the image

    Examples
    --------
    >>> compare_himgs(
    >>>     (himg1, wns1, 'Image#1'),
    >>>     (himg2, wns2, 'Another image'),
    >>>     (himg3, wns3),  # Image without a name
    >>>     init_wn=1745, percentile=3, mode='relative',
    >>>     normalizer=PeakScaler(wn=1654))

    Notes
    -----
    Jupyter notebook should be turned into an interactive
    matplotlib regime. Two most popular backgrounds for it
    are `notebook` and `widget`.

    >>> # Run one of the lines below in jupyter notebook
    >>> %matplotlib notebook  # requires ipywidgets (for notebook)
    >>> %matplotlib widget    # requires ipympl (for jupyterlab)

    >>> # Comeback to the default inline regime
    >>> %matplotlib inline

    Parameters
    ----------
    *himgs: `tuple (himg, wns, Optional[name]) or HimgData`
        List of hyperspectral images to be compared
    init_wn: `float`, optional
        wavenumber that will be first selected
    figsize: `(float, float)`, optional
        figsize parameter for the `plt.figure` func.
        Default (9, 6)
    percentile: float
       A value between 0 and 100 to suppress outliers. Will
       set color limits according to perc and 100-perc values.
       Default None.
    mode: str 'absolute' or 'relative'
        Decides whether to set color limits using all the images
        together (absolute) or individual ones (relative).
        Default 'relative'.
    normalizer: Callable
        A function to normalize spectra before plotting. The
        input is a list of tuples (spectrum, wns) and output
        should be list of normalized spectra. PeakNormalizer
        is ready to be used.
        Default None
    cmap: `str`, optional
        cmap parameter for the `plt.imshow` func.
        Default 'turbo'

    Returns
    -------
    Ipywidgets interactive layout
    """
    if not (1 < len(himgs) <= 5):
        raise ValueError('Number of himgs must be within 2-5')
    himgs = _args_to_himgdata_struct(himgs)

    if init_wn is None:
        init_wn = himgs[0].wns.mean()
    wn_slider = FloatSlider(
        init_wn,
        min=min(h.wns.min() for h in himgs),
        max=max(h.wns.max() for h in himgs),
        step=1,
        layout=Layout(width='95%'))
    is_cursor_fixed = [False]  # workaround for global variable

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, len(himgs))

    # Prepare image axes
    img_axes = []
    for i, h in enumerate(himgs):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        ax: plt.Axes = fig.add_subplot(gs[0, i])
        ax.axis('off')
        ax.set_title(h.name, loc='left')
        _add_line_next_to_title(ax, color, length=0.1, thickness=4)
        if i > 0:
            ax.sharex(img_axes[0])
            ax.sharey(img_axes[0])
        img_axes.append(ax)

    # Plot images and cursors
    pimgs = []
    cursors = []
    for ax, h in zip(img_axes, himgs):
        pimg = ax.imshow(h.himg.mean(axis=-1), cmap=cmap)
        cursor = plt.Circle((0, 0), radius=0.5, color='black', visible=False)
        ax.add_patch(cursor)
        pimgs.append(pimg)
        cursors.append(cursor)

    # Prepare spectra axes
    spec_ax = fig.add_subplot(gs[1, :])
    wn_line = spec_ax.axvline(init_wn, ls='--', lw=0.5, color='r')
    spec_ax.invert_xaxis()
    lines = []
    for h in himgs:
        line, = plt.plot(h.wns, h.himg[0, 0], lw=0.5, ls='--')
        lines.append(line)

    def onmove(event):
        if event.inaxes not in img_axes or is_cursor_fixed[0]:
            return
        # Update cursors
        for cursor in cursors:
            cursor.set_center((event.xdata, event.ydata))

        # Update lines
        i, j = round(event.ydata), round(event.xdata)
        spectra = [h.himg[i, j] for h in himgs]
        if normalizer is not None:
            spec_wns = zip(spectra, [h.wns for h in himgs])
            spectra = normalizer(spec_wns)
        for spec, line in zip(spectra, lines):
            line.set_ydata(spec)

        # Update limits
        spec_ax.relim()
        spec_ax.autoscale_view()
        fig.canvas.draw_idle()

    def onclick(event):
        # Select wavenumber
        if event.inaxes == spec_ax:
            wn_slider.value = event.xdata
        # Fix spectra
        elif event.inaxes in img_axes:
            is_cursor_fixed[0] = not is_cursor_fixed[0]
            idx = img_axes.index(event.inaxes)
            cursors[idx].set_visible(is_cursor_fixed[0])
            for l in lines:
                l.set_linestyle('-' if is_cursor_fixed[0] else '--')

    def toggle_cursors(event):
        try:
            idx = img_axes.index(event.inaxes)
            for ci, c in enumerate(cursors):
                other = ci != idx  # turn on all cursors except active
                c.set_visible(other or is_cursor_fixed[0])
        except ValueError:
            pass

    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('axes_enter_event', toggle_cursors)

    def update(wn):
        slices = [h.himg[..., _closest_wn_idx(h.wns, wn)]
                  for h in himgs]
        color_lims = _get_color_boundaries(slices, mode, percentile)
        for pimg, himg_slice, clim in zip(pimgs, slices, color_lims):
            pimg.set_data(himg_slice)
            pimg.set_clim(*clim)
        wn_line.set_data([wn, wn], [0, 1])
        fig.canvas.draw_idle()

    plt.tight_layout()

    return interact(update, wn=wn_slider)


def _args_to_himgdata_struct(himgs):
    himgdatas = []
    for i, h in enumerate(himgs):
        if isinstance(h, (tuple, list)) and 1 <= len(h) <= 3:
            hd = HimgData(*h)
        elif isinstance(h, HimgData):
            hd = h
        else:
            raise ValueError(
                f'Unkown input type for hyperspectral image {type(h)}')

        if hd.name is None:
            hd.name = f'Img#{i}'
        himgdatas.append(hd)
    return himgdatas


class PeakNormalizer:
    def __init__(self, wn, subtract_min=True, soft_min_percentile=5,
                 wn_half_window=5):
        self.wn = wn
        self.subtract_min = subtract_min
        self.soft_min_percentile = soft_min_percentile
        self.wn_half_window = wn_half_window

    def __call__(self, spec_wns):
        spectra = []
        for spec, wns in spec_wns:
            if self.subtract_min:
                if self.soft_min_percentile is None:
                    offset = spec.min()
                else:
                    offset = np.percentile(spec, self.soft_min_percentile)
            else:
                offset = 0
            i1 = _closest_wn_idx(wns, self.wn - self.wn_half_window)
            i2 = _closest_wn_idx(wns, self.wn + self.wn_half_window)
            i1, i2 = min(i1, i2), max(i1, i2)
            spec = spec - offset
            spec /= spec[..., i1:i2].max()
            spectra.append(spec)
        return spectra


def _add_line_next_to_title(ax, color, length=0.1, thickness=1):
    # Use annotate to add the line
    ax.annotate('', xy=(1 - length, 1.05), xytext=(1, 1.05),
                xycoords='axes fraction', annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=color, lw=thickness))


def _closest_wn_idx(wns, wn):
    return np.argmin(np.abs(wns - wn))


def _get_color_boundaries(himg_slices, mode, percentile):
    if percentile is None:
        percentile = 0
    if mode == 'absolute':
        vals = list(chain.from_iterable([h.flat for h in himg_slices]))
        cmin, cmax = np.percentile(vals, [percentile, 100 - percentile])
        return [(cmin, cmax)] * len(himg_slices)
    elif mode == 'relative':
        bounds = []
        for h in himg_slices:
            cmin, cmax = np.percentile(h, [percentile, 100 - percentile])
            bounds.append((cmin, cmax))
        return bounds
    else:
        ValueError(f'Unknown himg comparison mode {mode}')

