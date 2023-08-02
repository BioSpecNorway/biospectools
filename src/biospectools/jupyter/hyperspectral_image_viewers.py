from typing import Optional as Opt, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Layout


def interactive_himg(
        himg: np.ndarray,
        wns: Opt[np.ndarray] = None,
        init_wn: Opt[float] = None,
        figsize: Tuple[float, float] = (9, 3),
        cmap='turbo'):
    """Interactive exploration of a hyperspectral image.

    The figure shows selected wavenumber slice of a
    hyperspectral image and a corresponding spectrum
    under the pointer. Clicking on the image will fix
    the spectrum for comparison with others. Clicking
    on the spectrum will select a wavenumber for the
    slice of the hyperspectral image.

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
        i, j = int(event.ydata), int(event.xdata)
        sp_line.set_ydata(himg[i, j])
        spec_ax.relim()
        spec_ax.autoscale_view()
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', onmove)

    # Clicking on himg and spectrum
    def onclick(event):
        if event.inaxes == img_ax:
            i, j = int(event.ydata), int(event.xdata)
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
        img.set_data(himg[..., i])
        img.set_clim(himg[..., i].min(), himg[..., i].max())
        wn_line.set_data([wn, wn], [0, 1])
        fig.canvas.draw_idle()

    return interact(update, wn=wn_slider)
