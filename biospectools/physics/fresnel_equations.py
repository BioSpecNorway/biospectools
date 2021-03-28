import numpy as np

def reflection_amp(n, wavelength, l):
    k = 2*np.pi / wavelength
    n0 = 1
    nt = 1
    e = np.exp(1j*n*k*l)
    num = 1 / ((n0+n)*(n+nt) - e**2*(n-nt)*(n-n0))
    A = 2*n0*(n+nt) * num
    B = 2*n0*(n-nt)*e**2 * num
    return (1/(2*n0))*(A*(n0-n)+B*(n0+n))


def transmission_amp(n, wavelength, l):
    k = 2*np.pi / wavelength
    n0 = 1
    nt = 1
    e = np.exp(1j*n*k*l)
    num = 1 / ((n0+n)*(n+nt) - e**2*(n-nt)*(n-n0))
    A = 2*n0*(n+nt) * num
    B = 2*n0*(n-nt)*e**2 * num
    return np.exp(-1j*nt*k*l)*(A*e + B/e)


if __name__ == '__main__':
    from biospectools.physics.peak_shapes import *
    from biospectools.physics.misc import *
    import matplotlib.pyplot as plt

    wn = np.linspace(1000, 2500, 1500)*100
    wl = 1/wn

    lorentz_peak = lorentzian(wn, 1600*100, 0.05, 500)
    nkk = get_nkk(lorentz_peak, wl)
    n0 = 1.3

    n = n0 + nkk + 1j*lorentz_peak

    l = 10e-6

    t = transmission_amp(n, wl, l)
    T = np.abs(t)**2

    plt.figure()
    plt.plot(wn/100, T, label='Transmission')
    plt.plot(wn/100, -np.log10(T), label='Absorbance')
    plt.legend()

    plt.show()
