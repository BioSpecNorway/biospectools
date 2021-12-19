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
