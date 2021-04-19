from biospectools.preprocessing.me_emsc import ME_EMSC
from pkg_resources import resource_filename as rs_path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import time

import biospectools.data


def adjustWavenumbers(RefSpec, wnRefSpec, RawSpec, wnRawSpec):
    minWavenumber = max(min(wnRefSpec), min(wnRawSpec))
    maxWavenumber = min(max(wnRefSpec), max(wnRawSpec))
    i1 = np.argmin(abs(wnRefSpec - minWavenumber))
    i2 = np.argmin(abs(wnRefSpec - maxWavenumber))
    RefSpec = RefSpec[:, i1:i2]
    wnRefSpec = wnRefSpec[i1:i2]
    j1 = np.argmin(abs(wnRawSpec - minWavenumber))
    j2 = np.argmin(abs(wnRawSpec - maxWavenumber))
    RawSpec = RawSpec[:, j1:j2]
    wnRawSpec = wnRawSpec[j1:j2]

    RefSpecFitted = interp1d(wnRefSpec, RefSpec)(wnRawSpec[1:])

    RawSpecFitted = RawSpec[:, 1:]
    wn = wnRawSpec[1:]

    return RefSpecFitted, RawSpecFitted, wn


# matplotlib.use("TkAgg")
# plt.style.use("ggplot")

raw_data = loadmat("data/memsc_test_data/measuredSpectra.mat")

wn_raw = raw_data["Spectra"][0][0][1].astype("float64")
raw = raw_data["Spectra"][0][0][0]
wn_ref, ref = biospectools.data.load_matrigel_spectrum()
ref = ref[None]

print(
    f"Shape of ndarray with Referance Spectrum: {ref.shape}\n"
    f"Shape of ndarray with Reference Spectrum's Wavenumbers: {wn_ref.shape}\n"
    f"Shape of ndarray with Raw Data: {raw.shape}\n"
    f"Shape of ndarray with Raw Data's Wavenumbers: {wn_raw.shape}"
)

max_iter = 15
ref, raw, wn = adjustWavenumbers(ref, wn_ref, raw, wn_raw)

print(f"Adjust wavenumbers of Raw and Referance Spectra [wn.shape={wn.shape}]\n")

ref = ref / np.max(ref)

model = ME_EMSC(
    reference=ref,
    wn_reference=wn,
    ncomp=False,
    n0=np.linspace(1.1, 1.4, 10),
    a=np.linspace(2, 7.1, 10),
    max_iter=max_iter,
    tol=4,
    verbose=True,
)

t = time.time()

correction, residuals, RMSE, iterations = model.transform(raw, wn)

print(f"Correction lasted {time.time() - t:.2f} seconds")

corrected_spectra = correction[:, : raw.shape[1]]
emsc_parameters = correction[:, raw.shape[1] :]

"""
for spectrum in corrected_spectra:
    plt.plot(wn, spectrum)
plt.title("Corrected Spectra")
plt.xlabel(r"$\tilde{\nu}_n$", fontsize=24)
plt.ylabel("A", rotation=0, fontsize=20, labelpad=15)
plt.gca().invert_xaxis()
plt.gca().tick_params(axis="both", which="major", labelsize=20)
plt.show()

plt.hist(iterations, bins=max_iter)
plt.title("Histogram of Iterations Until Convergence")
plt.xlabel(r"Number of Iterations", fontsize=24)
plt.ylabel("Number of Spectra", fontsize=20)
plt.show()

plt.hist(RMSE, bins=30)
plt.title("Histogram of RMSE")
plt.xlabel(r"RMSE", fontsize=24)
plt.ylabel("Number of Spectra", fontsize=20)
plt.show()
"""
