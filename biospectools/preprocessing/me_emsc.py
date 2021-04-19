import numpy as np
from sklearn.decomposition import TruncatedSVD
from biospectools.physics.misc import calculate_complex_n


def fill_edges_1d(l):
    """Replace (inplace!) NaN at sides with the closest value"""
    loc = np.where(~np.isnan(l))[0]
    if len(loc):
        fi, li = loc[[0, -1]]
        l[:fi] = l[fi]
        l[li + 1 :] = l[li]


def fill_edges(mat):
    """Replace (inplace!) NaN at sides with the closest value"""
    for l in mat:
        fill_edges_1d(l)


def nan_extend_edges_and_interpolate(xs, X):
    """
    Handle NaNs at the edges are handled as with savgol_filter mode nearest:
    the edge values are interpolated. NaNs in the middle are interpolated
    so that they do not propagate.
    """
    nans = None
    if np.any(np.isnan(X)):
        nans = np.isnan(X)
        X = X.copy()
        xs, xsind, mon, X = transform_to_sorted_wavenumbers(xs, X)
        fill_edges(X)
        X = interp1d_with_unknowns_numpy(xs[xsind], X, xs[xsind])
        X = transform_back_to_features(xsind, mon, X)
    return X, nans


def spectra_mean(X):
    return np.nanmean(X, axis=0, dtype=np.float64)


def interpolate_to_data(other_xs, other_data, wavenumbers):
    # all input data needs to be interpolated (and NaNs removed)
    interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
    # we know that X is not NaN. same handling of reference as of X
    interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
    return interpolated


def interp1d_with_unknowns_numpy(x, ys, points, kind="linear"):
    if kind != "linear":
        raise NotImplementedError
    out = np.zeros((len(ys), len(points))) * np.nan
    sorti = np.argsort(x)
    x = x[sorti]
    for i, y in enumerate(ys):
        y = y[sorti]
        nan = np.isnan(y)
        xt = x[~nan]
        yt = y[~nan]
        # do not interpolate unknowns at the edges
        if len(xt):  # check if all values are removed
            bhg = np.interp(points.squeeze(), xt, yt, left=np.nan, right=np.nan)
            out[i] = bhg.squeeze()
    return out


def transform_to_sorted_wavenumbers(xs, X):
    xsind = np.argsort(xs)
    mon = is_increasing(xsind)
    X = X if mon else X[:, xsind]
    return xs, xsind, mon, X


def is_increasing(a):
    return np.all(np.diff(a) >= 0)


def transform_back_to_features(xsind, mon, X):
    return X if mon else X[:, np.argsort(xsind)]


def calculate_qext_curves(nkks, nprs, alpha0, gamma, wavenumbers):
    gamma_nprs = (1 + np.multiply.outer(gamma, nprs)) * (wavenumbers * 100)
    tanbeta = nkks / np.add.outer((1 / gamma.T), nprs)

    beta0 = np.arctan(tanbeta)
    cosB = np.cos(beta0)
    cos2B = np.cos(2.0 * beta0)

    n_alpha = len(alpha0)
    n_gamma = len(gamma)

    q_matrix = np.zeros((n_alpha * n_gamma, len(wavenumbers)))

    for i in range(n_alpha):
        rho = alpha0[i] * gamma_nprs
        rhocosB = cosB / rho
        q = 2.0 + (4 * rhocosB) * (
            -np.exp(-(rho) * (tanbeta))
            * (np.sin((rho) - (beta0)) + np.cos((rho - 2 * beta0)) * rhocosB)
            + cos2B * rhocosB
        )
        q_matrix[i * n_alpha : (i + 1) * n_alpha, :] = q
    return q_matrix


def orthogonalize_qext(qext, reference):
    m = np.dot(reference, reference)
    norm = np.sqrt(m)
    rnorm = reference / norm
    s = np.dot(qext, rnorm)
    qext_orthogonalized = qext - s[:, np.newaxis] * rnorm[np.newaxis, :]
    return qext_orthogonalized


def compress_mie_curves(qext_orthogonalized, num_comp):
    svd = TruncatedSVD(
        n_components=num_comp, n_iter=7, random_state=42
    )  # Self.ncomp needs to be specified
    svd.fit(qext_orthogonalized)
    badspectra = svd.components_[0:num_comp, :]
    return badspectra


def cal_ncomp(reference, wavenumbers, explained_var_lim, alpha0, gamma):
    nprs, nkks = calculate_complex_n(reference, wavenumbers)
    qext = calculate_qext_curves(nprs, nkks, alpha0, gamma, wavenumbers)
    qext_orthogonalized = orthogonalize_qext(qext, reference)
    max_ncomp = reference.shape[0] - 1
    svd = TruncatedSVD(n_components=min(max_ncomp, 30), n_iter=7, random_state=42)
    svd.fit(qext_orthogonalized)
    lda = np.array(
        [
            (sing_val ** 2) / (qext_orthogonalized.shape[0] - 1)
            for sing_val in svd.singular_values_
        ]
    )

    explained_var = 100 * lda / np.sum(lda)
    explained_var = np.cumsum(explained_var)
    num_comp = np.argmax(explained_var > explained_var_lim) + 1
    return num_comp


class ME_EMSC:
    def __init__(
        self,
        reference: np.ndarray = None,
        wn_reference: np.ndarray = None,
        weights: np.ndarray = None,
        ncomp: int = 0,
        n0: np.ndarray = np.linspace(1.1, 1.4, 10),
        a: np.ndarray = np.linspace(2, 7.1, 10),
        h: float = 0.25,
        max_iter: int = 30,
        tol: int = 4,
        verbose: bool = False,
        positive_ref: bool = True,
    ):

        if reference is None:
            raise ValueError("reference spectrum must be defined")

        if (wn_reference[1] - wn_reference[0]) < 0:
            raise ValueError("wn_reference must be ascending")

        self.reference = reference
        self.tol = tol
        self.wn_reference = wn_reference
        self.positive_ref = positive_ref
        self.weights = weights
        self.ncomp = ncomp
        self.verbose = verbose
        self.max_iter = max_iter

        self.n0 = n0
        self.a = a
        self.h = h

        self.alpha0 = (4 * np.pi * self.a * (self.n0 - 1)) * 1e-6
        self.gamma = (
            self.h
            * np.log(10)
            / (4 * np.pi * 0.5 * np.pi * (self.n0 - 1) * self.a * 1e-6)
        )

        explained_variance = 99.96
        if self.ncomp == 0:
            ref_x = np.atleast_2d(spectra_mean(np.expand_dims(self.reference, axis=0)))
            wavenumbers_ref = np.array(sorted(self.wn_reference))
            ref_x = interpolate_to_data(self.wn_reference.T, ref_x, wavenumbers_ref.T)
            ref_x = ref_x[0]
            self.ncomp = cal_ncomp(
                ref_x, wavenumbers_ref, explained_variance, self.alpha0, self.gamma
            )
        else:
            self.explained_variance = False

    def transform(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # compute average spectrum from the reference

        if (wavenumbers[1] - wavenumbers[0]) < 0:
            raise ValueError("wavenumbers must be ascending")

        def make_basic_emsc_mod(ref):
            N = wavenumbers.shape[0]
            m0 = -2.0 / (wavenumbers[0] - wavenumbers[N - 1])
            c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])
            m_basic = []
            for x in range(0, 3):
                m_basic.append((m0 * (wavenumbers - c_coeff)) ** x)
            m_basic.append(ref)  # always add reference spectrum to the model
            m_basic = np.vstack(m_basic).T
            return m_basic

        def cal_emsc_basic(m_basic, spectrum):
            m = np.linalg.lstsq(m_basic, spectrum, rcond=-1)[0]
            corrected = spectrum
            for x in range(0, 3):
                corrected = corrected - (m[x] * m_basic[:, x])
            corrected = corrected / m[3]
            scaled_spectrum = corrected
            return scaled_spectrum

        def make_emsc_model(badspectra, reference_spec):
            M = np.ones([len(wavenumbers), self.ncomp + 2])
            M[:, 1 : self.ncomp + 1] = np.array([spectrum for spectrum in badspectra.T])
            M[:, self.ncomp + 1] = reference_spec
            return M

        def cal_emsc(M, X):
            correctedspectra = np.zeros((X.shape[0], X.shape[1] + M.shape[1]))
            for i, rawspectrum in enumerate(X):
                m = np.linalg.lstsq(M, rawspectrum, rcond=-1)[0]
                corrected = rawspectrum
                for x in range(0, 1 + self.ncomp):
                    corrected = corrected - (m[x] * M[:, x])
                corrected = corrected / m[1 + self.ncomp]
                corrected[np.isinf(corrected)] = np.nan
                corrected = np.hstack((corrected, m))
                correctedspectra[i] = corrected

            params = correctedspectra[:, -(self.ncomp + 2) :]
            res = X - np.dot(params, M.T)
            return correctedspectra, res

        def iteration_step(spectrum, reference, wavenumbers, m_basic, alpha0, gamma):
            # scale with basic EMSC:
            reference = cal_emsc_basic(m_basic, reference)
            if np.all(np.isnan(reference)):
                raise np.linalg.LinAlgError()

            # Apply weights
            reference = reference * wei_x
            reference = reference[0]

            # set negative parts to zero
            nonzero_reference = reference.copy()
            nonzero_reference[nonzero_reference < 0] = 0

            if self.positive_ref:
                reference = nonzero_reference

            # calculate Qext-curves
            nprs, nkks = calculate_complex_n(nonzero_reference, wavenumbers)
            qext = calculate_qext_curves(nprs, nkks, alpha0, gamma, wavenumbers)
            qext = orthogonalize_qext(qext, reference)

            badspectra = compress_mie_curves(qext, self.ncomp)

            # build ME-EMSC model
            M = make_emsc_model(badspectra, reference)

            # calculate parameters and corrected spectra
            new_spectrum, res = cal_emsc(M, spectrum)

            return new_spectrum, res

        def iterate(
            spectra,
            corrected_first_iter,
            residual_first_iter,
            wavenumbers,
            m_basic,
            alpha0,
            gamma,
        ):
            new_spectra = np.full(corrected_first_iter.shape, np.nan)
            number_of_iterations = np.full(spectra.shape[0], np.nan)
            residuals = np.full(spectra.shape, np.nan)
            rmse_all = np.full([spectra.shape[0]], np.nan)
            N = corrected_first_iter.shape[0]
            for i in range(N):
                if self.verbose:
                    print(
                        "Corrected spectra ["
                        + int((i / N) * 20) * "#"
                        + int(((N - i - 1) / N) * 20) * " "
                        + f"] [{i}/{N}]",
                        end="\r",
                    )
                corr_spec = corrected_first_iter[i]
                prev_spec = corr_spec
                raw_spec = spectra[i, :]
                raw_spec = raw_spec.reshape(1, -1)
                rmse_list = [
                    round(
                        np.sqrt(
                            (1 / len(residual_first_iter[i, :]))
                            * np.sum(residual_first_iter[i, :] ** 2)
                        ),
                        self.tol,
                    )
                ]
                for iter_number in range(2, self.max_iter + 1):
                    try:
                        new_spec, res = iteration_step(
                            raw_spec,
                            corr_spec[: -self.ncomp - 2],
                            wavenumbers,
                            m_basic,
                            alpha0,
                            gamma,
                        )
                    except np.linalg.LinAlgError:
                        new_spectra[i, :] = np.full(
                            [raw_spec.shape[1] + self.ncomp], np.nan
                        )
                        residuals[i, :] = np.full(raw_spec.shape, np.nan)
                        rmse_all[i] = np.nan
                        break
                    corr_spec = new_spec[0, :]
                    rmse = round(
                        np.sqrt((1 / len(res[0, :])) * np.sum(res ** 2)), self.tol
                    )
                    rmse_list.append(rmse)

                    # Stop criterion
                    if iter_number == self.max_iter:
                        new_spectra[i, :] = corr_spec
                        number_of_iterations[i] = iter_number
                        residuals[i, :] = res
                        rmse_all[i] = rmse_list[-1]
                        break
                    elif iter_number > 2:
                        if rmse == rmse_list[-2] and rmse == rmse_list[-3]:
                            new_spectra[i, :] = corr_spec
                            number_of_iterations[i] = iter_number
                            residuals[i, :] = res
                            rmse_all[i] = rmse_list[-1]
                            break
                        if rmse > rmse_list[-2]:
                            new_spectra[i, :] = prev_spec
                            number_of_iterations[i] = iter_number - 1
                            rmse_all[i] = rmse_list[-2]
                            break

            if self.verbose:
                print(f"\n ----- Finished correcting {N} spectra ----- \n")
            return new_spectra, residuals, rmse_all, number_of_iterations

        ref_x = np.atleast_2d(spectra_mean(np.expand_dims(self.reference, axis=0)))
        ref_x = interpolate_to_data(self.wn_reference.T, ref_x, wavenumbers.T)
        ref_x = ref_x[0]

        if self.weights:
            wei_x = self.weights
        else:
            wei_x = np.ones((1, len(wavenumbers)))

        ref_x = ref_x * wei_x
        ref_x = ref_x[0]
        if self.positive_ref:
            ref_x[ref_x < 0] = 0

        resonant = True  # Possibility for using the 2008 version

        if resonant:
            # if this should be any point, we need to terminate after
            # 1 iteration for the non-resonant one
            nprs, nkks = calculate_complex_n(ref_x, wavenumbers)
        else:
            npr = np.zeros(len(wavenumbers))
            nprs = npr / (wavenumbers * 100)
            nkks = np.zeros(len(wavenumbers))

        # For the first iteration, make basic EMSC model
        m_basic = make_basic_emsc_mod(ref_x)
        # Calculate scattering curves for ME-EMSC
        qext = calculate_qext_curves(nprs, nkks, self.alpha0, self.gamma, wavenumbers)
        qext = orthogonalize_qext(qext, ref_x)
        badspectra = compress_mie_curves(qext, self.ncomp)
        # Establish ME-EMSC model
        M = make_emsc_model(badspectra, ref_x)
        # Correcting all spectra at once for the first iteration
        new_spectra, res = cal_emsc(M, X)
        if self.max_iter == 1:
            res = np.array(res)
            number_of_iterations = np.ones([1, new_spectra.shape[0]])
            rmse_all = [
                round(
                    np.sqrt((1 / res.shape[1]) * np.sum(res[specNum, :] ** 2)),
                    self.tol,
                )
                for specNum in range(new_spectra.shape[0])
            ]
            return new_spectra, res, rmse_all, number_of_iterations

        # Iterate
        new_spectra, residuals, rmse_all, number_of_iterations = iterate(
            X, new_spectra, res, wavenumbers, m_basic, self.alpha0, self.gamma
        )
        return new_spectra, residuals, rmse_all, number_of_iterations
