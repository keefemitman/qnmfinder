import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*ftol*")
warnings.filterwarnings("ignore", message=".*gtol*")

import copy
import joblib
import numpy as np

import qnm
from scri.sample_waveforms import modes_constructor

from . import utils
from . import varpro

import multiprocessing
from functools import partial

from termcolor import colored

_ksc = qnm.modes_cache


def omega_and_C(mode, target_mode, M_f, chi_f):
    """Compute a QNM's frequency and spherical-spheroidal mixing coefficients."""
    if mode == (0,0,0,0):
        ells = np.arange(max(2, abs(target_mode[1])), 20 + 1, 1)
        C = np.zeros_like(ells, dtype=complex)
        C[np.argmin(abs(ells - target_mode[0]))] = 1.0
        
        return 0., C, ells
    
    if type(mode) is list:
        omega = 0.0
        for ell, m, n, sign in mode:
            omega_prime, C, ells = omega_and_C(
                (ell, m, n, sign), target_mode, M_f, chi_f
            )
            omega += omega_prime

        ells = np.arange(max(2, abs(target_mode[1])), ells[-1] + 1, 1)
        C = np.zeros_like(ells, dtype=complex)
        C[np.argmin(abs(ells - target_mode[0]))] = 1.0

        return omega, C, ells
    else:
        pass

    ell, m, n, sign = mode
    if sign == +1:
        mode_seq = _ksc(-2, ell, m, n)
    elif sign == -1:
        mode_seq = _ksc(-2, ell, -m, n)
    else:
        raise ValueError(
            "Last element of mode label must be "
            "+1 or -1, instead got {}".format(sign)
        )

    ells = qnm.angular.ells(-2, m, mode_seq.l_max)

    try:
        Momega, _, C = mode_seq(chi_f, store=True)
    except:
        Momega, _, C = mode_seq(chi_f, interp_only=True)

    if sign == -1:
        Momega = -np.conj(Momega)
        C = (-1) ** (ell + ells) * np.conj(C)

    # convert from M*\omega to \omega
    omega = Momega / M_f

    return omega, C, ells


def compute_stable_window(
    QNM, t_0s, CV_tolerance=2.0e-2, min_t_0_window=None, min_t_0_window_factor=10.0, min_A_tolerance=0.
):
    """Find the largest stable window that meets self.CV_tolerance.

    Parameters
    ----------
    QNM : ringdown.QNM
        QNM under consideration.
    t_0s : ndarray
        time array over which the QNM was fit.
    CV_tolerance : float
        minimum coefficient of variation to QNM to be considered stable.
        [Default: 2.e-2]
    min_t_0_window : float
        minimum window over fitting start times to consider.
        [Default: -np.log(min_t_0_window_factor) / QNM.omega.imag.]
    min_t_0_window_factor : float
        factor by which to change the minimum stable window;
        this corresponds to the amount the amplitude should decay over the window.
        [Default: 10.0]
    min_A_tolerance : float
        minimum amplitude to consider physical.
        [Default: 0.]

    Returns
    -------
    best_window : tuple
        window over which the QNM has a CV below CV_tolerance.
    true_min_CV : float
        coefficient of variation over window.
    """
    min_CV = np.inf
    best_window = (0.0, 0.0)

    min_A_tolerance_idx = np.argmin(abs(abs(QNM.A_time_series * np.exp(-1j * QNM.omega * t_0s)) - min_A_tolerance)) + 1

    d_window_size = np.diff(t_0s)[0]
    if min_t_0_window is None:
        try:
            window_size = max(
                4 * d_window_size,
                (
                    round(-np.log(min_t_0_window_factor) / QNM.omega.imag / d_window_size)
                    * d_window_size
                ),
            )
        except:
            window_size = 4 * d_window_size

    idx1 = 0
    while t_0s[idx1] + window_size < t_0s[:min_A_tolerance_idx][-1]:
        idx2 = np.argmin(abs(t_0s - (t_0s[idx1] + window_size)))
        CV = np.std(QNM.A_time_series[idx1:idx2]) / np.mean(
            abs(QNM.A_time_series[idx1:idx2])
        )
        if CV < min_CV:
            min_CV = CV
            best_idx1 = idx1
            best_idx2 = idx2
            
        idx1 += 1

    if min_CV < CV_tolerance:
        best_window = (t_0s[best_idx1], t_0s[best_idx2])
        
    return best_window, min_CV


def fit_damped_sinusoid_functions(t, fixed_frequencies, free_frequencies, t_ref=0.0):
    """Damped sinusoid fitting function for varpro.
    Computes Phi (the waveform) and
    dPhi (the derivative of the waveform w.r.t. nonlinear parameters).

    Parameters
    ----------
    t : ndarray
        times.
    fixed_frequencies : list
        frequencies of linear parameters.
    free_frequencies : list
        frequencies of nonlinear parameters.
    t_ref : float
        reference time for complex amplitudes.
        [Default: 0.]
    """
    N_fixed = len(fixed_frequencies) // 2
    N_free = len(free_frequencies) // 2
    N = N_fixed + N_free

    omegas = [
        fixed_frequencies[2 * i] + 1j * fixed_frequencies[2 * i + 1]
        for i in range(N_fixed)
    ]
    omegas += [
        free_frequencies[2 * i] + 1j * free_frequencies[2 * i + 1]
        for i in range(N_free)
    ]

    # Construct Phi, with the four terms (per QNM) decomposed as
    # QNM = term1 + term2 + term3 + term4, where term1 and term2 are the real components
    # and term 3 and term 4 are the imaginary components. Specifically, these are
    # (a + i * b) * exp(-i \omega t)] =
    # a Re[exp(-i \omega t)] - b * Im[exp(-i \omega t)] +
    # i * (a * Im[exp(-i \omega t)] + b * Im[exp(-i \omega t)]).
    # We will put the real terms in the 1st part of Phi, and the imaginary terms in the 2nd part
    Phi = np.zeros((2 * t.size, 2 * N))
    for i in range(N):
        # re
        # term 1
        Phi[: t.size, 2 * i] = np.real(np.exp(-1j * omegas[i] * (t - t_ref)))
        # term 2
        Phi[: t.size, 2 * i + 1] = -np.imag(np.exp(-1j * omegas[i] * (t - t_ref)))
        # im
        # term 3
        Phi[t.size :, 2 * i] = np.imag(np.exp(-1j * omegas[i] * (t - t_ref)))
        # term 4
        Phi[t.size :, 2 * i + 1] = np.real(np.exp(-1j * omegas[i] * (t - t_ref)))

    # We have 4*N terms per Phi entry (4 terms (see above))
    # and 2*N_free parameters, since each frequency has a real and imaginary part.
    # So there Phi must be of length (4*N)*(2*N_free).
    # We'll order the nonlinear parameter dependence in the trivial way, i.e., 0, 1, 2, ...
    # but with the fixed QNMs first.
    Ind = np.array(
        [
            [i // (2 * N_free) for i in range((2 * N) * (2 * N_free))],
            (2 * N) * list(np.arange(2 * N_free)),
        ]
    )

    # Construct dPhi, where each of the 4 terms (per QNM), if the QNM is free, has two components.
    dPhi = np.zeros((2 * t.size, (2 * N) * (2 * N_free)))
    # Loop over freqs
    for freq in range(N):
        # Loop over terms in real and imaginary parts,
        # i.e., if term == 0 then we're considering term1 and term3
        # while if term == 1 then we're considering term2 and term4
        for term in range(2):
            # Loop over the number of freq_derivs we have to take
            # which is just the number of free QNMs
            for freq_deriv in range(N_free):
                # shift to current QNM, shift to current term, shift to current frequency
                idx = (2 * N_free) * (2 * freq) + (2 * N_free) * term + 2 * freq_deriv

                # First, set the dPhi terms to zero when they correspond to a QNM w/ fixed frequency
                if freq - N_fixed != freq_deriv:
                    # term1/term2
                    # deriv w.r.t real part of freq
                    dPhi[: t.size, idx] = 0
                    # deriv w.r.t imag part of freq
                    dPhi[: t.size, idx + 1] = 0
                    # term3/term4
                    # deriv w.r.t real part of freq
                    dPhi[t.size :, idx] = 0
                    # deriv w.r.t imag part of freq
                    dPhi[t.size :, idx + 1] = 0
                else:
                    if term == 0:
                        # term 1
                        # deriv w.r.t real part of freq
                        dPhi[: t.size, idx] = np.real(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[: t.size, idx + 1] = np.real(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # term 3
                        # deriv w.r.t real part of freq
                        dPhi[t.size :, idx] = np.imag(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[t.size :, idx + 1] = np.imag(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                    else:
                        # term 2
                        # deriv w.r.t real part of freq
                        dPhi[: t.size, idx] = -np.imag(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[: t.size, idx + 1] = -np.imag(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # term 4
                        # deriv w.r.t real part of freq
                        dPhi[t.size :, idx] = np.real(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[t.size :, idx + 1] = np.real(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )

    return Phi, dPhi, Ind


def merge_fit_models(fit_QNM_models, t_0s):
    """Merge fit_QNM_models from varios times into one.

    Parameters
    ----------
    fit_QNM_models : list
        list of QNMModel objects.
    t_0s : ndarray
        fit start times for each model.
    """
    As = []
    omegas = []
    L2_norms = []
    mismatches = []

    t_0s_in_model = []
    for i, fit_QNM_model in enumerate(fit_QNM_models):
        if fit_QNM_model is None:
            continue
        t_0s_in_model.append(t_0s[i])
        As.append([QNM.A for QNM in fit_QNM_model.QNMs])
        omegas.append(
            [
                non_QNM_damped_sinusoid.omega
                for non_QNM_damped_sinusoid in fit_QNM_model.non_QNM_damped_sinusoids
            ]
        )
        try:
            L2_norms.append(fit_QNM_model.L2_norm)
            mismatches.append(fit_QNM_model.mismatch)
        except:
            pass
    As = np.array(As)
    omegas = np.array(omegas)

    for i in range(len(fit_QNM_models)):
        try:
            fit_QNM_model = fit_QNM_models[i].copy()
            break
        except:
            pass
    if fit_QNM_model is None:
        raise ValueError("Varpro failed to fit at every t_0.")

    fit_QNM_model.t_0s = np.array(t_0s_in_model)

    for i, QNM in enumerate(fit_QNM_model.QNMs):
        QNM.A = None
        QNM.A_time_series = As[:, i]

    for i, non_QNM_damped_sinusoid in enumerate(fit_QNM_model.non_QNM_damped_sinusoids):
        non_QNM_damped_sinusoid.omegas = omegas[:, i]

    fit_QNM_model.L2_norms = np.array(L2_norms)
    fit_QNM_model.mismatches = np.array(mismatches)

    return fit_QNM_model


class DampedSinusoid:
    """Damped Sinusoid.

    Attributes
    ----------
    target_mode : tuple
        (\ell, m) for damped sinusoid to mix into.
    omega : complex
        complex frequency.
    A : complex
        complex amplitude.
    """

    def __init__(self, target_mode, omega, A=None):
        self.target_mode = target_mode
        self.omega = omega
        self.A = A

        return


class QNM:
    """QNM.

    Attributes
    ----------
    mode : tuple
        (\ell, m, n, sign) of QNM.
    target_mode : tuple
        (\ell, m) for 2nd order QNM to mix into;
        if QNM is first order, then this is just (\ell, m).
    A : complex
        complex QNM amplitude.
    """

    def __init__(self, mode, target_mode=None, A=None):
        self.mode = mode
        self.is_first_order_QNM = type(self.mode) is tuple

        if not self.is_first_order_QNM:
            self.mode = sorted(self.mode)

        if target_mode is None:
            if self.mode == (0,0,0,0):
                raise ValueError("constants must be provided a target mode!")
            if not self.is_first_order_QNM:
                raise ValueError("higher order QNMs must be provided a target mode!")
            else:
                self.target_mode = self.mode[:2]
        else:
            self.target_mode = target_mode

        self.A = A

        return

    def copy(self):
        return copy.deepcopy(self)

    def compute_omega_and_C(self, M_f, chi_f):
        """Compute the QNM's frequency and spherical-spheroidal mixing coefficients.

        Parameters
        ----------
        M_f : float
            remnant mass.
        chi_f : float
            remnant dimensionless spin magnitude.

        Returns
        -------
        omega : complex
            QNM frequency.
        C : complex ndarray
            QNM spherical-spheroidal mixing coefficients.
        """
        try:
            if self.M_f == M_f and self.chi_f == chi_f:
                return self.omega, self.C
        except:
            pass

        omega, C, ells = omega_and_C(self.mode, self.target_mode, M_f, chi_f)
        self.omega = omega
        self.C = C
        self.ells = ells

        self.M_f = M_f
        self.chi_f = chi_f

        return self.omega, self.C

    def mirror(self):
        """Compute the mirror mode."""
        mode = list(self.mode)
        target_mode = list(self.target_mode)
        if self.is_first_order_QNM:
            mode[1] = -mode[1]
            mode[3] = -mode[3]

            target_mode[1] = -target_mode[1]

            return QNM(tuple(mode), tuple(target_mode))
        else:
            mode_mirrors = []
            for mode in self.mode:
                mode_mirror = list(mode)
                mode_mirror[1] = -mode_mirror[1]
                mode_mirror[3] = -mode_mirror[3]

                mode_mirrors.append(tuple(mode_mirror))

            target_mode[1] = -target_mode[1]

            return QNM(mode_mirrors, tuple(target_mode))


class QNMModel:
    """QNM model for an NR waveform.

    Attributes
    ----------
    M_f : float
        remnant mass.
    chi_f : float
        remnant dimensionless spin magnitude.
    QNMs : list of QNMs
        list of QNMs.
    non_QNM_damped_sinusoids : list of DampedSinusoids
        list of non QNM damped sinusoids, e.g., free frequency fits.
    """

    def __init__(self, M_f, chi_f, QNMs=None, non_QNM_damped_sinusoids=None):
        self.M_f = M_f
        self.chi_f = chi_f
        if QNMs is None:
            self.QNMs = []
        else:
            self.QNMs = QNMs
        if non_QNM_damped_sinusoids is None:
            self.non_QNM_damped_sinusoids = []
        else:
            self.non_QNM_damped_sinusoids = non_QNM_damped_sinusoids

    def copy(self):
        return copy.deepcopy(self)

    def load(self, filename):
        return joblib.load(filename)

    def save(self, filename):
        joblib.dump(self, filename)

    def compute_omegas_and_Cs(self):
        """Compute the QNMs' frequencies and spherical-spheroidal mixing coefficients.

        Parameters
        ----------
        M_f : float
            remnant mass.
        chi_f : float
            remnant dimensionless spin magnitude.
        """
        for QNM in self.QNMs:
            QNM.compute_omega_and_C(self.M_f, self.chi_f)

    def filter_based_on_mode(self, mode):
        """Filter QNM model to only correspond to QNMs
           that mix into the input mode.

        Parameters
        ----------
        mode : tuple
            (\ell, m) mode to filter QNMs by.

        Returns
        -------
        QNM_model_filtered : ringdown.QNMModel
            model of filtered QNMs.
        """
        filtered_QNMs = []
        for QNM in self.QNMs:
            if QNM.target_mode[1] == mode[1]:
                filtered_QNMs.append(QNM.copy())

        QNM_model_filtered = QNMModel(self.M_f, self.chi_f, filtered_QNMs)

        return QNM_model_filtered

    def integrate(self, integration_number=1):
        """Integrate the model in time `integration_number` times.

        Parameters
        ----------
        integration_number : int
            number of times to integrate.
        [Default: 1]

        Returns
        -------
        QNM_model : ringdown.QNMModel
            integrated QNMModel.
        """
        QNM_model = self.copy()

        integrated = False
        for QNM in QNM_model.QNMs:
            try:
                QNM.A = QNM.A / (-1j * QNM.omega) ** integration_number
                integrated = True
            except:
                pass

            try:
                QNM.A_std = QNM.A_std / abs((-1j * QNM.omega) ** integration_number)
                integrated = True
            except:
                pass

            try:
                QNM.A_time_series = (
                    QNM.A_time_series / (-1j * QNM.omega) ** integration_number
                )
                integrated = True
            except:
                pass

        if not integrated:
            colored(
                "********\n" + "Warning: no amplitude data to change.\n" + "********",
                "red",
            )

        return QNM_model

    def analyze_model_time_series(
        self, CV_tolerance=2.0e-2, min_t_0_window=None, min_t_0_window_factor=10.0, min_A_tolerance=0.
    ):
        """Analyze time series data of model, i.e.,
           find the largest stable window for each QNM
           and extract the amplitude over said window.

        Parameters
        ----------
        CV_tolerance : float
            minimum coefficient of variation to QNM to be considered stable.
            [Default: 2.e-2]
        min_t_0_window : float
            minimum window over fitting start times to consider.
            [Default: -np.log(min_t_0_window_factor) / QNM.omega.imag.]
        min_t_0_window_factor : float
            factor by which to change the minimum stable window;
            this corresponds to the amount the amplitude should decay over the window.
            [Default: 10.0]
        min_A_tolerance : float
            minimum amplitude to consider physical.
            [Default: 0.]

        Returns
        -------
        QNM_model : ringdown.QNMModel
            model of QNMs with analyzed time series data.
        """
        QNM_model = self.copy()

        QNM_model.compute_omegas_and_Cs()
        for QNM in QNM_model.QNMs:
            stable_window, CV = compute_stable_window(
                QNM, QNM_model.t_0s, CV_tolerance, min_t_0_window, min_t_0_window_factor, min_A_tolerance
            )
            QNM.stable_window = stable_window
            QNM.CV = CV

            QNM.A = np.mean(
                QNM.A_time_series[
                    np.argmin(
                        abs(QNM_model.t_0s - stable_window[0])
                    ) : np.argmin(abs(QNM_model.t_0s - stable_window[1]))
                    + 1
                ].real
            ) + 1j * np.mean(
                QNM.A_time_series[
                    np.argmin(
                        abs(QNM_model.t_0s - stable_window[0])
                    ) : np.argmin(abs(QNM_model.t_0s - stable_window[1]))
                    + 1
                ].imag
            )
            QNM.A_std = np.std(
                QNM.A_time_series[
                    np.argmin(
                        abs(QNM_model.t_0s - stable_window[0])
                    ) : np.argmin(abs(QNM_model.t_0s - stable_window[1]))
                    + 1
                ].real
            ) + 1j * np.std(
                QNM.A_time_series[
                    np.argmin(
                        abs(QNM_model.t_0s - stable_window[0])
                    ) : np.argmin(abs(QNM_model.t_0s - stable_window[1]))
                    + 1
                ].imag
            )

        return QNM_model

    def compute_waveform(self, h_template, t_i=None, t_f=None, t_ref=0.0):
        """Compute a waveform from self.QNMs.

        Parameters
        ----------
        h_template : scri.WaveformModes
            template waveform whose time array will be used.
        t_i : scri.WaveformModes
            data for t < t_i is set to zero.
        t_f : float
            data for t > t_f is set to zero.
        t_ref : float
            reference time for QNM amplitudes.
            [Default: 0.]

        Returns
        -------
        h_QNM : scri.WaveformModes
            waveform corresponding to self.QNMs.
        """

        data = np.zeros_like(h_template.data, dtype=complex)

        self.compute_omegas_and_Cs()
        for QNM in self.QNMs:
            ell, m = QNM.target_mode
            for _l, _m in h_template.LM:
                if m == _m:
                    data[:, h_template.index(_l, _m)] += (
                        QNM.A
                        * QNM.C[QNM.ells == _l]
                        * np.exp(-1j * QNM.omega * (h_template.t - t_ref))
                    )
        for non_QNM in self.non_QNM_damped_sinusoids:
            _l, _m = non_QNM.target_mode
            data[:, h_template.index(_l, _m)] += non_QNM.A * np.exp(
                -1j * non_QNM.omega * (h_template.t - t_ref)
            )

        if not t_i is None:
            data[: np.argmin(abs(h_template.t - t_i))] *= 0.0

        if not t_f is None:
            data[np.argmin(abs(h_template.t - t_f)) + 1 :] *= 0.0

        h_QNM = h_template.copy()
        h_QNM.data = data

        return h_QNM

    def fit_LLSQ(
        self,
        h_NR,
        modes=None,
        t_i=0.0,
        t_f=100.0,
        t_ref=0.0,
        compute_fit_errors=False,
        return_h_NR_fitted=False,
    ):
        """Fit QNM model to an NR waveform via linear-least squares.

        Parameters
        ----------
        h_NR : scri.WaveformModes
            NR waveform to fit to.
        modes : list
            list of (\ell, m) tuples to fit to.
            [Default: None, i.e., every mode.]
        t_i : float or ndarray
            earliest time to fit.
            [Default: 0.]
        t_f : float
            latest time to fit.
            [Default: 100.]
        t_ref : float
            reference time for QNM amplitudes.
            [Default: 0.]
        compute_fit_errors : bool
            whether or not to compute the fit L2 norm and mismatch.
            [Default: False].
        return_h_NR_fitted : bool
            whether or not to return the fitted NR waveform.
            [Default: False].

        Returns
        -------
        fit_QNMs : QNMModel
            QNM model fit to NR waveform.
        h_NR_fitted : scri.WaveformModes
            NR waveform that was fit.
        """
        fit_QNM_model = self.copy()

        h_NR_fitted = h_NR.copy()[
            np.argmin(abs(h_NR.t - t_i)) : np.argmin(abs(h_NR.t - t_f)) + 1
        ]
        dt = h_NR_fitted.t[0]
        h_NR_fitted.t -= dt
        t_ref -= dt

        m_list = []
        [
            m_list.append(m)
            for (_, m) in [QNM.target_mode for QNM in self.QNMs]
            if m not in m_list
        ]
        # break problem into one m at a time;
        # the m's are decoupled, and the truncation in ell for each m is different.
        for m in m_list:
            QNMs_matching_m = [
                (i, QNM) for i, QNM in enumerate(self.QNMs) if QNM.target_mode[1] == m
            ]

            # restrict the modes included in the least squares fit to the modes of interest.
            ell_min_m = h_NR_fitted.ell_min
            ell_max_m = h_NR_fitted.ell_max
            if modes is None:
                data_index_m = [
                    h_NR_fitted.index(l, m) for l in range(ell_min_m, ell_max_m + 1)
                ]
            else:
                data_index_m = [
                    h_NR_fitted.index(l, m)
                    for l in range(ell_min_m, ell_max_m + 1)
                    if (l, m) in modes
                ]
                ell_min_m = min(np.array([_l for (_l, _m) in modes if m == _m]))
                ell_max_m = max(np.array([_l for (_l, _m) in modes if m == _m]))

            A = np.zeros((h_NR_fitted.t.size, ell_max_m - ell_min_m + 1), dtype=complex)
            B = np.zeros(
                (h_NR_fitted.t.size, ell_max_m - ell_min_m + 1, len(QNMs_matching_m)),
                dtype=complex,
            )

            h_NR_fitted_kurt = h_NR_fitted[:, : ell_max_m + 1]
            A = h_NR_fitted_kurt.data[:, data_index_m]
            for mode_index, (i, QNM) in enumerate(QNMs_matching_m):
                QNM_w_unit_A = QNM.copy()
                QNM_w_unit_A.A = 1.0

                QNM_model_w_unit_A = QNMModel(self.M_f, self.chi_f, [QNM_w_unit_A])

                h_QNM = QNM_model_w_unit_A.compute_waveform(
                    h_NR_fitted_kurt, t_ref=t_ref
                )
                
                B[:, :, mode_index] = h_QNM.data[:, data_index_m]

            A = np.reshape(A, h_NR_fitted_kurt.t.size * (ell_max_m - ell_min_m + 1))
            B = np.reshape(
                B,
                (
                    h_NR_fitted_kurt.t.size * (ell_max_m - ell_min_m + 1),
                    len(QNMs_matching_m),
                ),
            )
            C = np.linalg.lstsq(B, A, rcond=None)

            count = 0
            for i, QNM in QNMs_matching_m:
                fit_QNM_model.QNMs[i].A = C[0][count]
                count += 1

        h_NR_fitted.t += dt

        if compute_fit_errors:
            h_QNM = fit_QNM_model.compute_waveform(h_NR_fitted)
            fit_QNM_model.L2_norm = utils.compute_L2_norm(
                h_NR_fitted, h_QNM, modes=modes
            )
            fit_QNM_model.mismatch = utils.compute_mismatch(
                h_NR_fitted, h_QNM, modes=modes
            )

        if return_h_NR_fitted:
            return fit_QNM_model, h_NR_fitted
        else:
            return fit_QNM_model

    def fit_varpro(
        self,
        h_NR,
        mode,
        t_i=0,
        t_f=100,
        t_ref=0.0,
        N_free_frequencies=0,
        initial_guess=None,
        bounds=None,
        ftol=0.,
        gtol=0.
    ):
        """Fit QNM model to an NR waveform via varpro.

        Parameters
        ----------
        h_NR : scri.WaveformModes
            NR waveform to fit to.
        mode : tuple
            (\ell, m) mode to fit to.
        t_i : float
            earliest time to fit.
            [Default: 0.]
        t_f : float
            latest time to fit.
            [Default: 100.]
        t_ref : float
            reference time for QNM amplitudes.
            [Default: 0.]
        N_free_frequencies : int
            number of free frequencies to include in the fit.
            [Default: 0]
        initial_guess : list
            initial guesses for free frequencies.
            [Default: [0.5, -0.2] * N_free_frequencies.
        bounds : list
            bounds for free frequencies.
            [Default: [(-np.inf, np.inf), (-np.inf, 0)] * N_free_frequencies]
        ftol : float
            ftol used in nonlinear least squares optimization.
            [Default: 0.]
        gtol: float
            gtol used in nonlinear least squares optimization.
            [Default: 0.]

        Returns
        -------
        fit_QNM_model : QNMModel
            QNM model fit to NR waveform.
        h_NR_fitted : scri.WaveformModes
            NR waveform that was fit.
        """
        fit_QNM_model = QNMModel(self.M_f, self.chi_f, self.QNMs)

        L, M = mode

        h_NR_fitted = h_NR[
            np.argmin(abs(h_NR.t - t_i)) : np.argmin(abs(h_NR.t - t_f)) + 1
        ]

        N_QNMs = len(self.QNMs)
        N_damped_sinusoids = N_QNMs + N_free_frequencies

        w = np.ones(2 * h_NR_fitted.t.size)

        if initial_guess is not None:
            if len(initial_guess) != 2 * N_free_frequencies:
                raise ValueError(f"Initial guess = {initial_guess} must be of length 2 * N_free_frequencies = {2 * N_free_frequencies}")
        else:
            if N_free_frequencies == 1:
                initial_guess = np.array([0.5, -0.2])
            elif N_free_frequencies == 2:
                initial_guess = np.array([0.5, -0.2, -0.5, -0.2])
            elif N_free_frequencies == 3:
                initial_guess = np.array([0.5, -0.2, -0.5, -0.2, 0.0, -0.2])
            elif N_free_frequencies == 4:
                initial_guess = np.array([0.5, -0.2, -0.5, -0.2, 0.0, -0.2, 0.5, -0.4])
            elif N_free_frequencies == 5:
                initial_guess = np.array([0.5, -0.2, -0.5, -0.2, 0.0, -0.2, 0.5, -0.4, -0.5, -0.4])
            elif N_free_frequencies == 6:
                initial_guess = np.array([0.5, -0.2, -0.5, -0.2, 0.0, -0.2, 0.5, -0.4, -0.5, -0.4, 0., -0.4])

        QNM_frequencies = []
        for QNM in self.QNMs:
            QNM_frequencies.append(QNM.omega.real)
            QNM_frequencies.append(QNM.omega.imag)

        if bounds is None:
            bounds = ([-np.inf, -np.inf], [np.inf, 0])

        # run varpro
        try:
            (
                free_frequencies,
                As,
                wresid,
                wresid_norm,
                y_est,
                CorMx,
                std_dev_params,
            ) = varpro.varpro(
                h_NR_fitted.t,
                np.concatenate(
                    (
                        h_NR_fitted.data[:, h_NR_fitted.index(L, M)].real,
                        h_NR_fitted.data[:, h_NR_fitted.index(L, M)].imag,
                    )
                ),
                w,
                initial_guess,
                2 * N_damped_sinusoids,
                lambda alpha: fit_damped_sinusoid_functions(
                    h_NR_fitted.t, QNM_frequencies, alpha
                ),
                bounds=(bounds[0] * N_free_frequencies, bounds[1] * N_free_frequencies),
                ftol=ftol,
                gtol=gtol,
                verbose=False,
            )
        except:
            return None

        for i, QNM in enumerate(fit_QNM_model.QNMs):
            QNM.A = As[2 * i] + 1j * As[2 * i + 1]

        for i in range(N_free_frequencies):
            fit_QNM_model.non_QNM_damped_sinusoids.append(
                DampedSinusoid(
                    mode,
                    free_frequencies[2 * i] + 1j * free_frequencies[2 * i + 1],
                    As[2 * (N_QNMs + i)] + 1j * As[2 * (N_QNMs + i) + 1],
                )
            )

        fit_QNM_model.non_QNM_damped_sinusoids = [
            x for _, x in sorted(
                zip(
                    [non_QNM_damped_sinusoid.omega.real for non_QNM_damped_sinusoid in fit_QNM_model.non_QNM_damped_sinusoids],
                    fit_QNM_model.non_QNM_damped_sinusoids
                ),
                key=lambda pair: pair[0]
            )
        ]

        return fit_QNM_model

    def fit(
        self,
        h_NR,
        modes=None,
        t_i=0,
        t_f=100,
        t_ref=0.0,
        N_free_frequencies=0,
        initial_guess=None,
        bounds=None,
        ftol=0.0,
        gtol=0.0,
        recycle_varpro_results_as_initial_guess=False,
        n_procs=None,
    ):
        """Fit QNM model to an NR waveform.

        Parameters
        ----------
        h_NR : scri.WaveformModes
            NR waveform to fit to.
        modes : list
            list of (\ell, m) tuples to fit to.
            [Default: None, i.e., every mode.]
        t_i : float or ndarray
            earliest time to fit; if ndarray, fits every time.
            [Default: 0.]
        t_f : float
            latest time to fit.
            [Default: 100.]
        t_ref : float
            reference time for QNM amplitudes.
            [Default: 0.]
        compute_fit_errors : bool
            whether or not to compute the fit L2 norm and mismatch.
            [Default: False]
        N_free_frequencies : int
            number of free frequencies to include in the varpro fit;
            if N_free_frequencies != 0, modes can only be one mode.
            [Default: 0]
        initial_guess : list
            initial guesses for varpro free frequencies.
            [Default: [0.5, -0.2] * N_free_frequencies.
        bounds : list
            bounds for varpro free frequencies.
            [Default: [(-np.inf, np.inf), (-np.inf, 0)] * N_free_frequencies]
        ftol : float
            ftol used in varpro's nonlinear least squares optimization.
            [Default: 0.]
        gtol: float
            gtol used in varpro's nonlinear least squares optimization.
            [Default: 0.]
        recycle_varpro_results_as_initial_guess : bool
            whether or not to use the varpro results for subsequent initial guesses.
            [Default: False]
        n_procs : int
            number of cores to use; if -1, no multiprocessing is performed.
            [Default: maximum number of cores]

        Returns
        -------
        fit_QNM_model : QNMModel
            QNM model fit to NR waveform.
        h_NR_fitted : scri.WaveformModes
            NR waveform that was fit.
        """
        if not (type(modes) is list or modes is None):
            raise ValueError(f"modes = {modes} must be a list.")

        if N_free_frequencies != 0 and len(modes) > 1:
            raise ValueError(
                f"N_free_frequencies = {N_free_frequencies} > 0, so modes = {modes} can only be of length one. "
                + "(Atleast until varpro is extended to work over the two-sphere.)"
            )

        if n_procs is None:
            n_procs = multiprocessing.cpu_count()

        if n_procs != -1 and recycle_varpro_results_as_initial_guess:
            raise ValueError("Cannot recycle varpro results as initial guess when multiprocessing.")

        if not type(t_i) is np.ndarray:
            if N_free_frequencies == 0:
                return self.fit_LLSQ(h_NR, modes=modes, t_i=t_i, t_f=t_f, t_ref=t_ref)
            else:
                return self.fit_varpro(
                    h_NR,
                    mode=modes[0],
                    t_i=t_i,
                    t_f=t_f,
                    t_ref=t_ref,
                    N_free_frequencies=N_free_frequencies,
                    initial_guess=initial_guess,
                    bounds=bounds,
                    ftol=ftol,
                    gtol=gtol,
                )
        else:
            t_i_array = t_i
            if n_procs != -1:
                with multiprocessing.Pool(n_procs) as pool:
                    if N_free_frequencies == 0:
                        fit_QNM_models = pool.map(
                            partial(
                                self.fit_LLSQ,
                                h_NR,
                                modes,
                                t_f=t_f,
                                t_ref=t_ref,
                                return_h_NR_fitted=False,
                            ),
                            t_i_array[::-1],
                        )
                    else:
                        fit_QNM_models = pool.map(
                            partial(
                                self.fit_varpro,
                                h_NR,
                                modes[0],
                                t_f=t_f,
                                t_ref=t_ref,
                                N_free_frequencies=N_free_frequencies,
                                initial_guess=initial_guess,
                                bounds=bounds,
                                ftol=ftol,
                                gtol=gtol,
                            ),
                            t_i_array[::-1],
                        )

            else:
                if N_free_frequencies == 0:
                    fit_QNM_models = [
                        self.fit_LLSQ(
                            h_NR,
                            modes,
                            t_i=t_i,
                            t_f=t_f,
                            t_ref=t_ref,
                            return_h_NR_fitted=False,
                        )
                        for t_i in t_i_array[::-1]
                    ]
                else:
                    fit_QNM_models = []
                    for i, t_i in enumerate(t_i_array[::-1]):
                        if i == 0:
                            initial_guess = None
                        else:
                            initial_guess = []
                            for non_QNM_damped_sinusoid in fit_QNM_models[-i].non_QNM_damped_sinusoids:
                                initial_guess.append(non_QNM_damped_sinusoid.omega.real)
                                initial_guess.append(non_QNM_damped_sinusoid.omega.imag)
                            initial_guess = np.array(initial_guess)
                        fit_QNM_models.append(
                            self.fit_varpro(
                                h_NR,
                                modes[0],
                                t_i=t_i,
                                t_f=t_f,
                                t_ref=t_ref,
                                N_free_frequencies=N_free_frequencies,
                                initial_guess=initial_guess,
                                bounds=bounds,
                                ftol=ftol,
                                gtol=gtol,
                            )
                        )

            fit_QNM_model = merge_fit_models(fit_QNM_models[::-1], t_i_array)

            return fit_QNM_model
