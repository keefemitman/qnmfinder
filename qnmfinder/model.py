import numpy as np

import sxs
import scri
from scri.asymptotic_bondi_data.map_to_superrest_frame import WM_to_MT, MT_to_WM
from quaternion.calculus import indefinite_integral as integrate

from . import utils
from . import ringdown

import multiprocessing

from termcolor import colored

import matplotlib.pyplot as plt


class QNMModelBuilder:
    """Build QNM model for an NR waveform.

    Attributes
    ----------
    h_NR : scri.WaveformModes
        NR waveform to fit to.
    M_f : float
        remnant mass.
    chi_f : float
        remnant dimensionless spin magnitude.
    t_i : float
        earliest time (relative to peak luminosity) to fit.
        [Default: 0.]
    t_f : float
        latest time (relative to peak luminsoity) to fit.
        [Default: 100.]
    t_0_f : float
        latest start time (relative to peak luminosity) to fit.
        [Default: 80.]
    d_t_0 : float
        spacing between t_i and t_0_f to iterate over.
        [Default: 1.]
    ell_min_NR : int
        minimum QNM \ell value to consider.
        [Default: 2]
    ell_max_NR : int
        maximum QNM \ell value to consider.
        [Default: 5]
    modes : list
        (\ell, m) modes of NR waveform to fit.
        [Default: use all modes.]
    mode_power_tol : float
        lowest unmodeled power to consider.
        [Default: 0.]
    ell_min_QNM : int
        minimum QNM \ell value to consider.
        [Default: 2]
    ell_max_QNM : int
        maximum QNM \ell value to consider.
        [Default: 5]
    N_max : int
        maximum overtone value to consider.
        [Default: 4]
    include_2nd_order_QNMs : bool
        whether or not to include 2nd order QNMs
        [Default: True]
    require_1st_order_QNM_existence : bool
        require 1st order QNM to exist in model to be considered
        as a component of a 2nd order QNM.
        [Default: True]
    t_ref : float
        reference time (relative to peak luminosity) for QNM amplitudes.
        [Default: 0.]
    allow_more_than_one_free_frequency : bool
        whether or not to allow the use of two free frequencies if one fails.
        [Default: True]
    frequency_tolerance : float
        minimum modulus to match free frequency to QNM frequency.
        [Default: 1.e-1]
    CV_tolerance : float
        minimum coefficient of variation to QNM to be considered stable.
        [Default: 5.e-2]
    min_t_0_window : float
        minimum window over fitting start times to consider.
        [Default: -min_t_0_window_factor / QNM.omega.imag.]
    min_t_0_window_factor : float
        factor by which to change the minimum stable window.
        [Default: 0.5]
    reset_after_adding_QNM : bool
        whether or not to reset the fitting start time iteration
        after successfully adding a new QNM to the model.
        [Default: True]
    n_procs : int
        number of cores to use; if 'auto', optimal number based on the number of fit start times;
        if None, maximum number of cores; if -1, no multiprocessing is performed.
        [Default: 'auto']
    verbose : bool
        whether or not to print status updates.
        [Default: False]
    """

    def __init__(
        self,
        h_NR,
        M_f,
        chi_f,
        t_i=0,
        t_f=100,
        t_0_f=80,
        d_t_0=1,
        ell_min_NR=2,
        ell_max_NR=5,
        modes=None,
        mode_power_tol=0.0,
        ell_min_QNM=2,
        ell_max_QNM=5,
        N_max=4,
        include_2nd_order_QNMs=True,
        require_1st_order_QNM_existence=True,
        t_ref=0.0,
        allow_more_than_one_free_frequency=True,
        frequency_tolerance=1.0e-1,
        CV_tolerance=5.0e-2,
        min_t_0_window=None,
        min_t_0_window_factor=0.5,
        reset_after_adding_QNM=True,
        n_procs="auto",
        verbose=False,
    ):
        if type(h_NR) != scri.waveform_modes.WaveformModes:
            try:
                h_NR = MT_to_WM(WM_to_MT(h_NR))
            except:
                raise ValueError("Cannot convert NR waveform to scri.WaveformModes.")

        if t_i > t_f:
            raise ValueError(f"t_i = {t_i} > t_f = {t_f}")

        if t_0_f > t_f:
            raise ValueError(f"t_0_f = {t_0_f} > t_f = {t_f}")

        if N_max < 0 or N_max >= 8:
            raise ValueError(f"Need 8 > N_max = {N_max} >= 0")

        if not modes is None and require_1st_order_QNM_existence:
            print(
                colored(
                    "********\n"
                    + "Warning: requiring 1st order QNM existence for 2nd order QNMs "
                    + "with modes not None may miss 2nd order QNMs.\n"
                    + "********",
                    "red",
                )
            )

        self.h_NR = h_NR.copy()
        self.M_f = M_f
        self.chi_f = chi_f
        self.t_i = t_i
        self.t_f = t_f
        self.t_0_f = t_0_f
        self.d_t_0 = d_t_0
        self.ell_min_NR = ell_min_NR
        self.ell_max_NR = ell_max_NR
        self.modes = modes
        self.mode_power_tol = mode_power_tol
        if self.mode_power_tol != 0.0:
            print(
                colored(
                    "********\n"
                    + "Warning: mode power tolerance not yet implemented.\n"
                    + "********",
                    "red",
                )
            )
        self.ell_min_QNM = ell_min_QNM
        self.ell_max_QNM = ell_max_QNM
        self.N_max = N_max
        self.include_2nd_order_QNMs = include_2nd_order_QNMs
        self.require_1st_order_QNM_existence = require_1st_order_QNM_existence
        self.t_ref = t_ref
        self.allow_more_than_one_free_frequency = allow_more_than_one_free_frequency
        self.frequency_tolerance = frequency_tolerance
        self.CV_tolerance = CV_tolerance
        self.min_t_0_window = min_t_0_window
        self.min_t_0_window_factor = min_t_0_window_factor
        self.reset_after_adding_QNM = reset_after_adding_QNM
        self.n_procs = n_procs

        self.verbose = verbose

        # fix NR waveform so t_peak = time of peak luminosity
        self.h_NR.t -= self.compute_t_peak()
        self.h_NR = self.h_NR[
            np.argmin(abs(self.h_NR.t - self.t_i)) : np.argmin(
                abs(self.h_NR.t - self.t_f)
            )
            + 1,
            self.ell_min_NR : self.ell_max_NR + 1,
        ]

        # construct t0s
        self.t_0s = np.arange(t_i, t_0_f + d_t_0, d_t_0)

        if self.n_procs == "auto":
            if self.t_0s.size <= 20:
                self.n_procs = -1
            else:
                self.n_procs = None

        # initialize QNM model
        self.update_model(ringdown.QNMModel(self.M_f, self.chi_f))

        return

    def compute_t_peak(self):
        """Compute the time of peak luminosity."""
        news = MT_to_WM(WM_to_MT(self.h_NR).dot, dataType=scri.hdot)

        index = np.argmax(news.norm())
        index = max(2, min(len(self.h_NR.t) - 3, index))

        # Fit 5 points with a 2nd-order polynomial
        test_times = self.h_NR.t[index - 2 : index + 2 + 1] - self.h_NR.t[index]
        test_funcs = news.norm()[index - 2 : index + 2 + 1]

        x_vecs = np.array([np.ones(5), test_times, test_times**2.0])
        y_vec = np.array([test_funcs.dot(v1) for v1 in x_vecs])
        inv_mat = np.linalg.inv(
            np.array([[v1.dot(v2) for v1 in x_vecs] for v2 in x_vecs])
        )
        coeffs = np.array([y_vec.dot(v1) for v1 in inv_mat])

        return self.h_NR.t[index] - coeffs[1] / (2.0 * coeffs[2])

    def update_model(self, QNM_model):
        """Update self.QNM_model to reflect the content of QNM_model.

        Parameters
        ----------
        QNM_model : ringdown.QNMModel
            model containing current set of QNMs.
        """
        self.QNM_model = QNM_model

        self.failed_QNM_modes = []

        self.mode_to_model = None

        self.N_free_frequencies = 1

        self.update_model_waveform()

        self.update_model_errors()

    def update_model_waveform(self):
        """Update self.h_QNM and self.h_residual to reflect the content of self.QNM_model."""
        self.h_QNM = self.QNM_model.compute_waveform(self.h_NR)

        self.h_residual = self.h_NR.copy()
        self.h_residual.data -= self.h_QNM.data

        self.update_model_errors()

    def update_model_errors(self):
        """Update self.L2_norms and self.mismatches to reflect the content of self.residual."""
        L2_norms = []
        mismatches = []
        for t_0 in self.t_0s:
            L2_norm = utils.compute_L2_norm(
                self.h_NR, self.h_QNM, t_i=t_0, modes=self.modes
            )
            L2_norms.append(L2_norm)

            mismatch = utils.compute_mismatch(
                self.h_NR, self.h_QNM, t_i=t_0, modes=self.modes
            )
            mismatches.append(mismatch)

        self.L2_norms = L2_norms
        self.mismatches = mismatches

        self.L2_norms_int = integrate(L2_norms, self.t_0s)[-1]
        self.mismatches_int = integrate(mismatches, self.t_0s)[-1]

    def find_most_unmodeled_mode(self, t_0=None):
        """Find the mode which has the most unmodeled power.

        Parameters
        ----------
        t_0 : float
            time beyond which to compute the power.
            [Default is self.t_i]

        Returns
        -------
        unmodeled_mode : tuple
            (\ell, m) mode with most unmodeled power.
        """
        if t_0 is None:
            t_0 = self.t_i

        idx_t_0 = np.argmin(abs(self.h_NR.t - t_0))

        if not self.modes is None:
            mode_indices = []
            for L, M in self.h_NR.LM:
                if (L, M) in self.modes:
                    mode_indices.append(True)
                else:
                    mode_indices.append(False)
        else:
            mode_indices = [True] * self.h_NR.LM.shape[0]
        mode_indices = np.array(mode_indices)

        unmodeled_power = integrate(
            abs(self.h_residual.data[idx_t_0:, mode_indices]) ** 2,
            self.h_residual.t[idx_t_0:],
            axis=0,
        )[-1]

        LMs_ranked = [
            x for _, x in sorted(zip(unmodeled_power, self.h_residual.LM[mode_indices]))
        ][::-1]

        LMs_and_power = [
            (LM, sorted(unmodeled_power)[::-1][i]) for i, LM in enumerate(LMs_ranked)
        ]

        unmodeled_mode = tuple(LMs_ranked[0])

        return unmodeled_mode, LMs_and_power

    def match_damped_sinusoid_to_QNM(self, damped_sinusoid):
        """Match a damped sinusoid to a QNM.

        Parameters
        ----------
        damped_sinusoid : ringdown.DampedSinusoid
            damped sinusoid to match to a QNM.

        Returns
        -------
        QNM : ringdown.QNM
            QNM that best matches the damped sinusoid's frequency.
        """
        min_d_omega = np.inf
        first_order_QNMs = [
            QNM.mode for QNM in self.QNM_model.QNMs if QNM.is_first_order_QNM
        ]
        second_order_QNMs = [
            QNM.mode for QNM in self.QNM_model.QNMs if not QNM.is_first_order_QNM
        ]
        for L1 in range(self.ell_min_QNM, self.ell_max_QNM + 1):
            if self.include_2nd_order_QNMs:
                M1s = range(-L1, L1 + 1)
            else:
                M1s = [damped_sinusoid.target_mode[1]]
            for M1 in M1s:
                for N1 in range(0, self.N_max + 1):
                    for S1 in [-1, +1]:
                        omega1 = ringdown.omega_and_C(
                            (L1, M1, N1, S1), (L1, M1), self.M_f, self.chi_f
                        )[0]

                        # filter 1st order QNM
                        # no repeats
                        is_not_repeat = not (L1, M1, N1, S1) in first_order_QNMs

                        # m values must match
                        m_value_matches = M1 == damped_sinusoid.target_mode[1]

                        # lower overtones must exist
                        lower_overtones_exist = True
                        if N1 != 0:
                            lower_overtones_exist = (
                                L1,
                                M1,
                                N1 - 1,
                                S1,
                            ) in first_order_QNMs

                        if is_not_repeat and m_value_matches and lower_overtones_exist:
                            d_omega = abs(omega1 - damped_sinusoid.omega)
                            if d_omega < min_d_omega:
                                QNM = ringdown.QNM((L1, M1, N1, S1))
                                min_d_omega = d_omega

                        # second order QNMs (could be improved w/ better spatial structure via, e.g., Sizheng's work)
                        if self.require_1st_order_QNM_existence:
                            if is_not_repeat:
                                continue

                        for L2, M2, N2, S2 in first_order_QNMs:
                            # wigner 3j rules
                            if (
                                not damped_sinusoid.target_mode[0] >= abs(L1 - L2)
                                and damped_sinusoid.target_mode[0] <= L1 + L2
                            ):
                                continue
                            if (
                                M1 + M2 != damped_sinusoid.target_mode[1]
                            ):  # (this isn't exactly correct)
                                continue
                            if (
                                damped_sinusoid.target_mode[1] == M1
                                and M1 == M2
                                and M2 == 0
                                and not (L1 + L2 + damped_sinusoid.target_mode[0]) % 2
                                == 0
                            ):
                                continue

                            omega2 = ringdown.omega_and_C(
                                (L2, M2, N2, S2), (L2, M2), self.M_f, self.chi_f
                            )[0]

                            # filter 2nd order QNM
                            # no repeats
                            if (
                                sorted([(L1, M1, N1, S1), (L2, M2, N2, S2)])
                                in second_order_QNMs
                            ):
                                continue

                            # lower overtones must exist
                            lower_overtones_exist = False
                            for second_order_QNM in second_order_QNMs:
                                L1_test, M1_test, N1_test, S1_test = second_order_QNM[0]
                                L2_test, M2_test, N2_test, S2_test = second_order_QNM[1]
                                if (
                                    L1_test == L1
                                    and M1_test == M1
                                    and S1_test == S1
                                    and L2_test == L2
                                    and M2_test == M2
                                    and S2_test == S2
                                ):
                                    total_d_N = (N1_test - N1) + (N2_test - N2)
                                    if total_d_N == 1:
                                        lower_overtones_exist = True
                            if (N1 != 0 or N2 != 0) and not lower_overtones_exist:
                                continue

                            d_omega = abs((omega1 + omega2) - damped_sinusoid.omega)
                            if d_omega < min_d_omega:
                                QNM = ringdown.QNM(
                                    sorted([(L1, M1, N1, S1), (L2, M2, N2, S2)]),
                                    damped_sinusoid.target_mode,
                                )
                                min_d_omega = d_omega

        if min_d_omega < self.frequency_tolerance:
            if self.verbose:
                print(
                    colored(
                        f"* {QNM.mode} matched to frequency, d_omega = {round(min_d_omega, 4)}\n",
                        "light_cyan",
                    )
                )
            return QNM
        else:
            if self.verbose:
                print(
                    colored(
                        f"* best fit QNM {QNM.mode} not matched to frequency, "
                        + f"d_omega = {round(min_d_omega, 4)} > {self.frequency_tolerance}\n",
                        "red",
                    )
                )
            return

    def find_best_fitting_QNM(self, mode, t_0=None, N_free_frequencies=1):
        """Find the frequency which best fits the mode.

        Parameters
        ----------
        mode : tuple
            (\ell, m) mode to be fit to.
        t_0 : float
            time beyond which to fit.
            [Default is self.t_i]
        N_free_frequencies : int
            how many free damped sinusoids to fit with varpro.

        Returns
        -------
        QNM_model : ringdown.QNMModel
            model of QNMs matching self.QNM_model, but with the QNM matching
            the best fit frequency, if such a match is found.
            Otherwise returns False.
        """
        if t_0 is None:
            t_0 = self.t_i

        QNM_model_filtered = self.QNM_model.filter_based_on_mode(mode)

        fit_QNM_model_filtered = QNM_model_filtered.fit(
            self.h_NR, [mode], t_0, self.t_f, N_free_frequencies=N_free_frequencies
        )
        if fit_QNM_model_filtered is None:
            return False

        QNMs = [
            self.match_damped_sinusoid_to_QNM(non_QNM_sinusoid)
            for non_QNM_sinusoid in fit_QNM_model_filtered.non_QNM_sinusoids
        ]

        if np.all([not QNM is None for QNM in QNMs]):
            if len(QNMs) == 2:
                if QNMs[0].mode == QNMs[1].mode:
                    return False

            QNM_model = ringdown.QNMModel(
                self.M_f, self.chi_f, self.QNM_model.QNMs + QNMs
            )
            return QNM_model
        else:
            return False

    def compute_largest_stable_window(self, QNM):
        """Find the largest stable window.

        Parameters
        ----------
        QNM : ringdown.QNM
            QNM under consideration.

        Returns
        -------
        largest_window : tuple
            window over which the QNM has a CV below self.CV_tolerance.
        """
        true_min_CV = np.inf
        largest_window = (0.0, 0.0)

        d_window_size = np.diff(self.t_0s)[0]
        min_t_0_window = self.min_t_0_window
        if min_t_0_window is None:
            min_t_0_window = (
                round(-self.min_t_0_window_factor / QNM.omega.imag / d_window_size)
                * d_window_size
            )
        for window_size in np.arange(
            min_t_0_window,
            (self.t_0s[-1] - self.t_0s[0]) + d_window_size,
            d_window_size,
        ):
            idx1 = 0
            min_CV = np.inf

            while self.t_0s[idx1] + window_size < self.t_0s[-1]:
                idx2 = np.argmin(abs(self.t_0s - (self.t_0s[idx1] + window_size))) + 1
                CV = np.std(QNM.A_time_series[idx1:idx2]) / np.mean(
                    abs(QNM.A_time_series[idx1:idx2])
                )
                if CV < min_CV:
                    min_CV = CV
                    best_idx1 = idx1
                    best_idx2 = idx2
                idx1 += 1

            if min_CV < self.CV_tolerance:
                true_min_CV = min_CV
                largest_window = (self.t_0s[best_idx1], self.t_0s[best_idx2])
                continue
            else:
                break

        if largest_window == (0.0, 0.0):
            return largest_window, min_CV
        else:
            return largest_window, true_min_CV

    def analyze_model_time_series(self, QNM_model):
        """Analyze time series data of model, i.e.,
           find the largest stable window for each QNM
           and extract the amplitude over said window.

        Parameters
        ----------
        QNM_model : ringdown.QNMModel
            model of QNMs analyze time series data for.

        Returns
        -------
        QNM_model : ringdown.QNMModel
            model of QNMs with analyzed time series data.
        """
        for QNM in QNM_model.QNMs:
            largest_stable_window, CV = self.compute_largest_stable_window(QNM)
            QNM.largest_stable_window = largest_stable_window
            QNM.CV = CV

            QNM.A = np.mean(
                QNM.A_time_series[
                    np.argmin(abs(self.t_0s - largest_stable_window[0])) : np.argmin(
                        abs(self.t_0s - largest_stable_window[1])
                    )
                    + 1
                ].real
            ) + 1j * np.mean(
                QNM.A_time_series[
                    np.argmin(abs(self.t_0s - largest_stable_window[0])) : np.argmin(
                        abs(self.t_0s - largest_stable_window[1])
                    )
                    + 1
                ].imag
            )
            QNM.A_std = np.std(
                QNM.A_time_series[
                    np.argmin(abs(self.t_0s - largest_stable_window[0])) : np.argmin(
                        abs(self.t_0s - largest_stable_window[1])
                    )
                    + 1
                ].real
            ) + 1j * np.std(
                QNM.A_time_series[
                    np.argmin(abs(self.t_0s - largest_stable_window[0])) : np.argmin(
                        abs(self.t_0s - largest_stable_window[1])
                    )
                    + 1
                ].imag
            )

        return QNM_model

    def is_model_stable(self, QNM_model=None):
        """Verify model stability.

        Parameters
        ----------
        QNM_model : ringdown.QNMModel
            model of QNMs to check stability for.
        """
        if QNM_model is None:
            QNM_model = self.QNM_model

        # fit over t_0s
        fit_QNM_model = QNM_model.fit(
            self.h_NR, modes=self.modes, t_i=self.t_0s, n_procs=self.n_procs
        )

        # find largest stable windows and measure amplitudes
        QNM_model = self.analyze_model_time_series(fit_QNM_model)

        QNM_stable_windows = [QNM.largest_stable_window for QNM in QNM_model.QNMs]

        # if first QNM, use basic stability check
        if len(QNM_stable_windows) == 1:
            if self.verbose:
                print(colored("** model passed stability test!", "green"))
            self.previous_QNM_stable_windows = QNM_stable_windows
            return (
                -self.min_t_0_window_factor / fit_QNM_model.QNMs[0].omega.imag
                <= np.diff(QNM_stable_windows[-1]),
                QNM_model,
            )

        # otherwise, ensure window increase and doesn't move later
        d_N_QNMs = len(QNM_stable_windows) - len(self.previous_QNM_stable_windows)

        if np.all(np.diff(QNM_stable_windows) > 0):
            if self.verbose:
                print(colored("** model passed stability test!", "green"))
            self.previous_QNM_stable_windows = QNM_stable_windows
            return True, QNM_model
        else:
            if self.verbose:
                print(colored("** model failed stability test:", "red"))
                print(
                    "* previous windows passed:",
                    np.all(np.diff(QNM_stable_windows)[-d_N_QNMs:] > 0),
                )
                for i in range(len(QNM_stable_windows)):
                    if not np.diff(QNM_stable_windows[i]) > 0:
                        print(
                            f"{QNM_model.QNMs[i].mode}: {QNM_stable_windows[i]} vs. {self.previous_QNM_stable_windows[i]}"
                        )
                print(
                    "* new window(s) passed:",
                    np.all(np.diff(QNM_stable_windows[-d_N_QNMs:]) > 0),
                    ";",
                    [window for window in QNM_stable_windows[-d_N_QNMs:]],
                )
                print(
                    "* new CV(s):",
                )
                for QNM in QNM_model.QNMs[-d_N_QNMs:]:
                    print(QNM.mode, "->", QNM.CV)
                print()

            return False, QNM_model

    def build_model(self):
        """Build QNM model."""
        # iterate over fitting start times
        t_0_idx = self.t_0s.size - 1
        self.latest_t_0_idx = t_0_idx

        print(
            colored(
                "***************\n" + "Building Model!\n" + "***************\n", "blue"
            )
        )
        while t_0_idx >= 0:
            t_0 = self.t_0s[t_0_idx]

            if self.verbose:
                print(colored(f"** working on t_0 = {t_0}.\n", "light_blue"))

            # find most unmodeled mode, by power
            mode_to_model, LMs_and_power = self.find_most_unmodeled_mode(t_0)

            if self.allow_more_than_one_free_frequency:
                if t_0_idx == 0 or (
                    not self.mode_to_model is None
                    and self.mode_to_model != mode_to_model
                ):
                    # change fit to include two free damped sinusoids
                    # since we've failed to fit self.mode_to_model
                    if self.N_free_frequencies == 1:
                        if self.verbose:
                            print(
                                colored(
                                    f"** unable to find QNM in {self.mode_to_model} with {self.N_free_frequencies} "
                                    + f"free damped sinusoid;\ntrying again with {self.N_free_frequencies + 1} "
                                    + "free damped sinusoids.\n",
                                    "light_yellow",
                                )
                            )
                        self.N_free_frequencies = 2

                        t_0_idx = self.latest_t_0_idx

                        continue
                    else:
                        self.N_free_frequencies = 1

                        self.latest_t_0_idx = t_0_idx

            if self.verbose:
                if mode_to_model != self.mode_to_model:
                    print(colored(f"** modeling {mode_to_model}", "light_magenta"))
                    print(colored("* top four power rankings :", "light_magenta"))
                    for i, LM_and_power in enumerate(LMs_and_power[:4]):
                        print(f"- {LM_and_power[0]} -> {LM_and_power[1]}")
                    print()

            self.mode_to_model = mode_to_model

            # find best fitting QNM to said model; if no match, continue
            QNM_model = self.find_best_fitting_QNM(
                self.mode_to_model, t_0, self.N_free_frequencies
            )
            if not QNM_model:
                t_0_idx -= 1
                continue

            if [
                QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies :]
            ] in self.failed_QNM_modes:
                if self.verbose:
                    print(
                        colored(
                            f"* QNM {[QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies:]]} "
                            + "already failed stability; continuing.\n",
                            "red",
                        )
                    )
                t_0_idx -= 1
                continue

            # check if new QNM model is stable
            is_stable, QNM_model = self.is_model_stable(QNM_model)
            if not is_stable:
                self.failed_QNM_modes.append(
                    [QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies :]]
                )
                t_0_idx -= 1
                continue

            print(colored(f"** new model is", "green"))
            for QNM in QNM_model.QNMs:
                print(colored(f"- {QNM.mode}", "green"))
            print()

            # update model
            self.update_model(QNM_model)

            t_0_idx = self.latest_t_0_idx

        del self.latest_t_0_idx

    def build_model_iteratively(self):
        """Build QNM model iteratively."""
        raise NotImplementedError("Not implemented yet!")
