import numpy as np
import itertools

import sxs
import scri
from scri.asymptotic_bondi_data.map_to_superrest_frame import WM_to_MT, MT_to_WM
from quaternion.calculus import indefinite_integral as integrate

from . import utils
from . import ringdown
from . import plotting

import multiprocessing

from termcolor import colored


class QNMModelBuilder:
    """Build QNM model for an NR waveform.

    Attributes
    ----------
    h_NR : scri.WaveformModes
        NR waveform to fit to.
    M_total : float
        total Christodoulou mass of the system.
    M_f : float
        remnant Christodoulou mass.
    chi_f : float
        remnant dimensionless spin magnitude.
    t_i : float
        earliest time (relative to peak) to fit.
        [Default: 0.]
    t_f : float
        latest time (relative to peak) to fit.
        [Default: 100.]
    t_0_i : float
        first start time (relative to peak) to fit; if None, matches t_0_f.
        [Default: None]
    t_0_f : float
        latest start time (relative to peak) to fit.
        [Default: 80.]
    t_peak_norm_function : str
        function to use to compute the L^{2} norm; 'strain', 'news', or 'psi4'.
        [Default: 'news']
    d_t_0 : float
        spacing between t_i and t_0_f to iterate over.
        [Default: 1.]
    d_t_0_search : float
        step size to take backwards in fitting start times
        while performing reverse search.
        [Default: 1.]
    ell_min_NR : int
        minimum QNM \ell value to consider.
        [Default: 2]
    ell_max_NR : int
        maximum QNM \ell value to consider.
        [Default: 4]
    modes : list
        (\ell, m) modes of NR waveform to fit.
        [Default: use all modes.]
    fit_news : bool
        fit the news instead of the strain;
        returned amplitudes will still be for the strain.
        [Default: True]
    ell_min_QNM : int
        minimum QNM \ell value to consider.
        [Default: 2]
    ell_max_QNM : int
        maximum QNM \ell value to consider.
        [Default: 6]
    N_max : int
        maximum overtone value to consider.
        [Default: 4]
    try_mirrors : bool
        whether or not to try adding the mirror QNM for each QNM added.
        [Default: True]
    include_2nd_order_QNMs : bool
        whether or not to include 2nd order QNMs
        [Default: True]
    include_3rd_order_QNMs : bool
        whether or not to include 3rd order QNMs
        [Default: False]
    require_1st_order_QNM_existence : bool
        require 1st order QNM to exist in model to be considered
        as a component of a 2nd order QNM.
        [Default: True]
    exclude_zero_frequency_QNMs : bool
        whether or not to exclude zero frequency QNMs.
        [Default: True]
    t_ref : float
        reference time (relative to peak) for QNM amplitudes.
        [Default: 0.]
    N_free_frequencies_max : int
        maximum number of free frequencies to fit if one fails.
        [Default: 4]
    power_tolerance : float
        minimum unmodeled power needed to search for a QNM.
        [Default : 1.e-12]
    frequency_tolerance : float
        minimum modulus to match free frequency to QNM frequency.
        [Default: 2.e-1]
    CV_tolerance : float
        minimum coefficient of variation to QNM to be considered stable.
        [Default: 5.e-2]
    min_t_0_window : float
        minimum window over fitting start times to consider.
        [Default: -np.log(min_t_0_window_factor) / QNM.omega.imag.]
    min_t_0_window_factor : float
        factor by which to change the minimum stable window;
        this corresponds to the amount the amplitude should decay over the window.
        [Default: 10.0]
    min_A_tolerance : float
        minimum amplitude to consider during stability tests.
        [Default: 1.e-12.]
    reset_after_adding_QNM : bool
        whether or not to reset the fitting start time iteration
        after successfully adding a new QNM to the model.
        [Default: True]
    preexisting_model : ringdown.QNM_model
        QNM model to build on top of.
        [Default: None]
    n_procs : int
        number of cores to use; if 'auto', optimal number based on the number of fit start times;
        if None, maximum number of cores; if -1, no multiprocessing is performed.
        [Default: 'auto']
    verbose : bool
        whether or not to print status updates.
        [Default: False]
    plot_status : bool
        whether or not to plot the model status for animation.
        [Default: False]
    """

    def __init__(
        self,
        h_NR,
        M_total,
        M_f,
        chi_f,
        t_i=0,
        t_f=100,
        t_0_i=None,
        t_0_f=80,
        t_peak_norm_function='news',
        d_t_0=1.0,
        d_t_0_search=1.0,
        ell_min_NR=2,
        ell_max_NR=4,
        modes=None,
        fit_news=True,
        ell_min_QNM=2,
        ell_max_QNM=6,
        N_max=4,
        try_mirrors=True,
        include_2nd_order_QNMs=True,
        include_3rd_order_QNMs=False,
        require_1st_order_QNM_existence=True,
        exclude_zero_frequency_QNMs=True,
        t_ref=0.0,
        N_free_frequencies_max=4,
        power_tolerance=1.0e-12,
        frequency_tolerance=2.0e-1,
        CV_tolerance=5.0e-2,
        min_t_0_window=None,
        min_t_0_window_factor=10.0,
        min_A_tolerance=1e-12,
        reset_after_adding_QNM=True,
        preexisting_model=None,
        n_procs="auto",
        verbose=False,
        plot_status=False
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
        self.M_total = M_total
        self.M_f = M_f
        self.chi_f = chi_f
        self.t_i = t_i
        self.t_f = t_f
        self.t_peak_norm_function = t_peak_norm_function
        if t_0_i is None:
            t_0_i = t_0_f
        self.t_0_i = t_0_i
        self.t_0_f = t_0_f
        if d_t_0 < min(np.diff(self.h_NR.t)):
            print(
                colored(
                    "********\n"
                    + f"Warning: d_t_0 = {d_t_0} < h_NR's min dt = {min(np.diff(self.h_NR.t))}\n"
                    + "********",
                    "red",
                )
            )
        self.d_t_0 = d_t_0
        if d_t_0_search < min(np.diff(self.h_NR.t)):
            print(
                colored(
                    "********\n"
                    + f"Warning: d_t_0_search = {d_t_0_search} < h_NR's min dt = {min(np.diff(self.h_NR.t))}\n"
                    + "********",
                    "red",
                )
            )
        self.d_t_0_search = d_t_0_search
        self.ell_min_NR = ell_min_NR
        self.ell_max_NR = ell_max_NR
        self.modes = modes
        self.fit_news = fit_news
        self.ell_min_QNM = ell_min_QNM
        self.ell_max_QNM = ell_max_QNM
        self.N_max = N_max
        self.try_mirrors = try_mirrors
        self.include_2nd_order_QNMs = include_2nd_order_QNMs
        if include_3rd_order_QNMs:
            print(
                colored(
                    "********\n"
                    + f"Warning: 3rd order QNM search not yet optimized and could produce nonsensical results\n"
                    + "********",
                    "red",
                )
            )
        self.include_3rd_order_QNMs = include_3rd_order_QNMs
        self.require_1st_order_QNM_existence = require_1st_order_QNM_existence
        self.exclude_zero_frequency_QNMs = exclude_zero_frequency_QNMs
        self.t_ref = t_ref
        self.N_free_frequencies_max = N_free_frequencies_max
        self.power_tolerance = power_tolerance
        self.frequency_tolerance = frequency_tolerance
        self.CV_tolerance = CV_tolerance
        self.min_t_0_window = min_t_0_window
        self.min_t_0_window_factor = min_t_0_window_factor
        self.min_A_tolerance = min_A_tolerance
        self.reset_after_adding_QNM = reset_after_adding_QNM
        self.n_procs = n_procs
        self.verbose = verbose
        self.plot_status = plot_status

        # fix NR waveform so t_peak = time of peak
        self.h_NR.t *= self.M_total
        self.h_NR.t -= self.compute_t_peak(self.t_peak_norm_function)
        self.h_NR_fit = self.h_NR.copy()[
            np.argmin(abs(self.h_NR.t - self.t_i)) : np.argmin(
                abs(self.h_NR.t - self.t_f)
            )
            + 1,
            self.ell_min_NR : self.ell_max_NR + 1,
        ]

        if self.fit_news:
            h_NR_news = self.h_NR_fit.copy()
            h_NR_news.data = h_NR_news.data_dot
            h_NR_news.dataType = scri.hdot
            self.h_NR_fit = h_NR_news

        # construct t0s
        self.t_0s = np.arange(self.t_i, self.t_0_f + self.d_t_0, self.d_t_0)

        if self.n_procs == "auto":
            if self.t_0s.size <= 20:
                self.n_procs = -1
            else:
                self.n_procs = None

        # initialize QNM model
        if preexisting_model is None:
            self.update_model(ringdown.QNMModel(self.M_f, self.chi_f))
        else:
            is_stable, QNM_model = self.is_model_stable(
                preexisting_model, len(preexisting_model.QNMs)
            )
            if not is_stable:
                raise ValueError("Preexisting model is not stable!")
            else:
                self.update_model(QNM_model)

    def compute_t_peak(self, norm_function_name='news', mode=None):
        """Compute the time of peak of some norm function.

        Parameters
        ----------
        norm_function_name : str
            function to use to compute the L^{2} norm; 'strain', 'news', or 'psi4'.
            [Default: 'news']
        mode : (tuple)
            mode to compute the time of peak from instead of the L^{2} norm over the sphere;
            if None, then the L^{2} norm over the sphere is used.
            [Default: None]

        Returns
        -------
        t_peak : float
            time of peak.
        """
        norm_function = self.h_NR.copy()
        if norm_function_name == 'news':
            norm_function.data = norm_function.data_dot
        elif norm_function_name == 'psi4':
            norm_function.data = norm_function.data_ddot

        if mode is None:
            norm = norm_function.norm()
        else:
            norm = abs(norm_function.data[:,norm_function.index(*mode)])
        
        index = np.argmax(norm)
        index = max(2, min(len(self.h_NR.t) - 3, index))

        # Fit 5 points with a 2nd-order polynomial
        test_times = self.h_NR.t[index - 2 : index + 2 + 1] - self.h_NR.t[index]
        test_funcs = norm[index - 2 : index + 2 + 1]

        x_vecs = np.array([np.ones(5), test_times, test_times**2.0])
        y_vec = np.array([test_funcs.dot(v1) for v1 in x_vecs])
        inv_mat = np.linalg.inv(
            np.array([[v1.dot(v2) for v1 in x_vecs] for v2 in x_vecs])
        )
        coeffs = np.array([y_vec.dot(v1) for v1 in inv_mat])

        t_peak = self.h_NR.t[index] - coeffs[1] / (2.0 * coeffs[2])

        return t_peak

    def update_model(self, QNM_model):
        """Update self.QNM_model to reflect the content of QNM_model.

        Parameters
        ----------
        QNM_model : ringdown.QNMModel
            model containing current set of QNMs.
        """
        self.QNM_model = QNM_model

        if hasattr(self, 'failed_QNM_modes'):
            failed_QNM_modes_filtered = []
            if not self.mode_to_model is None:
                for QNM_mode in self.failed_QNM_modes:
                    if type(QNM_mode[0]) != list:
                        target_m = QNM_mode[0][1]
                    else:
                        target_m = np.sum([mode[1] for mode in QNM_mode[0]])
                        
                    if target_m == self.mode_to_model[1]:
                        continue
                    
                    failed_QNM_modes_filtered.append(QNM_mode)

            self.failed_QNM_modes = failed_QNM_modes_filtered
        else:
            self.failed_QNM_modes = []
                
        self.mode_to_model = None

        self.N_free_frequencies = 1

        self.compute_model_waveform()

    def compute_model_waveform(self):
        """Update self.h_QNM_fit and self.h_residual_fit to reflect the content of self.QNM_model."""
        self.h_QNM_fit = self.QNM_model.compute_waveform(self.h_NR_fit)

        self.h_residual_fit = self.h_NR_fit.copy()
        self.h_residual_fit.data -= self.h_QNM_fit.data

    def compute_model_errors(self, compute_mismatch=False, compute_strain_errors=False):
        """Update self.L2_norms to reflect the content of self.residual.
        
        Parameters
        ----------
        compute_mismatch : bool
            whether or not to update self.mismatches.
            [Default: False]
        compute_strain_errors : bool
            whether or not to comptue strain errors if self.fit_news = True.
            [Default: False]
        """
        L2_norms = []
        mismatches = []
        for t_0 in self.t_0s:
            L2_norm = utils.compute_L2_norm(
                self.h_NR_fit, self.h_QNM_fit, t_i=t_0, modes=self.modes
            )
            L2_norms.append(L2_norm)

            if compute_mismatch:
                mismatch = utils.compute_mismatch(
                    self.h_NR_fit, self.h_QNM_fit, t_i=t_0, modes=self.modes
                )
                mismatches.append(mismatch)

        self.L2_norms = L2_norms
        if compute_mismatch:
            self.mismatches = mismatches
            
        if self.fit_news and compute_strain_errors:
            L2_norms = []
            mismatches = []

            QNM_model = self.QNM_model.integrate()
            h_QNM = QNM_model.compute_waveform(self.h_NR)

            for t_0 in self.t_0s:
                L2_norm = utils.compute_L2_norm(
                    self.h_NR, h_QNM, t_i=t_0, modes=self.modes
                )
                L2_norms.append(L2_norm)

                if compute_mismatch:
                    mismatch = utils.compute_mismatch(
                        self.h_NR, h_QNM, t_i=t_0, modes=self.modes
                    )
                    mismatches.append(mismatch)

            self.L2_norms_strain = L2_norms
            if compute_mismatch:
                self.mismatches_strain = mismatches

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

        idx_t_0 = np.argmin(abs(self.h_NR_fit.t - t_0))

        if not self.modes is None:
            mode_indices = []
            for L, M in self.h_NR_fit.LM:
                if (L, M) in self.modes:
                    mode_indices.append(True)
                else:
                    mode_indices.append(False)
        else:
            mode_indices = [True] * self.h_NR_fit.LM.shape[0]
        mode_indices = np.array(mode_indices)

        unmodeled_power = integrate(
            abs(self.h_residual_fit.data[idx_t_0:, mode_indices]) ** 2,
            self.h_residual_fit.t[idx_t_0:],
            axis=0,
        )[-1]/(self.h_residual_fit.t[idx_t_0:][-1] - self.h_residual_fit.t[idx_t_0:][0])
        
        LMs_ranked = [
            x
            for _, x in sorted(
                zip(unmodeled_power, self.h_residual_fit.LM[mode_indices]),
                key=lambda pair: pair[0]    
            )
        ][::-1]

        LMs_and_power = [
            (LM, sorted(unmodeled_power)[::-1][i]) for i, LM in enumerate(LMs_ranked)
        ]

        unmodeled_mode = tuple(LMs_ranked[0])

        return unmodeled_mode, LMs_and_power

    def verify_set_of_QNMs(self, QNMs):
        """Verify that a set of QNMs (created via match_damped_sinusoid_to_QNM)
        respects the requirement that overtones are added sequentially.

        Parameters
        ----------
        QNMs : list of ringdown.QNM
            list of QNMs output by match_damped_sinusoid_to_QNM.

        Returns
        -------
        is_valid : bool
            whether or not the set of QNMs respects the sequential overtone requirement.
        """
        is_valid = True

        for QNM1 in QNMs:
            if QNM1.is_first_order_QNM:
                L1, M1, N1, S1 = QNM1.mode

                overtone_values = []
                for QNM2 in [
                        QNM for QNM in self.QNM_model.QNMs + QNMs if QNM.is_first_order_QNM
                        and QNM.mode[0] == L1 and QNM.mode[1] == M1 and QNM.mode[3] == S1
                ]:
                    overtone_values.append(QNM2.mode[2])
                max_overtone_value = max(overtone_values)
                
                for N in range(max_overtone_value):
                    if not N in overtone_values:
                        is_valid = False
            else:
                overtone_tuples = []
                for QNM2 in [QNM for QNM in self.QNM_model.QNMs + QNMs if not QNM.is_first_order_QNM and len(QNM.mode) == len(QNM1.mode)]:
                    if not [(QNM[0], QNM[1], QNM[3]) for QNM in QNM2.mode] == [(QNM[0], QNM[1], QNM[3]) for QNM in QNM1.mode]:
                        continue
                    overtone_tuple = []
                    for QNM in QNM2.mode:
                        overtone_tuple.append(QNM[2])
                    overtone_tuples.append(tuple(overtone_tuple))
                    
                for overtone_tuple in overtone_tuples:
                    if not all([(x in overtone_tuples) for x in list(itertools.product(*[range(i + 1) for i in overtone_tuple]))]):
                        is_valid = False

        return is_valid
        
        
    def match_damped_sinusoid_to_QNM(self, damped_sinusoid, N_damped_sinusoids=1):
        """Match a damped sinusoid to a QNM.

        Parameters
        ----------
        damped_sinusoid : ringdown.DampedSinusoid
            damped sinusoid to match to a QNM.
        N_damped_sinusoids : int
            number of damped sinusoids being matched to QNMs.

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
            (QNM.mode, QNM.target_mode) for QNM in self.QNM_model.QNMs if not QNM.is_first_order_QNM and len(QNM.mode) == 2
        ]
        third_order_QNMs = [
            (QNM.mode, QNM.target_mode) for QNM in self.QNM_model.QNMs if not QNM.is_first_order_QNM and len(QNM.mode) == 3
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
                            max_N_in_first_order_QNMs = len([N for L, M, N, S in first_order_QNMs if L == L1 and M == M1 and S == S1]) - 1
                            lower_overtones_exist = (N1 - N_damped_sinusoids) <= max_N_in_first_order_QNMs

                        if is_not_repeat and m_value_matches and lower_overtones_exist:
                            d_omega = abs(omega1 - damped_sinusoid.omega)
                            if d_omega < min_d_omega:
                                QNM = ringdown.QNM((L1, M1, N1, S1))
                                min_d_omega = d_omega

                        if not self.include_2nd_order_QNMs:
                            continue

                        if self.require_1st_order_QNM_existence:
                            if is_not_repeat:
                                continue

                        for L2, M2, N2, S2 in first_order_QNMs:
                            # wigner 3j rules;
                            # note that the restriction \ell values
                            # is not applied, since we expect 2nd order QNMs to mix
                            # to higher and lower \ell modes
                            passes_wigner_3j_rules = True
                            if (
                                M1 + M2 != damped_sinusoid.target_mode[1]
                            ): 
                                passes_wigner_3j_rules = False

                            omega2 = ringdown.omega_and_C(
                                (L2, M2, N2, S2), (L2, M2), self.M_f, self.chi_f
                            )[0]

                            # no repeats
                            is_not_repeat = True
                            for second_order_QNM, target_mode in second_order_QNMs:
                                if (
                                        sorted([(L1, M1, N1, S1), (L2, M2, N2, S2)]) == second_order_QNM
                                        and damped_sinusoid.target_mode == target_mode
                                ):
                                    is_not_repeat = False

                            # lower overtones must exist
                            lower_overtones_exist = False
                            for second_order_QNM, target_mode in second_order_QNMs:
                                L1_test, M1_test, N1_test, S1_test = second_order_QNM[0]
                                L2_test, M2_test, N2_test, S2_test = second_order_QNM[1]
                                if (
                                    L1_test == L1
                                    and M1_test == M1
                                    and S1_test == S1
                                    and L2_test == L2
                                    and M2_test == M2
                                    and S2_test == S2
                                    and target_mode == damped_sinusoid.target_mode
                                ):
                                    if (N1 - N1_test) + (N2 - N2_test) <= N_damped_sinusoids:
                                        lower_overtones_exist = True

                            lower_overtones_exist = (N1 == 0 and N2 == 0) or lower_overtones_exist

                            is_zero_frequency = (omega1 + omega2).real == 0

                            if passes_wigner_3j_rules and is_not_repeat and lower_overtones_exist and not (self.exclude_zero_frequency_QNMs and is_zero_frequency):
                                d_omega = abs((omega1 + omega2) - damped_sinusoid.omega)
                                if d_omega < min_d_omega:
                                    QNM = ringdown.QNM(
                                        sorted([(L1, M1, N1, S1), (L2, M2, N2, S2)]),
                                        damped_sinusoid.target_mode,
                                    )
                                    min_d_omega = d_omega

                            if not self.include_3rd_order_QNMs:
                                continue

                            for L3, M3, N3, S3 in first_order_QNMs:
                                # wigner 3j rules (with some ignored, see above)
                                passes_wigner_3j_rules = True
                                if (
                                        M1 + M2 + M3 != damped_sinusoid.target_mode[1]
                                ): 
                                    passes_wigner_3j_rules = False

                                omega3 = ringdown.omega_and_C(
                                    (L3, M3, N3, S3), (L3, M3), self.M_f, self.chi_f
                                )[0]
                                
                                # no repeats
                                is_not_repeat = True
                                for third_order_QNM, target_mode in third_order_QNMs:
                                    if (
                                            sorted([(L1, M1, N1, S1), (L2, M2, N2, S2), (L3, M3, N3, S3)]) == third_order_QNM
                                            and damped_sinusoid.target_mode == target_mode
                                    ):
                                        is_not_repeat = False
                                
                                # lower overtones must exist
                                lower_overtones_exist = False
                                for third_order_QNM, target_mode in third_order_QNMs:
                                    L1_test, M1_test, N1_test, S1_test = third_order_QNM[0]
                                    L2_test, M2_test, N2_test, S2_test = third_order_QNM[1]
                                    L3_test, M3_test, N3_test, S3_test = third_order_QNM[2]
                                    if (
                                            L1_test == L1
                                            and M1_test == M1
                                            and S1_test == S1
                                            and L2_test == L2
                                            and M2_test == M2
                                            and S2_test == S2
                                            and L3_test == L3
                                            and M3_test == M3
                                            and S3_test == S3
                                    ):
                                        total_d_N = (N1_test - N1) + (N2_test - N2) + (N3_test - N3)
                                        if total_d_N <= N_damped_sinusoids:
                                            lower_overtones_exist = True

                                lower_overtones_exist = (N1 == 0 and N2 == 0 and N3 == 0) or lower_overtones_exist
                                
                                if passes_wigner_3j_rules and is_not_repeat and lower_overtones_exist:
                                    d_omega = abs((omega1 + omega2 + omega3) - damped_sinusoid.omega)
                                    if d_omega < min_d_omega:
                                        QNM = ringdown.QNM(
                                            sorted([(L1, M1, N1, S1), (L2, M2, N2, S2), (L3, M3, N3, S3)]),
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
            self.h_NR_fit,
            [mode],
            t_0,
            self.t_f,
            self.t_ref,
            N_free_frequencies=N_free_frequencies,
            n_procs=self.n_procs
        )
        if fit_QNM_model_filtered is None:
            return False

        QNMs = [
            self.match_damped_sinusoid_to_QNM(non_QNM_damped_sinusoid, N_free_frequencies)
            for non_QNM_damped_sinusoid in fit_QNM_model_filtered.non_QNM_damped_sinusoids
        ]

        if np.all([not QNM is None for QNM in QNMs]):
            if len(QNMs) > 1:
                # QNMs can't appear twice
                QNM_modes = [QNM.mode for QNM in QNMs]
                for QNM_mode in QNM_modes:
                    if QNM_modes.count(QNM_mode) > 1:
                        return False

                # QNMs must have sequential overtones
                if not self.verify_set_of_QNMs(QNMs):
                    return False

            QNM_model = ringdown.QNMModel(
                self.M_f, self.chi_f, self.QNM_model.QNMs + QNMs
            )
            return QNM_model
        else:
            return False

    def is_model_stable(self, QNM_model, d_N_QNMs):
        """Verify model stability.

        Parameters
        ----------
        QNM_model : ringdown.QNMModel
            model of QNMs to check stability for.

        d_N_QNMs : bool
            number of new QNMs being checked.
        """
        if QNM_model is None:
            QNM_model = self.QNM_model

        # fit over t_0s
        fit_QNM_model = QNM_model.fit(
            self.h_NR_fit,
            modes=self.modes,
            t_i=self.t_0s,
            t_f=self.t_f,
            t_ref=self.t_ref,
            n_procs=self.n_procs,
        )

        # find stable windows and measure amplitudes
        QNM_model = fit_QNM_model.analyze_model_time_series(
            self.CV_tolerance, self.min_t_0_window, self.min_t_0_window_factor, self.min_A_tolerance
        )

        QNM_stable_windows = [QNM.stable_window for QNM in QNM_model.QNMs]

        if np.all(np.diff(QNM_stable_windows) > 0):
            if self.verbose:
                print(colored("** model passed stability test!", "green"))
                print(
                    "* new window(s):",
                )
                for QNM in QNM_model.QNMs[-d_N_QNMs:]:
                    print(QNM.mode, "->", QNM.stable_window)
                print(
                    "* new CV(s):",
                )
                for QNM in QNM_model.QNMs[-d_N_QNMs:]:
                    print(QNM.mode, "->", QNM.CV)
            self.previous_QNM_stable_windows = QNM_stable_windows
            return True, QNM_model
        else:
            if self.verbose:
                print(colored("** model failed stability test:", "red"))
                try:
                    print(
                        "* previous windows passed:",
                        np.all(np.diff(QNM_stable_windows)[:-d_N_QNMs] > 0),
                    )
                    for i in range(len(QNM_stable_windows[:-d_N_QNMs])):
                        if not np.diff(QNM_stable_windows[i]) > 0:
                            print(
                                f"{QNM_model.QNMs[i].mode}: {QNM_stable_windows[i]} vs. {self.previous_QNM_stable_windows[i]}, CV = {QNM_model.QNMs[i].CV}"
                            )
                except:
                    pass

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
        t_0 = self.t_0_i
        self.latest_t_0s = [self.t_0_i]
        self.latest_t_0s_ms = [None]
        self.plot_count = 0

        print(
            colored(
                "***************\n"
                + "Building Model!\n"
                + "***************\n",
                "blue"
            )
        )
        if self.verbose:
            print(
                colored(
                    f"* t_i = {self.t_i}\n"
                    + f"* t_f = {self.t_f}\n"
                    + f"* t_0_f = {self.t_0_f}\n"
                    + f"* t_peak_norm_function = {self.t_peak_norm_function}\n"
                    + f"* d_t_0 = {self.d_t_0}\n"
                    + f"* d_t_0_search = {self.d_t_0_search}\n"
                    + f"* fit_news = {self.fit_news}\n"
                    + f"* include_2nd_order_QNMs = {self.include_2nd_order_QNMs}\n"
                    + f"* include_3rd_order_QNMs = {self.include_3rd_order_QNMs}\n"
                    + f"* N_free_frequencies_max = {self.N_free_frequencies_max}\n"
                    + f"* power_tolerance = {self.power_tolerance}\n"
                    + f"* CV_tolerance = {self.CV_tolerance}\n"
                    + f"* min_t_0_window_factor = {self.min_t_0_window_factor}\n"
                    + f"* min_A_tolerance = {self.min_A_tolerance}\n",
                    "light_blue"
                )
            )
            
        while t_0 >= self.t_i:
            if self.verbose:
                print(colored(f"** working on t_0 = {t_0}.\n", "light_blue"))

            # find most unmodeled mode, by power
            mode_to_model, LMs_and_power = self.find_most_unmodeled_mode(t_0)
            
            if LMs_and_power[0][1] < self.power_tolerance:
                if self.verbose:
                    print(
                        colored(
                            f"** unmodeled power in {mode_to_model} = {LMs_and_power[0][1]} < power tolerance = {self.power_tolerance};\n"
                            + "continuing search at the next earliest time.\n",
                            "light_green"
                        )
                    )
                t_0 -= self.d_t_0_search
                continue

            if self.N_free_frequencies_max > 1:
                if abs(t_0 - self.t_i) < self.d_t_0_search or (
                    not self.mode_to_model is None
                    and self.mode_to_model != mode_to_model
                ):
                    # change fit to include two free damped sinusoids
                    # since we've failed to fit self.mode_to_model
                    if self.N_free_frequencies < self.N_free_frequencies_max:
                        if self.verbose:
                            print(
                                colored(
                                    f"** unable to find QNM in {self.mode_to_model} with {self.N_free_frequencies} "
                                    + f"free damped sinusoid;\ntrying again with {self.N_free_frequencies + 1} "
                                    + "free damped sinusoids.\n",
                                    "light_yellow",
                                )
                            )
                        self.N_free_frequencies += 1

                        t_0 = self.latest_t_0s[-1]

                        continue
                    else:
                        self.N_free_frequencies = 1

                        if self.mode_to_model is not None:
                            self.latest_t_0s.append(t_0)
                            
                            self.latest_t_0s_ms.append(self.mode_to_model[1])

            if self.verbose:
                if mode_to_model != self.mode_to_model:
                    print(colored(f"** modeling {mode_to_model}", "light_magenta"))
                    print(colored("* top four power rankings :", "light_magenta"))
                    for i, LM_and_power in enumerate(LMs_and_power[:4]):
                        print(f"- {LM_and_power[0]} -> {LM_and_power[1]}")
                    print()

            self.mode_to_model = mode_to_model

            if self.plot_status:
                N_count = 1
                if self.plot_count > 0 and t_0 == self.latest_t_0s[-1] and self.N_free_frequencies == 1:
                    N_count = 5
                for i in range(N_count):
                    plotting.plot_status(self, t_0, self.mode_to_model, self.N_free_frequencies, self.plot_count)
                    self.plot_count += 1

            # find best fitting QNM to said model; if no match, continue
            QNM_model = self.find_best_fitting_QNM(
                self.mode_to_model, t_0, self.N_free_frequencies
            )
            if not QNM_model:
                t_0 -= self.d_t_0_search
                continue

            if [
                QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies :]
            ] in self.failed_QNM_modes:
                if self.verbose:
                    print(
                        colored(
                            f"* QNM(s) {[QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies:]]} "
                            + "already failed stability; continuing.\n",
                            "red",
                        )
                    )
                t_0 -= self.d_t_0_search
                continue

            # check if new QNM model is stable
            is_stable, QNM_model = self.is_model_stable(
                QNM_model, self.N_free_frequencies
            )
            if not is_stable:
                self.failed_QNM_modes.append(
                    [QNM.mode for QNM in QNM_model.QNMs[-self.N_free_frequencies :]]
                )
                t_0 -= self.d_t_0_search
                continue

            print(colored(f"** new model is", "green"))
            for QNM in QNM_model.QNMs:
                print(colored(f"- {QNM.mode}", "green"))
            print()

            N_new_QNMs = self.N_free_frequencies

            # update model
            self.update_model(QNM_model)

            # update latest_t_0s
            try:
                last_index_of_mode_to_model = [idx for idx, mode in enumerate(self.latest_t_0s_ms) if m == mode_to_model[1]][-1]
                self.latest_t_0s = self.latest_t_0s[:last_index_of_mode_to_model]
                self.latest_t_0s_ms = self.latest_t_0s_ms[:last_index_of_mode_to_model]
            except:
                pass
            
            # try mirror QNM(s)
            if self.try_mirrors:
                QNM_model = QNM_model.copy()

                mirrors_to_examine = False
                for QNM in QNM_model.QNMs[-N_new_QNMs:]:
                    QNM_mirror = QNM.mirror()
                    if QNM_mirror.mode in [QNM.mode for QNM in QNM_model.QNMs]:
                        continue

                    QNM_model.QNMs.append(QNM_mirror)

                    mirrors_to_examine = True

                if mirrors_to_examine:
                    if self.verbose:
                        print(colored(f"* examining mirror mode(s):", "light_cyan"))
                        for QNM in QNM_model.QNMs[-N_new_QNMs:]:
                            print("-", QNM.mode)

                    # check if new QNM model is stable
                    is_stable, QNM_model = self.is_model_stable(QNM_model, N_new_QNMs)
                    if not is_stable:
                        self.failed_QNM_modes.append(
                            [QNM.mode for QNM in QNM_model.QNMs[-N_new_QNMs:]]
                        )
                        t_0 -= self.d_t_0_search
                        continue

                    print(colored(f"** new model is", "green"))
                    for QNM in QNM_model.QNMs:
                        print(colored(f"- {QNM.mode}", "green"))
                    print()

                    # update model
                    self.update_model(QNM_model)

                    # update latest_t_0s
                    try:
                        last_index_of_mode_to_model = [idx for idx, m in enumerate(self.latest_t_0s_ms) if m == -mode_to_model[1]][-1]
                        self.latest_t_0s = self.latest_t_0s[:last_index_of_mode_to_model]
                        self.latest_t_0s_ms = self.latest_t_0s_ms[:last_index_of_mode_to_model]
                    except:
                        pass
                    
            t_0 = self.latest_t_0s[-1]

        del self.latest_t_0s
        del self.latest_t_0s_ms

    def build_model_iteratively(self):
        """Build QNM model iteratively."""
        raise NotImplementedError("Not implemented yet!")
