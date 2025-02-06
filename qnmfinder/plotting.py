import numpy as np
import matplotlib.pyplot as plt

from . import ringdown


def plot_amplitudes_and_phases(QNM_model, plot_mirror_modes=False, plot_phases=True, vert_limits=None):
    """Plot time-dependent amplitudes and phases of a ringdown.QNMModel.

    Parameters
    ----------
    QNM_model : ringdown.QNMModel
        QNM_model whose amplitudes/phases will be plot.
    plot_mirror_modes : bool
        whether or not to plot the mirror modes.
        [Default: False]
    plot_phases : bool
        whether or not to plot the phases.
        [Default: True]
    vert_limits : tuple
        vertical axis limits.
        [Default: None]
    """
    if plot_phases:
        fig, axis = plt.subplots(1, 3, width_ratios=[1, 1, 0])
    else:
        fig, axis = plt.subplots(1, 2, width_ratios=[1, 0])
    plt.subplots_adjust(wspace=0.31)

    for QNM in QNM_model.QNMs:
        if not plot_mirror_modes:
            if QNM.target_mode[1] < 0:
                continue
        p = axis[0].plot(
            QNM_model.t_0s,
            abs(
                QNM.A_time_series
                * np.exp(-1j * QNM.omega * QNM_model.t_0s)
            ),
            lw=0.5,
        )
        idx1 = np.argmin(abs(QNM_model.t_0s - QNM.largest_stable_window[0]))
        idx2 = np.argmin(abs(QNM_model.t_0s - QNM.largest_stable_window[1])) + 1
        axis[0].plot(
            QNM_model.t_0s[idx1:idx2],
            abs(
                QNM.A_time_series
                * np.exp(-1j * QNM.omega * QNM_model.t_0s)
            )[idx1:idx2],
            lw=2,
            color=p[0].get_color(),
        )

        idx = 1
        if plot_phases:
            p = axis[1].plot(
                QNM_model.t_0s, np.angle(QNM.A_time_series), lw=0.5
            )
            idx1 = np.argmin(abs(QNM_model.t_0s - QNM.largest_stable_window[0]))
            idx2 = np.argmin(abs(QNM_model.t_0s - QNM.largest_stable_window[1])) + 1
            axis[1].plot(
                QNM_model.t_0s[idx1:idx2],
                np.angle(QNM.A_time_series)[idx1:idx2],
                lw=2,
                color=p[0].get_color(),
            )
            idx = 2

        axis[idx].plot([None], [None], lw=0.5, label=str(QNM.mode))

    axis[0].set_yscale("log")
    axis[0].set_ylim(vert_limits)
    axis[idx].legend(loc="upper left")

    axis[idx].spines["top"].set_visible(False)
    axis[idx].spines["right"].set_visible(False)
    axis[idx].spines["bottom"].set_visible(False)
    axis[idx].spines["left"].set_visible(False)
    axis[idx].get_xaxis().set_ticks([])
    axis[idx].get_yaxis().set_ticks([])

    axis[0].set_xlabel("fit start time $t_{0}$")
    axis[0].set_title("$A_{\mathrm{QNM}}(t=t_{0})$")
    if plot_phases:
        axis[1].set_xlabel("fit start time $t_{0}$")
        axis[1].set_title("$\phi_{\mathrm{QNM}}(t=t_{\mathrm{peak}})$")

    plt.show()


def plot_free_frequency_evolution(
    QNM_model,
    h_NR,
    t_0s,
    t_f,
    mode,
    N_free_frequencies=1,
    frequency_tolerance=1.0e-1,
    QNMs_to_plot=[],
    recycle_varpro_results_as_initial_guess=True,
    n_procs=-1,
):
    """Plot the free frequency evolution of a ringdown.QNMModel.

    Parameters
    ----------
    QNM_model: ringdown.QNMModel
        QNM_model that will be used in the fit to h_NR.
    h_NR : scri.WaveformModes
        NR waveform to fit to.
    t_0s : ndarray
        fitting start times.
    t_f : float
        latest time to fit.
    mode : tuple
        (\ell, m) mode to fit to.
    N_free_frequencies : int
        number of free frequencies <= 3 to fit.
        [Default: 1]
    frequency_tolerance : float
        modulus window to plot.
        [Default: 1.e-1]
    QNMs_to_plot : list of tuples
        list of (\ell, m, n, s) QNMs (or 2nd order QNMs) to plot.
        [Default: []]
    recycle_varpro_results_as_initial_guess : bool
        whether or not to use the varpro results for subsequent initial guesses.
        [Default: True]
    n_procs : int
        number of cores to use; if 'auto', optimal number based on the number of fit start times;
        if None, maximum number of cores; if -1, no multiprocessing is performed.
        [Default: -1]
    """
    if n_procs == "auto":
        if t_0s.size <= 20:
            n_procs = -1
        else:
            n_procs = None

    QNM_model.compute_omegas_and_Cs()

    fit_QNM_model = QNM_model.fit(
        h_NR, [mode], t_0s, t_f, N_free_frequencies=N_free_frequencies,
        recycle_varpro_results_as_initial_guess=recycle_varpro_results_as_initial_guess, n_procs=n_procs
    )

    fig, axis = plt.subplots(1, 1)

    for i, non_QNM_damped_sinusoid in enumerate(fit_QNM_model.non_QNM_damped_sinusoids):
        omegas = non_QNM_damped_sinusoid.omegas
        plot = axis.scatter(
            omegas.real,
            -omegas.imag,
            c=fit_QNM_model.t_0s,
            marker=f'${i}$'
        )
    c = plt.colorbar(plot)

    min_omega_re = np.inf
    max_omega_re = -np.inf
    min_omega_im = np.inf
    for QNM in QNMs_to_plot:
        omega = ringdown.omega_and_C(QNM, mode, QNM_model.M_f, QNM_model.chi_f)[0]
        if omega.real < min_omega_re:
            min_omega_re = omega.real
        if omega.real > max_omega_re:
            max_omega_re = omega.real
        if omega.imag < min_omega_im:
            min_omega_im = omega.imag
        axis.scatter(omega.real, -omega.imag, marker="x", color="k")
        circle = plt.Circle(
            (omega.real, -omega.imag), frequency_tolerance, color="k", fill=False
        )
        axis.add_patch(circle)

    if QNMs_to_plot != []:
        d_omega_re = max_omega_re - min_omega_re
        axis.set_xlim(min_omega_re - 0.2 * d_omega_re, max_omega_re + 0.2 * d_omega_re)
        axis.set_ylim(-0.1, -min_omega_im + 0.2 * d_omega_re)

    axis.set_xlabel(r"$\mathrm{Re}[\omega]$")
    axis.set_ylabel(r"$-\mathrm{Im}[\omega]$")
    c.set_label(r"$t_{0}$")

    plt.show()
