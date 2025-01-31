import warnings

warnings.filterwarnings(
    "ignore", message=".*RuntimeWarning: invalid value encountered.*"
)

import numpy as np

import sxs
import scri
from scri.asymptotic_bondi_data.map_to_superrest_frame import MT_to_WM, WM_to_MT
from quaternion.calculus import indefinite_integral as integrate


def lists_are_equal(list_1, list_2):
    if (len(list_1) == len(list_2)) and (all(i in list_1 for i in list_2)):
        return True
    else:
        return False


def compute_L2_norm(
    h_1, h_2, t_i=None, t_f=None, modes=None, modes_for_norm=None, absolute_error=False
):
    """Compute the L2 norm between two waveforms over the
    time window (t_i, t_f) using the modes specified by `modes`.

    Parameters
    ----------
    h_1 : scri.WaveformModes
    h_2 : scri.sWaveformModes
    t_i : float
        beginning of L2 norm integral.
        [Default: beginning of waveform.]
    t_f : float
        end of L2 norm integral.
        [Default: end of waveform.]
    modes : list, optional
        modes (\ell, m) to include in numerator of L2 norm calculation.
        [Default: use all modes.]
    modes_for_norm : list, optional
        Modes (\ell, m) to include in denominator of L2 norm calculation.
        [Default: use all modes.]
    absolute_error : bool, optional
        whether or not to return the error and norm separately.
        [Default: False]

    Returns
    -------
    L2_norm: float
        L2 norm between to the waveforms;
        this is sqrt( ||h_1 - h_2||² / ||h_1||² ).
    """
    h_1 = h_1.copy()
    h_2 = h_2.copy()

    if t_i is None:
        t_i = h_1.t[0]

    if t_f is None:
        t_f = h_1.t[-1]

    h_1 = h_1[np.argmin(abs(h_1.t - t_i)) : np.argmin(abs(h_1.t - t_f)) + 1]
    h_2 = h_2[np.argmin(abs(h_2.t - t_i)) : np.argmin(abs(h_2.t - t_f)) + 1]

    h_diff = h_1.copy()
    h_diff = MT_to_WM(WM_to_MT(h_diff))
    h_diff.data -= h_2.data

    h_for_norm = h_1.copy()

    # Eliminate unwanted modes
    if modes is not None or modes_for_norm is not None:
        ell_min = min(h_1.ell_min, h_2.ell_min)
        ell_max = max(h_1.ell_max, h_2.ell_max)
        for L in range(ell_min, ell_max + 1):
            for M in range(-L, L + 1):
                if modes is not None:
                    if not (L, M) in modes:
                        h_diff.data[:, h_diff.index(L, M)] *= 0
                if modes_for_norm is not None:
                    if not (L, M) in modes_for_norm:
                        h_for_norm.data[:, h_for_norm.index(L, M)] *= 0

    if not absolute_error:
        L2_norm = (
            integrate(h_diff.norm(), h_diff.t)[-1]
            / integrate(h_for_norm.norm(), h_for_norm.t)[-1]
        )

        return L2_norm
    else:
        L2_norm = integrate(h_diff.norm(), h_diff.t)[-1]
        norm = integrate(h_for_norm.norm(), h_for_norm.t)[-1]

        return L2_norm, norm


def overlap(h_1_mts, h_2_mts, t_i, t_f):
    """Compute the overlap between two waveforms over the
    time window (t_i, t_f) using the modes specified by `modes`.

    Parameters
    ----------
    h_1_mts : scri.ModesTimeSeries
    h_2_mts : scri.sModesTimeSeries
    t_i : float
        beginning of overlap integral.
        [Default: beginning of waveform.]
    t_f : float
        end of overlap integral.
        [Default: end of waveform.]

    Returns
    -------
    overlap : float
        overlap between the two waveforms.
    """
    idx_i = np.argmin(abs(h_1_mts.t - t_i))
    idx_f = np.argmin(abs(h_1_mts.t - t_f)) + 1
    # Include factor of sqrt(4\pi) to account for unwanted SWSH integral factor
    overlap = integrate(
        np.sqrt(4 * np.pi)
        * h_1_mts.multiply(h_2_mts.bar, truncator=lambda tup: 0).ndarray[
            idx_i:idx_f, 0
        ],
        h_1_mts.t[idx_i:idx_f],
    )[-1]

    return overlap


def compute_mismatch(h_1, h_2, t_i=None, t_f=None, modes=None):
    """Compute the mismatch between two waveforms over the
    time window (t_i, t_f) using the modes specified by `modes`.

    Parameters
    ----------
    h_1 : scri.sWaveformModes
    h_2 : scri.WaveformModes
    t_i : float
        beginning of L2 norm integral.
        [Default: beginning of waveform.]
    t_f : float
        end of L2 norm integral.
        [Default: end of waveform.]
    modes : list, optional
        modes (\ell, m) to include in mismatch calculation.
        [Default: use all modes.]

    Returns
    -------
    mismatch : float
        mismatch between the two waveforms.
    """
    h_1 = h_1.copy()
    h_2 = h_2.copy()

    if t_i is None:
        t_i = h_1.t[0]

    if t_f is None:
        t_f = h_1.t[-1]

    # Eliminate unwanted modes
    if modes is not None:
        ell_min = min(h_1.ell_min, h_2.ell_min)
        ell_max = max(h_1.ell_max, h_2.ell_max)
        for L in range(ell_min, ell_max + 1):
            for M in range(-L, L + 1):
                if not (L, M) in modes:
                    h_1.data[:, h_1.index(L, M)] *= 0
                    h_2.data[:, h_2.index(L, M)] *= 0

    h_1_mts = WM_to_MT(h_1)
    h_2_mts = WM_to_MT(h_2)

    h_1_h_2_overlap = overlap(h_1_mts, h_2_mts, t_i, t_f).real
    h_1_norm = overlap(h_1_mts, h_1_mts, t_i, t_f).real
    h_2_norm = overlap(h_2_mts, h_2_mts, t_i, t_f).real

    mismatch = 1 - h_1_h_2_overlap / np.sqrt(h_1_norm * h_2_norm)

    return mismatch
