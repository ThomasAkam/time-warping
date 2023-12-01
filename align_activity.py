"""Code for time warping neurophysiology data to align activity across 
trials.  The align_spikes function is used to align spike activity in
the form of neuron IDs corresponding spike times. The align_signals 
function is used to align continuous data, e.g. photometry signals or 
motion tracking data from DeepLabCut. 

https://github.com/ThomasAkam/time-warping

Copyright (c) Thomas Akam 2022. Licenced under the GNU General Public License v3.
"""

import numpy as np
import pylab as plt
from scipy.stats import norm
from scipy.special import erf

# Set plotting paramters.

plt.rcParams["pdf.fonttype"] = 42
plt.rc("axes.spines", top=False, right=False)

# --------------------------------------------------------------------------------
# Align spikes
# --------------------------------------------------------------------------------


def align_spikes(trial_times, target_times, spikes, pre_win=0, post_win=0, fs_out=25, smooth_SD="default", plot=False):
    """Calculate trial aligned smoothed firing rates from spike time data. Spike times
    are first transformed from the original time frame to a trial aligned time frame in
    which a set of reference time points for each trial are mapped onto a set of target
    time points (e.g. the median trial timings), with linear interpolation of spike times
    between the reference points. Once the spike times have been transformed into the
    trial aligned reference frame the firing rate is calculated at a specified sampling
    rate, using Gaussian smoothing with a specified standard deviation, taking into
    acount the change in spike density due to the time warping.  The pre_win and post_win
    arguments can be used to specify time windows before the first and after the last
    alignment event on each trial to be included in the output signals.

    Arguments:
    trial_times : Array of reference point times for each trial (seconds) [n_trials, n_ref_points]
    target_times: Reference point times to warp each trial onto (seconds) [n_ref_points]
    spikes:  Array of neuron IDs and spike times [2, n_spikes]
             spikes[0,:] is neuron IDs, spikes [1,:] is spike times (seconds).
    pre_win: Time window before first event to include in aligned rates (seconds).
    post_win: Time window after last event to include in aligned rates (seconds).
    fs_out: Sampling rate of output firing rate (Hz).
    smooth_SD: Standard deviation of gaussian smoothing applied to ouput rate (seconds).
               If set to default, smooth_SD is set to the inter sample interval.
    plot: If set to True, plots the time warping for the most active neuron on each trial.
          of the most active neurons activity.
    Returns dictionary with items:
    aligned_rates: Array of trial aligned smoothed firing rates (Hz) [n_trials, n_neurons, n_timepoints].
    t_out: Times of each output firing rate time point (seconds).
    min_max_stretch: Minimum and maximum stretch factor for each trial.  Used to exclude
                     trials which have extreme deviation from target timings [n_trials, 2]
    """
    if smooth_SD == "default":
        smooth_SD = 1 / fs_out
    n_trials = trial_times.shape[0]
    neuron_IDs = np.sort(np.unique(spikes[0, :]))
    n_neurons = len(neuron_IDs)
    t_out = np.arange(target_times[0] - pre_win, target_times[-1] + post_win, 1 / fs_out)  # Output timepoints.
    pad_len = smooth_SD * 4
    # Add non-warped interval before and after specified intervals to prevent edge effects and
    # include the pre and post windows if specified.
    target_times = np.hstack([target_times[0] - pre_win - pad_len, target_times, target_times[-1] + post_win + pad_len])
    trial_times = np.hstack(
        [trial_times[:, 0, None] - pre_win - pad_len, trial_times, trial_times[:, -1, None] + post_win + pad_len]
    )
    # Apply timewarping.
    target_deltas = np.diff(target_times)  # Intervals between target time points.
    trial_deltas = np.diff(trial_times, 1)  # Intervals between reference points for each trial.
    stretch_factors = target_deltas / trial_deltas  # Amount each interval of each trial must be stretched/squashed by.
    min_max_stretch = np.vstack(
        [np.min(stretch_factors, 1), np.max(stretch_factors, 1)]
    ).T  # Minimum and maximum stretch factor for each trial.
    aligned_rates = np.zeros([n_trials, n_neurons, len(t_out)])  # Array to store trial aligned firing rates.
    for tr in range(n_trials):  # Loop over trials.
        trial_spikes = spikes[:, (trial_times[tr, 0] < spikes[1, :]) & (spikes[1, :] < trial_times[tr, -1])]
        trial_spike_IDs = trial_spikes[0, :]
        trial_spike_times = trial_spikes[1, :]
        aligned_sp_times = trial_spike_times.copy()
        # Change times of trial_spikes to map them onto target_times.
        spike_stretch = np.zeros(trial_spikes.shape[1])  # Stretch factor for each spike.
        for i in range(len(target_times) - 1):  # Loop over intervals.
            interval_mask = (
                trial_times[tr, i] < trial_spike_times
            ) & (  # Boolean mask indicating which spikes are in interval i.
                trial_spike_times < trial_times[tr, i + 1]
            )
            aligned_sp_times[interval_mask] = target_times[i] + stretch_factors[tr, i] * (
                aligned_sp_times[interval_mask] - trial_times[tr, i]
            )
            spike_stretch[interval_mask] = stretch_factors[tr, i]
        for j, n in enumerate(neuron_IDs):  # Loop over neurons.
            if n in trial_spike_IDs:
                neuron_mask = trial_spike_IDs == n
                n_spike_times = aligned_sp_times[neuron_mask]
                aligned_rates[tr, j, :] = np.sum(
                    norm.pdf(n_spike_times[None, :] - t_out[:, None], scale=smooth_SD) * spike_stretch[neuron_mask], 1
                )
        if plot:  # Plot trial alignment for neuron most active on trial.
            plt.figure(1, figsize=[4.6, 7], clear=True)
            fig, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [5, 1, 2]}, num=1)
            # Plot spike times in input and output reference frames.
            plt.sca(ax[0])
            most_active = np.argmax(np.mean(aligned_rates[tr, :, :], axis=1))
            neuron_mask = trial_spike_IDs == neuron_IDs[most_active]
            n_spikes = np.sum(neuron_mask)
            ylim = [
                trial_times[tr, 0] - trial_times[tr, 1] + pad_len,
                trial_times[tr, -1] - trial_times[tr, 1] - pad_len,
            ]
            plt.scatter(
                np.ones(n_spikes) * t_out[0], trial_spike_times[neuron_mask] - trial_times[tr, 1], s=60, marker=1
            )
            plt.scatter(
                aligned_sp_times[neuron_mask], np.ones(n_spikes) * ylim[0], s=60 * spike_stretch[neuron_mask], marker=2
            )
            plt.plot(target_times, trial_times[tr, :] - trial_times[tr, 1], ".:", c="k")
            plt.ylabel("True time (seconds)")
            plt.xlim(t_out[0], t_out[-1])
            plt.ylim(*ylim)
            # Plot Gaussians used for smoothing output firing rate.
            plt.sca(ax[1])
            t_g = np.arange(t_out[0], t_out[-1])
            for t in t_out:
                g_max = 1 / norm.pdf(0, scale=smooth_SD)
                plt.plot(t_g, g_max * norm.pdf(t - t_g, scale=smooth_SD), c="grey", linewidth=0.5)
            for t in target_times[1:-1]:
                plt.axvline(t, color="k", linestyle=":")
            plt.xlim(t_out[0], t_out[-1])
            plt.tick_params(labelbottom=False, labelleft=False)
            # Plot output firing rate.
            plt.sca(ax[2])
            plt.plot(t_out, aligned_rates[tr, most_active, :])
            for t in target_times[1:-1]:
                plt.axvline(t, color="k", linestyle=":")
            plt.xlim(t_out[0], t_out[-1])
            plt.ylim(ymin=0)
            plt.xlabel("Aligned time (seconds)")
            plt.ylabel("Firing rate (Hz)")
            plt.tight_layout()
            plt.pause(0.05)
            if input("Press enter for next trial, 'x' to stop plotting:") == "x":
                plot = False
    return {"aligned_rates": aligned_rates, "t_out": t_out, "min_max_stretch": min_max_stretch}


# --------------------------------------------------------------------------------
# Align signals
# --------------------------------------------------------------------------------


def align_signals(
    signals,
    sample_times,
    trial_times,
    target_times,
    pre_win=0,
    post_win=0,
    fs_out=25,
    smooth_SD="auto",
    plot_warp=False,
):
    """
    Timewarp continuous signals to align event times on each trial to specified target
    event times. For each trial, input sample times are linearly time warped to align
    that trial's event times with the target times.  Activity is then evaluated at a set of
    regularly spaced timepoints relative to the target event times by linearly interpolating
    signals between input samples followed by Gaussian smoothing around output timepoints.
    This allows a single mathematical operation to handle both time streching (where
    interpolation in needed) and time compression (where averaging is needed). The pre_win
    and post_win arguments can be used to specify time windows before the first and after
    the last alignment event on each trial to be included in the output signals.

    Arguments:
        signals      : Signals to be aligned, either 1D [n_samples] or 2D [n_signals, n_samples]
        sample_times : Times when the samples occured (seconds) [n_samples]
        trial_times  : Times of events used for alignment for each trial (seconds) [n_trials, n_events]
        target_times : Times of events used for alignment in output aligned trial (seconds) [n_events].
        pre_win      : Time window before first event to include in aligned signals (seconds).
        post_win     : Time window after last event to include in aligned signals (seconds).
        fs_out       : The sampling rate of the aligned output signals (Hz).
        smooth_SD    : Standard deviation (seconds) of Gaussian smoothing applied to output signals.
                       If set to 'auto', smooth_SD is set to 1/fs_out.
        plot_warp    : If True the input and output signals are plotted for the most active
                       neurons for each trial.

    Returns:
        aligned_signals : Array of trial aligned signals [n_trials, n_timepoints, n_signals]
                          or [n_trials, n_timepoints] if signals is a 1D array]
        t_out: Times of each output firing rate time point (seconds).
        min_max_stretch: Minimum and maximum stretch factor for each trial.  Used to exclude
                         trials which have extreme deviation from target timings [n_trials, 2]
    """
    assert not np.any(np.diff(trial_times, 1) < 0), "trial_times give negative interval duration"
    assert not np.any(np.diff(target_times) < 0), "target_times give negative interval duration"

    one_dim_signal = len(signals.shape) == 1
    if one_dim_signal:
        signals = signals[np.newaxis, :]

    if smooth_SD == "auto":
        smooth_SD = 1 / fs_out

    t_out = np.arange(
        target_times[0] - pre_win, target_times[-1] + post_win, 1 / fs_out
    )  # Timepoints of output samples.

    n_trials = trial_times.shape[0]
    n_signals = signals.shape[0]
    n_timepoints = len(t_out)

    # Add non-warped interval before and after specified intervals to prevent edge effects and
    # include the pre and post windows if specified.

    pad_len = smooth_SD * 4  # Extension to alignement interval to prevent edge effects.
    target_times = np.hstack([target_times[0] - pre_win - pad_len, target_times, target_times[-1] + post_win + pad_len])
    trial_times = np.hstack(
        [trial_times[:, 0, None] - pre_win - pad_len, trial_times, trial_times[:, -1, None] + post_win + pad_len]
    )

    # Compute inter-event intervals and stretch factors to align trial intervals to target intervals.

    target_deltas = np.diff(target_times)  # Duration of inter-event intervals for aligned signals (ms).
    trial_deltas = np.diff(trial_times, 1)  # Duration of inter-event intervals for each trial (ms).

    stretch_factors = target_deltas / trial_deltas  # Amount each interval of each trial must be stretched/squashed by.
    min_max_stretch = np.stack([np.min(stretch_factors, 1), np.max(stretch_factors, 1)]).T  # Trial min & max stretch.

    # Loop over trial computing aligned signals.

    aligned_signals = np.full([n_trials, n_timepoints, n_signals], np.nan)

    for tr in np.arange(n_trials):
        if trial_times[tr, 0] < sample_times[0]:
            continue  # This trial occured before signals started.
        if trial_times[tr, -1] > sample_times[-1]:
            break  # This and subsequent trials occured after signals finshed.

        # Linearly warp sample times to align inter-event intervals to target.
        trial_samples = (trial_times[tr, 0] <= sample_times) & (sample_times < trial_times[tr, -1])
        trial_signals = signals[:, trial_samples]
        trial_sample_time = sample_times[trial_samples]  # Trial sample times before warping
        aligned_sample_times = np.zeros(len(trial_sample_time))  # Trial sample times after warping
        for j in range(target_deltas.shape[0]):
            mask = (trial_times[tr, j] <= trial_sample_time) & (trial_sample_time < trial_times[tr, j + 1])
            aligned_sample_times[mask] = (trial_sample_time[mask] - trial_times[tr, j]) * (
                target_deltas[j] / trial_deltas[tr, j]
            ) + target_times[j]

        # Calculate aligned signals.
        aligned_signals[tr, :, :] = _compute_aligned_signals(trial_signals, aligned_sample_times, t_out, smooth_SD)

        if plot_warp:  # Plot input and output signals for the most active neurons.
            most_active = np.argsort(np.mean(trial_signals, 1))[-5:]
            plt.figure(2, figsize=[10, 3.2]).clf()
            plt.subplot2grid((1, 3), (0, 0))
            plt.plot(trial_sample_time, aligned_sample_times, ".-")
            plt.ylabel("Aligned trial time (seconds)")
            plt.xlabel("True trial time (seconds)")
            plt.subplot2grid((2, 3), (0, 1), colspan=2)
            plt.plot(trial_sample_time, trial_signals[most_active, :].T, ".-")
            for x in trial_times[tr, :]:
                plt.axvline(x, color="k", linestyle=":")
            plt.xlim(trial_times[tr, 1] - pre_win, trial_times[tr, -2] + post_win)
            plt.ylabel("Activity")
            plt.xlabel("True trial time (seconds)")
            plt.subplot2grid((2, 3), (1, 1), colspan=2)
            plt.plot(t_out, aligned_signals[tr, :, most_active].T, ".-")
            for x in target_times:
                plt.axvline(x, color="k", linestyle=":")
            plt.xlim(t_out[0], t_out[-1])
            plt.ylabel("Activity")
            plt.xlabel("Aligned trial time (seconds)")
            plt.tight_layout()
            plt.pause(0.05)
            if input("Press enter for next trial, 'x' to stop plotting:") == "x":
                plot_warp = False

    if one_dim_signal:  # Drop singleton dimension from output.
        aligned_signals = aligned_signals[:, :, 0]

    return aligned_signals, t_out, min_max_stretch


def _compute_aligned_signals(trial_signals, aligned_sample_times, t_out, smooth_SD):
    """Evaluate the time aligned signals for a single trial by computing the overlap integral between
    a linear interpolation of the signal between each adjacent pair of samples in the time-aligned
    reference frame, and a set of Gaussians around the output timepoints. The integral is
    computed seperately for each linear section of signal between adjacent samples using the
    _integrate_gaussian_linear_product function below, with numpy array broadcasting used to
    compute the integrals for all sections, signals and output timepoints in a single function call.
    Paramters:
        trial_signals:  Signal samples for the current trial [n_signals, n_trial_timepoints]
        aligned_sample_times: Sample times in the aligned reference frame [n_trial_timepoints]
        t_out: Regularly spaced timepoints in the aligned reference frame at which to evaluate the output signal.
        smooth_SD: Standard deviation of Gaussian distibution around output samples used for smoothing.
    """
    s0 = trial_signals[:, :-1]
    s1 = trial_signals[:, 1:]
    t0 = aligned_sample_times[:-1]
    t1 = aligned_sample_times[1:]
    piece_wise_integral = _integrate_gaussian_linear_product(s0, s1, t0, t1, u=t_out[:, None, None], s=smooth_SD)
    aligned_signals = np.sum(piece_wise_integral, 2)  # Sum over signal sections.
    return aligned_signals


def _integrate_gaussian_linear_product(s0, s1, t0, t1, u, s):
    """Evaluate the overlap integral between a straight line and Gaussian probability density function
    over the interval t0 to t1, where the line takes values s0 at t0 and s1 at t1, and the Gaussian
    is mean u, standard deviation s.

    integrate (s0+(s1-s0)*(t-t0)/(t1-t0))*Gaussian_pdf(u,s) dt from t0 to t1
    """
    r2pi = np.sqrt(2 * np.pi)
    r2 = np.sqrt(2)
    return (1 / (2 * r2pi * (t1 - t0))) * (
        r2pi * (s0 * (t1 - u) + s1 * (u - t0)) * (erf((t1 - u) / (r2 * s)) - erf((t0 - u) / (r2 * s)))
        + 2 * s * (s0 - s1) * (np.exp(-((t1 - u) ** 2) / (2 * s**2)) - np.exp(-((t0 - u) ** 2) / (2 * s**2)))
    )
