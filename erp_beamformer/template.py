"""
Some functions for designing spatial and temporal patterns for use with
beamformer filters.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


def infer_spatial_pattern(X, y, roi_time=None, roi_channels=None,
                          method='peak'):
    """Estimate the spatial pattern of an ERP component.

    The spatial pattern is constructed by constructing the ERP difference
    waveform between two experimental conditions. The spatial pattern is then
    defined as the signal at the time of maximal difference (method='peak'),
    or the mean signal over a given time interval (method='mean').

    Parameters
    ----------
    X : 3D array (n_trials, n_channels, n_samples)
        The trials.
    y : 1D array (n_trials,) or 2D array (n_trials, 1)
        For each trial, a scalar representing an estimation of the amplitude of
        the ERP component of interest in that trial.
    roi_time : tuple of ints (start, end) | None
        The start and end time (in samples, end is exclusive) of the time
        region of interest.  When method='peak', the search for maximum
        difference is restricted to this time window. When method='mean', the
        mean signal across this time window is used. If None, the entire time
        window is used.  Defaults to None.
    roi_channels : list of ints | None
        When method='peak', restrict the search for maximum difference to the
        channels with the given indices. When None, do not restrict the search.
        Defaults to None.
    method : 'peak' | 'mean'
        When 'peak', the spatial pattern is the signal at the time of maximum
        squared difference between the experimental conditions.
        When 'mean', the spatial pattern is the mean difference waveform
        between the experimental conditions. Defaults to 'peak'.

    Returns
    -------
    spat_pat : 1D array (n_channels,)
        The spatial pattern of the ERP component.
    """
    n_trials, n_channels, n_samples = X.shape

    if roi_channels is None:
        roi_channels = range(n_channels)

    if roi_time is None:
        roi_time = (0, n_samples)

    # Compute slope ERP
    model = LinearRegression().fit(X.reshape(n_trials, -1), y)
    slope = model.coef_.reshape(n_channels, n_samples)

    if method == 'peak':
        # Extract region of interest to look for the roi
        ROI = slope[roi_channels, roi_time[0]:roi_time[1]]

        # Determine peak time for the roi
        peak_time = np.argmax(np.sum(ROI ** 2, axis=0)) + roi_time[0]

        spat_pat = slope[:, peak_time]

    elif method == 'mean':
        # Extract region of interest to look for the roi
        ROI = slope[:, roi_time[0]:roi_time[1]]
        spat_pat = ROI.mean(axis=1)

    # Normalize the spatial template so all values are in the range [-1, 1]
    spat_pat /= np.max(np.abs(spat_pat))

    return spat_pat


def refine_pattern(pattern, method, method_params):
    """Refine a pattern.

    Refine a spatio-temporal or temporal pattern by setting some samples to
    zero.

    Parameters
    ----------
    pattern : 1D array (n_samples) | 2D array (n_channels, n_samples)
        The temporal pattern to refine
    method : 'zero' | 'peak-mean' | 'thres' | 'gauss'
        The method used to refine the template:
        'zero':      Zero out everything outside the time region of interest.
        'peak-mean': Find the peak inside the time region of interest. Then,
                     find the points before and after the peak, where the
                     signal drops below the average signal outside the time
                     region of interest. Zero out everything outside those
                     points.
        'thres':     As well as zero-ing out everything outside the time region
                     of interest, also zero out any part of the signal which
                     amplitude is below 4 standard deviations of the signal
                     amplitude during the baseline period.
        'gauss':     Multiply the signal with a Gaussian kernel that is defined
                     over time.
    method_params : dict
        Parameters for the chosen method. Each method uses different parameters
        taken from this dictionary. Possible parameters are:

        Used by 'zero', 'peak-mean' and 'thres':
        roi_time : tuple of ints
            The start and end time (in samples, end is exclusive) of the time
            region of interest.

        Used by 'thres':
        baseline_time : tuple of ints
            The start and end time (in samples, end is exclusive) of the
            baseline period (the period before the onset of the event marker).

        Used by 'gauss':
        mu : int
            Sample at which to center the Gaussian kernel.
        sigma : float
            Standard deviation (in samples) of the Gaussian kernel.

    Returns
    -------
    ref_pat : 1D array (n_samples) | 2D array (n_channels, n_samples)
        The refined pattern.
    """
    pattern = np.asarray(pattern, float)
    orig_dim = pattern.shape
    ref_pat = np.atleast_2d(pattern.copy())

    if method == 'zero':
        try:
            roi_time = method_params['roi_time']
        except:
            raise ValueError('The parameter "roi_time" is missing from the '
                             'method_params dictionary.')
        ref_pat[:, :roi_time[0]] = 0
        ref_pat[:, roi_time[1]:] = 0

    elif method == 'peak-mean':
        try:
            roi_time = method_params['roi_time']
        except:
            raise ValueError('The parameter "roi_time" is missing from the '
                             'method_params dictionary.')

        if roi_time[0] >= roi_time[1]:
            raise ValueError('Empty ROI given for roi_time parameter.')

        # Re-baseline using the mean of the signal outside the ROI
        ref_pat -= (
            np.mean(
                np.hstack(
                    (ref_pat[:, :roi_time[0]],
                     ref_pat[:, roi_time[1]:])
                ),
                axis=1
            )[:, np.newaxis]
        ) / 2.

        # Determine for each channel the peak time
        peak_time = np.argmax(
            ref_pat[:, roi_time[0]:roi_time[1]] ** 2,
            axis=1
        )
        peak_time += roi_time[0]

        # Find zero crossings
        for ch in range(ref_pat.shape[0]):
            sign = np.sign(ref_pat[ch, peak_time[ch]])
            lcross = peak_time[ch]
            while lcross > 0 and np.sign(ref_pat[ch, lcross]) == sign:
                lcross -= 1

            rcross = peak_time[ch]
            while (rcross < ref_pat.shape[1] - 1 and
                   np.sign(ref_pat[ch, rcross]) == sign):
                rcross += 1

            # Limit temporal pattern to area between zero crossings
            ref_pat[ch, :lcross + 1] = 0
            ref_pat[ch, rcross:] = 0

    elif method == 'thres':
        try:
            roi_time = method_params['roi_time']
            baseline_time = method_params['baseline_time']
        except:
            raise ValueError('The parameter "roi_time" and/or "baseline_time" '
                             'is missing from the method_params dictionary.')
        baseline_std = np.std(
            np.abs(ref_pat[:, baseline_time[0]:baseline_time[1]])
        )
        mask = np.abs(ref_pat) < 4 * baseline_std
        ref_pat[mask] = 0
        ref_pat[:, :roi_time[0]] = 0
        ref_pat[:, roi_time[1]:] = 0

    elif method == 'gauss':
        try:
            mu = method_params['mu']
            sigma = method_params['sigma']
        except:
            raise ValueError('The parameter "mu" and/or "sigma" is missing '
                             'from the method_params dictionary.')

        kernel = norm(mu, sigma).pdf(np.arange(ref_pat.shape[1]))
        kernel /= kernel.max()
        ref_pat *= kernel[np.newaxis, :]

    else:
        raise ValueError("Invalid value for refine parameter. Choose one of "
                         "'zero', 'peak-mean', 'thres' or 'gauss'.")

    return ref_pat.reshape(orig_dim)


def infer_temporal_pattern(X, y, spat_bf, baseline_time, refine=None,
                           refine_params=None):
    """Estimate the temporal pattern of an ERP component.

    The temporal pattern is constructed by using a spatial beamformer to
    estimate the ERP timecourse of trials belonging to two experimental
    conditions. The temporal pattern is then defined as the difference
    timecourse between at the two conditions. This pattern is then optionally
    refined by zero-ing out irrelevant samples.

    Parameters
    ----------
    X : 3D array (n_trials, n_channels, n_samples)
        The trials.
    y : 1D array (n_trials,) or 2D array (n_trials, 1)
        For each trial, a scalar representing an estimation of the amplitude of
        the ERP component of interest in that trial.
    spat_bf : instance of LCMV
        The spatial beamformer that will be used to extract the ERP timecourse.
    baseline_time : tuple of ints
        The start and end time (in samples, end is exclusive) of the baseline
        period (the period before the onset of the event marker). This period
        is used to re-baseline the signal after the spatial beamformer has been
        applied to it.
    refine : 'zero' | 'peak-mean' | 'thres' | 'guass' | None
        The method used to refine the template:
        'zero':      Zero out everything outside the time region of interest.
        'peak-mean': Find the peak inside the time region of interest. Then,
                     find the points before and after the peak, where the
                     signal drops below the average signal outside the time
                     region of interest.  Zero out everything outside those
                     points.
        'thres':     As well as zero-ing out everything outside the time
                     region of interest, also zero out any part of the signal
                     which amplitude is below 4 standard deviations of the
                     signal amplitude during the baseline period.
        'gauss':     Multiply the signal with a Gaussian kernel that is defined
                     over time.
        Defaults to None, which means no refining of the template is performed.
    refine_params : dict | None
        Parameters for the chosen refining method. Each method uses different
        parameters taken from this dictionary. Possible parameters are:

        Used by 'zero', 'peak-mean' and 'thres':
        roi_time : tuple of ints
            The start and end time (in samples, end is exclusive) of the time
            region of interest.

        Used by 'thres':
        baseline_time : tuple of ints
            Copied from the baseline_time parameter given to the
            infer_temporal_pattern function.

        Used by 'gauss':
        mu : int
            Sample at which to center the Gaussian kernel.
        sigma : float
            Standard deviation (in samples) of the Gaussian kernel.

    Returns
    -------
    temp_pat : 1D array (n_samples,)
        The temporal pattern of the ERP component.
    """
    # Apply spatial beamformer to extract the ERP component timecourse
    timecourse = spat_bf.fit_transform(X, y)

    # Re-baseline the signal
    timecourse -= np.mean(timecourse[:, baseline_time[0]:baseline_time[1]])

    # Compute slope ERP, which will be the temporal template
    print timecourse.shape, y.shape
    model = LinearRegression().fit(timecourse, y)
    temp_pat = model.coef_.ravel()

    if refine is not None:
        if refine_params is None:
            refine_params = dict()
        refine_params['baseline_time'] = baseline_time
        temp_pat = refine_pattern(temp_pat, refine, refine_params)

    # Re-baseline the signal
    temp_pat -= np.mean(temp_pat[baseline_time[0]:baseline_time[1]])

    # Normalize the temporal template so all values are in the range [-1, 1]
    temp_pat /= np.max(np.abs(temp_pat))

    return temp_pat
