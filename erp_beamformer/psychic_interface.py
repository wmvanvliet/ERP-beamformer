"""
Code for inter-operability with the Psychic package
https://github.com/wmvanvliet/psychic

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
import psychic
from matplotlib import pyplot as plt

import template

def infer_spatial_pattern(data, y=None, roi_time=None, roi_channels=None,
                          method='peak'):
    """Estimate the spatial pattern of an ERP component from a psychic DataSet.

    This function uses the metadata present in the psychic DataSet object to
    provide a convenient interface to the general purpose infer_spatial_pattern
    function.

    Parameters
    ----------
    data : psychic.DataSet
        The trials in the format of a psychic DataSet.
    y : 1D array (n_trials,) | 2D array (n_trials, 1) | None
        For each trial, a label indicating to which experimental condition
        the trial belongs. If None, data.y is used instead. Defaults to None.
    roi_time : tuple of floats (start, end) | None
        The start and end time (in seconds, end is exclusive) of the time
        region of interest.  When method='peak', the search for maximum
        difference is restricted to this time window. When method='mean', the
        mean signal across this time window is used. If None, the entire time
        window is used. Defaults to None.
    roi_channels : list of floats | None
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
    spat_pat : psychic.DataSet
        The spatial pattern of the ERP component, stored in a psychic DataSet
        object.
    """
    if y is None:
        y = data.y

    channel_names, time = data.feat_lab[:2]

    if roi_time is not None:
        # Translate between seconds and samples
        roi_time = (np.searchsorted(time, roi_time[0]),
                    np.searchsorted(time, roi_time[1]))

    if roi_channels is not None:
        # Translate between channel names and indices
        roi_channels = [channel_names.index(ch) for ch in roi_channels]

    spat_pat = template.infer_spatial_pattern(
        data.data.transpose(2, 0, 1), y, roi_time, roi_channels, method)
    return psychic.DataSet(spat_pat, ids=channel_names)


def infer_temporal_pattern(data, spat_bf, y=None, refine=None,
                           refine_params=None):
    """Estimate temporal pattern of an ERP component from a psychic DataSet.

    This function uses the metadata present in the psychic DataSet object to
    provide a convenient interface to the general purpose
    infer_temporal_pattern function.

    Parameters
    ----------
    data : psychic.DataSet
        The trials in the format of a psychic DataSet.
    spat_bf : instance of LCMV
        The spatial beamformer that will be used to extract the ERP timecourse.
    y : 1D array (n_trials,) | 2D array (n_trials, 1) | None
        For each trial, a label indicating to which experimental condition
        the trial belongs. If None, data.y is used instead. Defaults to None.
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
        roi_time : tuple of floats
            The start and end time (in seconds, end is exclusive) of the time
            region of interest.

        Used by 'gauss':
        mu : float
            Time at which to center the Gaussian kernel in seconds.
        sigma : float
            Standard deviation (in seconds) of the Gaussian kernel.

    Returns
    -------
    temp_pat : psychic.DataSet
        The temporal pattern of the ERP component, stored in a psychic DataSet
        object.
    """
    if y is None:
        y = data.y

    channel_names, time = data.feat_lab[:2]

    if refine_params is None:
        refine_params = dict()

    # Translate between seconds and samples
    if 'roi_time' in refine_params:
        roi_time = refine_params['roi_time']
        roi_time = (np.searchsorted(time, roi_time[0]),
                    np.searchsorted(time, roi_time[1]))
        refine_params['roi_time'] = roi_time
    if 'mu' in refine_params:
        refine_params['mu'] = np.searchsorted(time, refine_params['mu'])
    if 'sigma' in refine_params:
        refine_params['sigma'] = np.searchsorted(time, refine_params['sigma'])

    baseline_time = (0, np.searchsorted(time, 0))

    temp_pat = template.infer_temporal_pattern(
        data.data.transpose(2, 0, 1), y, spat_bf, baseline_time, refine,
        refine_params)
    return psychic.DataSet(temp_pat, ids=time)


def plot_spatial_pattern(spatial_pattern, fig=None):
    if fig is None:
        fig = plt.figure()

    cm = np.max(np.abs(spatial_pattern.data))
    psychic.scalpplot.plot_scalp(spatial_pattern.data.flatten(),
                                 spatial_pattern.ids[0,:],
                                 cmap=plt.get_cmap('RdBu_r'))
    plt.clim(-cm, cm)
    plt.title('spatial pattern')

    return fig


def plot_temporal_pattern(temporal_pattern, fig=None):
    if fig is None:
        fig = plt.figure()

    plt.plot(temporal_pattern.ids[0,:], temporal_pattern.data[0,:], '-k')
    plt.xlim(temporal_pattern.ids.min(), temporal_pattern.ids.max())
    plt.xlabel('time (s)')
    plt.title('temporal pattern')

    return fig


def plot_patterns(spatial_pattern, temporal_pattern, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.subplot(211)
    plot_spatial_pattern(spatial_pattern, fig)
    plt.subplot(212)
    plot_temporal_pattern(temporal_pattern, fig)
    plt.tight_layout()

    return fig
