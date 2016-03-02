import numpy as np
from sklearn.covariance import (EmpiricalCovariance, ShrunkCovariance, OAS,
                                LedoitWolf)
from sklearn.base import BaseEstimator, TransformerMixin


class LCMV(BaseEstimator, TransformerMixin):
    '''
    LCMV Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels)
        Spatial activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    center : bool (default: True)
        Whether to remove the channel mean before applying the filter.

    Attributes
    ----------
    W_ : 2D array (1 x n_channels)
        Row vector containing the filter weights.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        self.template = template
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.center = center

        if center:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def fit(self, X, y=None):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : 3D array (n_channels, n_samples, n_trials)
            The trials.
        y : None
            Unused.
        """
        if self.center:
            X = X - X.mean(axis=0)

        # Concatenate trials
        cont_eeg = np.transpose(X, [0, 2, 1]).reshape((X.shape[0], -1))

        # Calculate spatial covariance matrix
        c = self.cov().fit(cont_eeg.T)
        sigma_x_i = c.precision_

        # Compute spatial LCMV filter
        self.W_ = sigma_x_i.dot(self.template)

        # Noise normalization
        self.W_ = self.W_.dot(
            np.linalg.inv(
                reduce(np.dot, [self.template.T, sigma_x_i, self.template])
            )
        )

        return self

    def transform(self, X):
        """Transform the data using the beamformer.

        Parameters
        ----------
        X : 3D array (n_channels, n_samples, n_trials)
            The trials.

        Returns
        -------
        X_trans : 3D array (1, n_samples, n_trials)
            The transformed data.
        """
        if self.center:
            X = X - X.mean(axis=0)

        n_channels = self.W_.shape[1]
        n_samples = X.shape[1]
        n_trials = X.shape[2]

        X_trans = np.zeros((n_channels, n_samples, n_trials))
        for i in range(n_trials):
            X_trans[:, :, i] = np.dot(self.W_.T, X[:, :, i])

        return X_trans


class stLCMV(BaseEstimator, TransformerMixin):
    '''
    Spatio-temporal LCMV beamformer operating on a spatio-temporal template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Spatio-temporal activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    center : bool (default: True)
        Whether to remove the data mean before applying the filter.

    Attributes
    ----------
    W_ : 2D array (1 x (n_channels * n_samples))
        Row vector containing the filter weights.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        self.template = template
        self.template = np.atleast_2d(template)
        self.center = center

        if center:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def _center(self, X):
        data_mean = X.reshape(-1, X.shape[2]).mean(axis=1)
        data_mean = data_mean.reshape(X.shape[:2] + (1,))
        return X - data_mean

    def fit(self, X, y):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : 3D array (n_channels, n_samples, n_trials)
            The trials.
        y : None
            Unused.
        """
        if self.center:
            X = self._center(X)

        nsamples, ntrials = X.shape[1:]
        template = self.template[:, :nsamples]

        c = self.cov().fit(X.reshape(-1, ntrials).T)
        sigma_x_i = c.precision_

        template = self.template.flatten()[:, np.newaxis]
        self.W_ = sigma_x_i.dot(template)

        # Noise normalization
        self.W_ = self.W_.dot(
            np.linalg.inv(reduce(np.dot, [template.T, sigma_x_i, template]))
        )

        return self

    def transform(self, X):
        """Transform the data using the beamformer.

        Parameters
        ----------
        X : 3D array (n_channels, n_samples, n_trials)
            The trials.

        Returns
        -------
        X_trans : 3D array (1, n_trials)
            The transformed data.
        """
        if self.center:
            X = self._center(X)

        n_trials = X.shape[2]
        X_trans = self.W_.T.dot(X.reshape(-1, n_trials))
        return X_trans


def infer_spatial_pattern(X, y, roi_time=None, roi_channels=None,
                          method='peak'):
    """Estimate the spatial pattern of an ERP component.

    The spatial pattern is constructed by constructing the ERP difference
    waveform between two experimental conditions. The spatial pattern is then
    defined as the signal at the time of maximal difference (method='peak'),
    or the mean signal over a given time interval (method='mean').

    Parameters
    ----------
    X : 3D array (n_channels, n_samples, n_trials)
        The trials.
    y : list of ints
        For each trial, a label indicating to which experimental condition
        the trial belongs.
    roi_time : tuple of ints (start, end) | None
        The start and end time (in samples) of the time region of interest.
        When method='peak', the search for maximum difference is restricted
        to this time window. When method='mean', the mean signal across this
        time window is used. If None, the entire time window is used.
        Defaults to None.
    roi_channels : list of ints | None
        When method='peak', restrict the search for maximum difference to the
        channels with the given indices. When None, do not restrict the search.
        Defaults to None.
    method : 'peak' | 'mean'
        When 'peak', the spatial pattern is the signal at the time of maximum
        difference between the experimental conditions.
        When 'mean', the spatial pattern is the mean difference waveform
        between the experimental conditions. Defaults to 'peak'.

    Returns
    -------
    spat_pat : 1D array (n_channels)
        The spatial pattern of the ERP component.
    """
    conditions = np.unique(y)
    if len(conditions) != 2:
        raise ValueError('There should be exactly 2 experimental conditions.')

    if roi_channels is None:
        roi_channels = (0, X.shape[0])

    if roi_time is None:
        roi_time = (0, X.shape[1])

    # Compute ERP difference waveform
    diff = (X[:, :, y == conditions[0]].mean(axis=2) -
            X[:, :, y == conditions[1]].mean(axis=2))

    if method == 'peak':
        # Extract region of interest to look for the roi
        ROI = diff[roi_channels, roi_time[0]:roi_time[1]]

        # Determine peak time for the roi
        peak_time = np.argmax(np.max(np.abs(ROI), axis=0)) + roi_time[0]

        spat_pat = diff[:, peak_time]

    elif method == 'mean':
        # Extract region of interest to look for the roi
        ROI = diff[:, roi_time[0]:roi_time[1]]

        spat_pat = diff.mean(axis=1)

    # Normalize the spatial template so all values are in the range [-1, 1]
    spat_pat /= np.max(np.abs(spat_pat))

    return spat_pat


def refine_temporal_pattern(temp_pat, baseline_time, roi_time=None,
                            method='peak_mean'):
    """Refine the estimation of a temporal pattern.

    Parameters
    ----------
    temp_pat : 1D array (n_samples)
        The temporal pattern to refine
    baseline_time : tuple of ints
        The start and end time (in samples) of the baseline period (the period
        before the onset of the event marker). This period is used to
        re-baseline the signal after the spatial beamformer has been applied to
        it.
    roi_time : tuple of ints | None
        The start and end time (in samples) of the time region of interest.
        This region is used during the refining stage. If None, all samples are
        marked as the ROI. Defaults to None.
    method : 'zero' | 'zero-cross' | 'thres' | None
        The method used to refine the template:
        'zero':      Zero out everything outside the time region of interest.
        'peak-mean': Find the peak inside the time region of interest. Then,
                     find the points before and after the peak, where the
                     signal drops below the average signal outside the time
                     region of interest.  Zero out everything outside those
                     points.
        'thres':     As well as zero-ing out everything outside the time region
                     of interest, also zero out any part of the signal which
                     amplitude is below 4 standard deviations of the signal
                     amplitude during the baseline period.
        Defaults to 'peak-mean'.

    Returns
    -------
    ref_temp_pat : 1D array (n_samples)
        The refined pattern.
    """
    ref_temp_pat = temp_pat.copy()

    # Refine the template if requested
    if method == 'zero':
        ref_temp_pat -= np.min(ref_temp_pat[:, roi_time[0]:roi_time[1]])
        ref_temp_pat[:, :roi_time[0]] = 0
        ref_temp_pat[:, roi_time[1]:] = 0

    elif method == 'peak-mean':
        ref_temp_pat -= (
            np.mean(ref_temp_pat[:, :roi_time[0]]) +
            np.mean(ref_temp_pat[:, roi_time[1]:])
        ) / 2.

        peak_time = np.argmax(np.abs(ref_temp_pat[:, roi_time[0]:roi_time[1]]))
        peak_time += roi_time[0]

        # Find zero crossings
        lcross = peak_time
        while lcross > 0 and ref_temp_pat[:, lcross] > 0:
            lcross -= 1

        rcross = peak_time
        while rcross < len(ref_temp_pat)-1 and ref_temp_pat[:, rcross] > 0:
            rcross += 1

        # Limit temporal pattern to area between zero crossings
        ref_temp_pat[:, :lcross + 1] = 0
        ref_temp_pat[:, rcross:] = 0
        ref_temp_pat -= np.min(ref_temp_pat[:, lcross + 1:rcross])

    elif method == 'thres':
        baseline_std = np.std(
            np.abs(ref_temp_pat[:, baseline_time[0]:baseline_time[1]])
        )
        mask = np.abs(ref_temp_pat) < 4 * baseline_std
        ref_temp_pat -= np.min(ref_temp_pat[np.logical_not(mask)])
        ref_temp_pat[mask] = 0
        ref_temp_pat[:, :roi_time[0]] = 0
        ref_temp_pat[:, roi_time[1]:] = 0

    elif method is not None:
        raise ValueError("Invalid value for refine parameter. Choose one of "
                         "'zero', 'peak-mean', 'thres' or None.")

    return ref_temp_pat


def infer_temporal_pattern(X, y, spat_bf, baseline_time, roi_time=None,
                           refine='zero'):
    """Estimate the temporal pattern of an ERP component.

    The temporal pattern is constructed by using a spatial beamformer to
    estimate the ERP timecourse of trials belonging to two experimental
    conditions. The temporal pattern is then defined as the difference
    timecourse between at the two conditions. This pattern is than refined by
    zero-ing out everything outside of the time region of interest.

    Parameters
    ----------
    X : 3D array (n_channels, n_samples, n_trials)
        The trials.
    y : list of ints
        For each trial, a label indicating to which experimental condition
        the trial belongs.
    spat_bf : instance of LCMV
        The spatial beamformer that will be used to extract the ERP timecourse.
    baseline_time : tuple of ints
        The start and end time (in samples) of the baseline period (the period
        before the onset of the event marker). This period is used to
        re-baseline the signal after the spatial beamformer has been applied to
        it.
    roi_time : tuple of ints | None
        The start and end time (in samples) of the time region of interest.
        This region is used during the refining stage. If None, all samples are
        marked as the ROI. Defaults to None.

    Returns
    -------
    temp_pat : 1D array (n_samples,)
        The temporal pattern of the ERP component.
    """
    conditions = np.unique(y)
    if len(conditions) != 2:
        raise ValueError('There should be exactly 2 experimental conditions.')

    if roi_time is None:
        roi_time = (0, X.shape[1])

    timecourses = spat_bf.fit_transform(X, y)

    # Re-baseline the signal
    timecourses -= np.mean(
        timecourses[:, baseline_time[0]:baseline_time[1], :],
        axis=1
    )

    # Use the difference timecourse as initial estimate of the temporal pattern
    temp_pat = (timecourses[:, :, y == conditions[0]].mean(axis=2) -
                timecourses[:, :, y == conditions[1]].mean(axis=2))

    temp_pat = refine_temporal_pattern(temp_pat, baseline_time, roi_time,
                                       refine)

    # Normalize the temporal template so all values are in the range [-1, 1]
    temp_pat /= np.max(np.abs(temp_pat))

    return temp_pat
