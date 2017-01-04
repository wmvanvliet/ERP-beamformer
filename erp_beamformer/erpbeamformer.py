import numpy as np
from sklearn.covariance import (EmpiricalCovariance, ShrunkCovariance, OAS,
                                LedoitWolf)
from sklearn.base import BaseEstimator, TransformerMixin


class LCMV(BaseEstimator, TransformerMixin):
    '''
    LCMV Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels,)
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
        X : 3D array (n_trials, n_channels, n_samples)
            The trials.
        y : None
            Unused.
        """
        if self.center:
            X = X - X.mean(axis=1, keepdims=True)

        # Concatenate trials into an (n_channels, n_samples) matrix
        cont_eeg = np.transpose(X, [1, 0, 2]).reshape((X.shape[1], -1))

        # Calculate spatial covariance matrix
        c = self.cov.fit(cont_eeg.T)
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
        X : 3D array (n_trials, n_channels, n_samples)
            The trials.

        Returns
        -------
        X_trans : 2D array (n_trials, n_samples)
            The transformed data.
        """
        if self.center:
            X = X - X.mean(axis=1, keepdims=True)

        n_trials, _, n_samples = X.shape
        X_trans = np.zeros((n_trials, n_samples))
        for i in range(n_trials):
            X_trans[i, :] = np.dot(self.W_.T, X[i, :, :]).ravel()

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
        data_mean = X.reshape(X.shape[0], -1).mean(axis=0)
        data_mean = data_mean.reshape((1,) + X.shape[1:])
        return X - data_mean

    def fit(self, X, y):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : 3D array (n_trials, n_channels, n_samples)
            The trials.
        y : None
            Unused.
        """
        if self.center:
            X = self._center(X)

        n_trials, _, n_samples = X.shape
        template = self.template[:, :n_samples]

        c = self.cov().fit(X.reshape(n_trials, -1))
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
        X : 3D array (n_trials, n_channels, n_samples)
            The trials.

        Returns
        -------
        X_trans : 3D array (n_trials, 1)
            The transformed data.
        """
        if self.center:
            X = self._center(X)

        n_trials = X.shape[0]
        X_trans = X.reshape(n_trials, -1).dot(self.W_)
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
    X : 3D array (n_trials, n_channels, n_samples)
        The trials.
    y : 1D array (n_trials,) or 2D array (n_trials, 1)
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
        squared difference between the experimental conditions.
        When 'mean', the spatial pattern is the mean difference waveform
        between the experimental conditions. Defaults to 'peak'.

    Returns
    -------
    spat_pat : 1D array (n_channels)
        The spatial pattern of the ERP component.
    """
    if y.ndims == 2:
        if y.shape[1] != 1:
            raise ValueError('y should be an (n_trials, 1) array.')
        y = y.ravel()

    conditions = np.unique(y)
    if len(conditions) != 2:
        raise ValueError('There should be exactly 2 experimental conditions.')

    if roi_channels is None:
        roi_channels = (0, X.shape[0])

    if roi_time is None:
        roi_time = (0, X.shape[1])

    # Compute ERP difference waveform
    diff = (X[y == conditions[0]].mean(axis=0) -
            X[y == conditions[1]].mean(axis=0))

    if method == 'peak':
        # Extract region of interest to look for the roi
        ROI = diff[roi_channels, roi_time[0]:roi_time[1]]

        # Determine peak time for the roi
        peak_time = np.argmax(np.max(ROI ** 2, axis=0)) + roi_time[0]

        spat_pat = diff[:, peak_time]

    elif method == 'mean':
        # Extract region of interest to look for the roi
        ROI = diff[:, roi_time[0]:roi_time[1]]
        spat_pat = ROI.mean(axis=1)

    # Normalize the spatial template so all values are in the range [-1, 1]
    spat_pat /= np.max(np.abs(spat_pat))

    return spat_pat


def refine_pattern(temp_pat, method, method_params):
    """Refine a pattern.

    Refine a spatio-temporal or temporal pattern by setting some samples to
    zero.

    Parameters
    ----------
    temp_pat : 1D array (n_samples) | 2D array (n_channels, n_samples)
        The temporal pattern to refine
    method : 'zero' | 'peak-mean' | 'thres'
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
    method_params : dict
        Parameters for the chosen method. Each method uses different parameters
        taken from this dictionary. Possible parameters are:

        Used by 'zero', 'peak-mean' and 'thres':
        roi_time : tuple of ints
            The start and end time (in samples) of the time region of interest.

        Used by 'thres':
        baseline_time : tuple of ints
            The start and end time (in samples) of the baseline period (the
            period before the onset of the event marker).

    Returns
    -------
    ref_temp_pat : 1D array (n_samples) | 2D array (n_channels, n_samples)
        The refined pattern.
    """
    orig_dim = temp_pat.shape
    ref_temp_pat = np.atleast_2d(temp_pat.copy())

    if method == 'zero':
        try:
            roi_time = method_params['roi_time']
        except:
            raise ValueError('The parameter "roi_time" is missing from the '
                             'method_params dictionary.')
        ref_temp_pat -= np.min(ref_temp_pat[roi_time[0]:roi_time[1]])
        ref_temp_pat[:, :roi_time[0]] = 0
        ref_temp_pat[:, roi_time[1]:] = 0

    elif method == 'peak-mean':
        try:
            roi_time = method_params['roi_time']
        except:
            raise ValueError('The parameter "roi_time" is missing from the '
                             'method_params dictionary.')
        ref_temp_pat -= (
            np.mean(
                np.hstack(
                    (ref_temp_pat[:, :roi_time[0]],
                     ref_temp_pat[:, roi_time[1]:])
                ),
                axis=1
            )[:, np.newaxis]
        ) / 2.

        return ref_temp_pat

        peak_time = np.argmax(np.abs(ref_temp_pat[:, roi_time[0]:roi_time[1]]))
        peak_time += roi_time[0]

        # Find zero crossings
        lcross = peak_time
        while lcross > 0 and ref_temp_pat[:, lcross] > 0:
            lcross -= 1
        print lcross

        rcross = peak_time
        while rcross < len(ref_temp_pat)-1 and ref_temp_pat[:, rcross] > 0:
            rcross += 1

        # Limit temporal pattern to area between zero crossings
        ref_temp_pat[:, :lcross + 1] = 0
        ref_temp_pat[:, rcross:] = 0
        ref_temp_pat -= np.min(ref_temp_pat[:, lcross + 1:rcross])

    elif method == 'thres':
        try:
            roi_time = method_params['roi_time']
            baseline_time = method_params['baseline_time']
        except:
            raise ValueError('The parameter "roi_time"  or "baselien_time" is '
                             'missing from the method_params dictionary.')
        baseline_std = np.std(
            np.abs(ref_temp_pat[:, baseline_time[0]:baseline_time[1]])
        )
        mask = np.abs(ref_temp_pat) < 4 * baseline_std
        ref_temp_pat -= np.min(ref_temp_pat[np.logical_not(mask)])
        ref_temp_pat[mask] = 0
        ref_temp_pat[:, :roi_time[0]] = 0
        ref_temp_pat[:, roi_time[1]:] = 0

    else:
        raise ValueError("Invalid value for refine parameter. Choose one of "
                         "'zero', 'peak-mean' or 'thres'.")

    return ref_temp_pat.reshape(orig_dim)


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
        For each trial, a label indicating to which experimental condition
        the trial belongs.
    spat_bf : instance of LCMV
        The spatial beamformer that will be used to extract the ERP timecourse.
    baseline_time : tuple of ints
        The start and end time (in samples) of the baseline period (the period
        before the onset of the event marker). This period is used to
        re-baseline the signal after the spatial beamformer has been applied to
        it.
    refine : 'zero' | 'peak-mean' | 'thres' | None
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
        Defaults to None, which means no refining of the template is performed.
    refine_params : dict | None
        Parameters for the chosen refining method. Each method uses different
        parameters taken from this dictionary. Possible parameters are:

        Used by 'zero', 'peak-mean' and 'thres':
        roi_time : tuple of ints
            The start and end time (in samples) of the time region of interest.

        Used by 'thres':
        baseline_time : tuple of ints
            Copied from the baseline_time parameter given to the
            infer_temporal_pattern function.

    Returns
    -------
    temp_pat : 1D array (n_samples,)
        The temporal pattern of the ERP component.
    """
    if y.ndims == 2:
        if y.shape[1] != 1:
            raise ValueError('y should be an (n_trials, 1) array.')
        y = y.ravel()

    conditions = np.unique(y)
    if len(conditions) != 2:
        raise ValueError('There should be exactly 2 experimental conditions.')

    timecourses = spat_bf.fit_transform(X, y)

    # Re-baseline the signal
    timecourses -= np.mean(
        timecourses[..., baseline_time[0]:baseline_time[1]],
        axis=-1,
        keepdims=True,
    )

    # Use the difference timecourse as initial estimate of the temporal pattern
    temp_pat = (timecourses[y == conditions[0]].mean(axis=2) -
                timecourses[y == conditions[1]].mean(axis=2))

    if refine is not None:
        if refine_params is None:
            refine_params = dict()
        refine_params['baseline_time'] = baseline_time
        temp_pat = refine_pattern(temp_pat, refine, refine_params)

    # Normalize the temporal template so all values are in the range [-1, 1]
    temp_pat /= np.max(np.abs(temp_pat))

    return temp_pat
