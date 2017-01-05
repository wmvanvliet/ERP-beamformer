"""
Classes for spatial and spatio-temporal beamforming. The API is designed to be
interoperable with scikit-learn.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
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
        WARNING: only set to False if the data has been pre-centered. Applying
        the filter to un-centered data may result in inaccuracies.

    Attributes
    ----------
    W_ : 2D array (n_channels, 1)
        Column vector containing the filter weights.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.center = center

        if center:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS()
        elif shrinkage == 'lw':
            self.cov = LedoitWolf()
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance()
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)
        else:
            raise ValueError('Invalid value for shrinkage parameter.')

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
        WARNING: only set to False if the data has been pre-centered. Applying
        the filter to un-centered data may result in inaccuracies.

    Attributes
    ----------
    W_ : 2D array (n_channels * n_samples, 1)
        Column vector containing the filter weights.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        self.template = template
        self.template = np.atleast_2d(template)
        self.center = center

        if center:
            self.template -= np.mean(self.template)

        if shrinkage == 'oas':
            self.cov = OAS()
        elif shrinkage == 'lw':
            self.cov = LedoitWolf()
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance()
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)
        else:
            raise ValueError('Invalid value for shrinkage parameter.')

    def _center(self, X):
        data_mean = X.reshape(X.shape[0], -1).mean(axis=1)
        return X - data_mean[:, np.newaxis, np.newaxis]

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

        c = self.cov.fit(X.reshape(n_trials, -1))
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
