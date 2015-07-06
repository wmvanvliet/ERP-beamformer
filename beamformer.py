import numpy as np
from sklearn.covariance import EmpericalCovariance, ShrunkCovariance, OAS, LedoitWolf
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

    normalize : bool (default: True)
        Whether to remove the channel mean before applying the filter.
    '''
    def __init__(self, template, shrinkage='oas', normalize=True):
        self.template = template
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.normalize = normalize

        if normalize:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpericalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def fit(self, X, y=None):
        if self.normalize:
            X = X - X.mean(axis=0)

        # Concatenate trials
        cont_eeg = np.transpose(X, [0,2,1]).reshape((X.shape[0], -1))

        # Calculate spatial covariance matrix
        c = self.cov().fit(cont_eeg.T)
        sigma_x_i = c.precision_
        
        # Compute spatial LCMV filter
        self.W = sigma_x_i.dot(self.template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(
                reduce(np.dot, [self.template.T, sigma_x_i, self.template])
            )
        )

    def transform(self, X):
        if self.normalize:
            X = X - X.mean(axis=0)

        nchannels = self.W.shape[1]
        nsamples = X.shape[1]
        ntrials = X.shape[2]

        new_X = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            new_X[:, :, i] = np.dot(self.W.T, X[:,:,i])

        return new_X


class stLCMV(BaseEstimator, TransformerMixin):
    '''
    spatio-temporal LCMV Beamformer operating on a spatio-temporal template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    normalize : bool (default: True)
        Whether to remove the mean before applying the filter.
    '''
    def __init__(self, template, shrinkage='oas', normalize=True):
        self.template = template
        self.template = np.atleast_2d(template)
        self.normalize = normalize

        if normalize:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpericalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def center(self, X):
        data_mean = X.reshape(-1, X.shape[2]).mean(axis=1)
        data_mean = data_mean.reshape(X.shape[:2] + (1,))
        return X - data_mean

    def fit(self, X, y):
        if self.normalize:
            X = self.center(X)

        nsamples, ntrials = X.shape[1:]
        template = self.template[:, :nsamples]

        c = self.cov().fit(X.reshape(-1, ntrials).T)
        sigma_x_i = c.precision_

        template = self.template.flatten()[:, np.newaxis]
        self.W = sigma_x_i.dot(template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(reduce(np.dot, [template.T, sigma_x_i, template]))
        )

    def transform(self, X):
        if self.normalize:
            X = self.center(X)

        ntrials = X.shape[2]
        new_X = self.W.T.dot(X.reshape(-1, ntrials))
        return new_X
