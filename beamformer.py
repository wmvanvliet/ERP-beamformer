from basenode import BaseNode
from ..dataset import DataSet
from ..trials import baseline, concatenate_trials
import numpy as np
from ..stat import lw_cov
from psychic.nodes.spatialfilter import SpatialFilter
from sklearn.grid_search import GridSearchCV
from sklearn.covariance import ShrunkCovariance, OAS

class SpatialBeamformer(SpatialFilter):
    '''
    LCMV Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels)
       Spatial activation pattern of the component to extract.

    reg : float (default: 1e-5)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=None, cov_func=None, normalize=True):
        SpatialFilter.__init__(self, 1)
        self.template = template
        self.cov_func = cov_func
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.normalize = normalize
        if reg is None:
            self.reg = np.r_[0, np.logspace(-5, 0, 6)]
        else:
            self.reg = reg

        if normalize:
            self.template -= self.template.mean()

    def train_(self, d):
        if self.normalize:
            d = DataSet(d.data - d.data.mean(axis=0), default=d)
            d = baseline(d)

        # Calculate spatial covariance matrix
        if type(self.reg) == list or type(self.reg) == np.ndarray:
            #gs = GridSearchCV(ShrunkCovariance(assume_centered=True), [{'shrinkage': self.reg}], cv=2)
            #c = gs.fit(concatenate_trials(d).X).best_estimator_
            c = OAS(assume_centered=True).fit(concatenate_trials(d).X)
            self.log.info('Estimated shrinkage: %f' % c.shrinkage_)
            self.reg = c.shrinkage_
        else:
            c = ShrunkCovariance(assume_centered=True, shrinkage=self.reg).fit(concatenate_trials(d).X)
        sigma_x_i = c.precision_
        
        #sigma_x = np.mean(
        #    [self.cov_func(t) for t in np.rollaxis(d.data, -1)],
        #    axis=0) / (len(d) - 1)
        #sigma_x += self.reg * np.eye(sigma_x.shape[0])
        #sigma_x_i = np.linalg.inv(sigma_x)
        self.W = sigma_x_i.dot(self.template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(reduce(np.dot, [self.template.T, sigma_x_i, self.template]))
        )

    def apply_(self, d):
        if self.normalize:
            d = DataSet(d.data - d.data.mean(axis=0), default=d)
            d = baseline(d)
        return SpatialFilter.apply_(self, d)


class TemplateBeamformer(BaseNode):
    '''
    LCMV Beamformer operating on a template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Activation pattern of the component to extract.

    reg : float (default: 1e-5)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=None, cov_func=None, normalize=True):
        BaseNode.__init__(self)
        self.template = template
        self.reg = reg
        self.cov_func = cov_func
        self.template = np.atleast_2d(template)
        self.normalize = normalize
        if self.reg is None:
            self.reg = np.r_[0, np.linspace(0, 1, 6)]
        else:
            self.reg = reg

    def center(self, d):
        data_mean = d.data.reshape(-1, len(d)).mean(axis=1)
        data_mean = data_mean.reshape(d.feat_shape + (1,))
        return DataSet(d.data - data_mean, default=d)

    def train_(self, d):
        if self.normalize:
            d = self.center(d)

        nsamples, ntrials = d.data.shape[1:]
        template = self.template[:, :nsamples]
        #sigma_x = self.cov_func(d.data.reshape(-1, ntrials))
        #sigma_x += self.reg * np.eye(sigma_x.shape[0])
        #sigma_x_i = np.linalg.inv(sigma_x)

        if type(self.reg) == list or type(self.reg) == np.ndarray:
            #gs = GridSearchCV(ShrunkCovariance(assume_centered=True), [{'shrinkage': self.reg}], cv=2)
            #c = gs.fit(d.data.reshape(-1, ntrials).T).best_estimator_
            c = OAS(assume_centered=True).fit(d.data.reshape(-1, ntrials).T)
            self.log.info('Estimated shrinkage: %f' % c.shrinkage_)
            self.reg = c.shrinkage_
        else:
            c = ShrunkCovariance(assume_centered=True, shrinkage=self.reg).fit(d.data.reshape(-1, ntrials).T)
        sigma_x_i = c.precision_

        template = self.template.flatten()[:, np.newaxis]
        self.W = sigma_x_i.dot(template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(reduce(np.dot, [template.T, sigma_x_i, template]))
        )

    def apply_(self, d):
        if self.normalize:
            d = self.center(d)

        ntrials = d.data.shape[2]
        X = self.W.T.dot(d.data.reshape(-1, ntrials))
        return DataSet(data=X, feat_lab=None, default=d)
