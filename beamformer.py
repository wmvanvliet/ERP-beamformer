import psychic
import golem
from golem.stat import lw_cov
from numpy import cov
from golem.nodes.basenode import BaseNode
from psychic.nodes.spatialfilter import BaseSpatialFilter

import numpy as np


class SpatialBeamformer(BaseSpatialFilter):
    '''
    Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels)
       Spatial activation pattern of the component to extract.

    reg : float (default: 0.2)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=0.2, cov_func=cov):
        BaseSpatialFilter.__init__(self, 1)
        self.template = template
        self.reg = reg
        self.cov_func = cov_func
        self.template = np.asarray(template).flatten()[:, np.newaxis]

    def train_(self, d):
        sigma_x = np.mean(
            [self.cov_func(t) for t in np.rollaxis(d.ndX, -1)],
            axis=0)
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        self.W = sigma_x_i.dot(self.template)


class TemplateBeamformer(BaseNode):
    '''
    Beamformer operating on a template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Activation pattern of the component to extract.

    reg : float (default: 0.2)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=0.2, cov_func=cov):
        BaseNode.__init__(self)
        self.template = template
        self.reg = reg
        self.cov_func = cov_func
        self.template = np.atleast_2d(template)

    def train_(self, d):
        nsamples, ntrials = d.ndX.shape[1:]
        template = self.template[:, :nsamples]
        sigma_x = self.cov_func(d.ndX.reshape(-1, ntrials))
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        self.W = sigma_x_i.dot(template.flatten()).ravel()

    def apply_(self, d):
        ntrials = d.ndX.shape[2]
        y = self.W.dot(d.ndX.reshape(-1, ntrials))
        y -= np.mean(y)
        X = np.c_[-y, y].T
        feat_lab = None
        return golem.DataSet(ndX=X, feat_lab=feat_lab, default=d)
