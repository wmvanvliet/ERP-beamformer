from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import itertools
from sklearn.covariance import (EmpiricalCovariance, ShrunkCovariance, OAS,
                                LedoitWolf)

from erp_beamformer import LCMV, stLCMV

n_channels = 7
n_samples = 12
n_trials = 5


def _gen_spatial_pattern():
    return np.array([1, 4, 2, -3, 5, 0, 0]).astype(float)


def _gen_temporal_pattern():
    return np.array([0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0]).astype(float)


def _gen_spat_temp_pattern():
    return np.outer(_gen_spatial_pattern(), _gen_temporal_pattern())


def _gen_trials():
    y = np.array([1, 2, 4, -2, 3]).astype(float)
    return np.einsum('i,jk->ijk', y, _gen_spat_temp_pattern()), y


def test_lcmv():
    """Test the LCMV class."""

    # Generate some mock data without any noise
    spat_pat = _gen_spatial_pattern()
    X, y = _gen_trials()

    # All possible combinations of parameters
    parameters = itertools.product(
        ['none', 'oas', 'lw', 0, 0.5, 1],
        [True, False]
    )

    # Test normal operation
    for shrinkage, center in parameters:
        filt = LCMV(spat_pat, shrinkage=shrinkage, center=center)

        # Test result of the filter
        assert_allclose(filt.fit_transform(X, y),
                        np.outer(y, _gen_temporal_pattern()))

        # Test if weight matrix has been properly stored
        assert_equal(filt.W_.shape, (n_channels, 1))

        # Test if shrinkage parameter was properly interpreted
        if shrinkage == 'none':
            assert_true(isinstance(filt.cov, EmpiricalCovariance))
        elif shrinkage == 'oas':
            assert_true(isinstance(filt.cov, OAS))
        elif shrinkage == 'lw':
            assert_true(isinstance(filt.cov, LedoitWolf))
        elif type(shrinkage) == float:
            assert_true(isinstance(filt.cov, ShrunkCovariance))
            assert_equal(filt.cov.shrinkage, shrinkage)

    # Invalid shrinkage parameter
    assert_raises(ValueError, LCMV, spat_pat, 'invalid')


def test_stlcmv():
    """Test the stLCMV class."""

    # Generate some mock data without any noise
    spat_temp_pat = _gen_spat_temp_pattern()
    X, y = _gen_trials()
    y = y[:, np.newaxis]

    # All possible combinations of parameters
    parameters = itertools.product(
        ['none', 'oas', 'lw', 0, 0.5, 1],
        [True, False]
    )

    # Test normal operation
    for shrinkage, center in parameters:
        filt = stLCMV(spat_temp_pat, shrinkage=shrinkage, center=center)

        # Test result of the filter
        if center:
            assert_allclose(filt.fit_transform(X, y), y)
        else:
            # Result may be off, use correlation instead
            assert_allclose(np.corrcoef(filt.fit_transform(X, y).T, y.T), 1)
                        
        # Test if weight matrix has been properly stored
        assert_equal(filt.W_.shape, (n_channels * n_samples, 1))

        # Test if shrinkage parameter was properly interpreted
        if shrinkage == 'none':
            assert_true(isinstance(filt.cov, EmpiricalCovariance))
        elif shrinkage == 'oas':
            assert_true(isinstance(filt.cov, OAS))
        elif shrinkage == 'lw':
            assert_true(isinstance(filt.cov, LedoitWolf))
        elif type(shrinkage) == float:
            assert_true(isinstance(filt.cov, ShrunkCovariance))
            assert_equal(filt.cov.shrinkage, shrinkage)

    # Invalid shrinkage parameter
    assert_raises(ValueError, stLCMV, spat_temp_pat, 'invalid')
