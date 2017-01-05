from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from erp_beamformer import (infer_spatial_pattern, infer_temporal_pattern,
                            refine_pattern, LCMV)

n_channels = 7
n_samples = 12
n_trials = 6


def _gen_spatial_pattern(pattern=0):
    return np.array([
        [1, 4, 2, -3, 5, 0, 0],
        [3, 1, 4, 5, 2, -2, -3],
    ])[pattern].astype(float)


def _gen_temporal_pattern(pattern=0):
    return np.array([
        [0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
        [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ])[pattern].astype(float)


def _gen_spat_temp_pattern(spat_pattern=0, temp_pattern=0):
    return np.outer(_gen_spatial_pattern(spat_pattern),
                    _gen_temporal_pattern(temp_pattern))


def _gen_trials(spat_pattern=0, temp_pattern=0):
    y = np.array([-1, 1, -1, 1, 1, -1]).astype(float)
    X = np.einsum(
        'i,jk->ijk',
        y,
        _gen_spat_temp_pattern(spat_pattern, temp_pattern)
    )
    return X, y


def test_infer_spatial_pattern():
    """Test the infer_spatial_pattern function."""
    # The default
    spat_pat = _gen_spatial_pattern()
    X, y = _gen_trials()
    inferred_pattern = infer_spatial_pattern(X, y)
    assert_allclose(inferred_pattern, -spat_pat / np.abs(spat_pat).max())

    inferred_pattern = infer_spatial_pattern(X, np.array(y)[:, np.newaxis])
    assert_allclose(inferred_pattern, -spat_pat / np.abs(spat_pat).max())

    # method='peak'
    spat_pat = _gen_spatial_pattern(0)  # First pattern should be detected
    spat_pat /= np.abs(spat_pat).max()
    X, y = _gen_trials(0, 0)
    X += _gen_trials(1, 1)[0]
    inferred_pattern = infer_spatial_pattern(X, y, method='peak')
    assert_allclose(inferred_pattern, -spat_pat)

    # test ROI channels
    spat_pat = _gen_spatial_pattern(1)  # Second pattern should be detected
    spat_pat /= np.abs(spat_pat).max()
    inferred_pattern = infer_spatial_pattern(X, y, method='peak',
                                             roi_channels=[5, 6])
    assert_allclose(inferred_pattern, -spat_pat)

    # test ROI time
    spat_pat = _gen_spatial_pattern(1)  # Second pattern should be detected
    spat_pat /= np.abs(spat_pat).max()
    inferred_pattern = infer_spatial_pattern(X, y, method='peak',
                                             roi_time=(0, 6))
    assert_allclose(inferred_pattern, -spat_pat)

    # method='mean'
    diff = X[y == -1].mean(axis=0) - X[y == 1].mean(axis=0)
    spat_pat = diff[:, 2:4].mean(axis=1)
    spat_pat /= np.abs(spat_pat).max()
    inferred_pattern = infer_spatial_pattern(X, y, method='mean',
                                             roi_time=(2, 4))
    assert_allclose(inferred_pattern, spat_pat)

    # ROI channels should have no effect when method='mean'
    inferred_pattern = infer_spatial_pattern(X, y, method='mean',
                                             roi_time=(2, 4),
                                             roi_channels=[0, 2, 3])
    assert_allclose(inferred_pattern, spat_pat)

    # Invalid inputs
    assert_raises(ValueError, infer_spatial_pattern, X, range(n_trials))
    assert_raises(ValueError, infer_spatial_pattern, X, y[:3])
    assert_raises(ValueError, infer_spatial_pattern, X, [[1, 2], [1, 2]])


def test_infer_temporal_pattern():
    """Test the infer_temporal_pattern function."""
    # The default
    spat_pat = _gen_spatial_pattern()
    temp_pat = _gen_temporal_pattern()
    temp_pat /= np.abs(temp_pat).max()
    X, y = _gen_trials()
    inferred_pattern = infer_temporal_pattern(X, y, LCMV(spat_pat), (0, 1))
    assert_allclose(inferred_pattern, -temp_pat)

    # Refining the template
    temp_pat[:4] = 0
    temp_pat[8:] = 0
    temp_pat /= np.abs(temp_pat).max()
    inferred_pattern = infer_temporal_pattern(
        X, y, LCMV(spat_pat), (0, 1),
        refine='zero', refine_params=dict(roi_time=(4, 8))
    )
    assert_allclose(inferred_pattern, -temp_pat)

    # Test baseline
    temp_pat = _gen_temporal_pattern()
    temp_pat -= temp_pat[3:5].mean()
    temp_pat /= np.abs(temp_pat).max()
    inferred_pattern = infer_temporal_pattern(X, y, LCMV(spat_pat), (3, 5))
    assert_allclose(inferred_pattern, -temp_pat)


def test_refine_pattern():
    """The the refine_pattern function."""
    temp = np.array([0., 1., 4., 1., 0., 0.])

    assert_allclose(
        refine_pattern(temp, 'zero', dict(roi_time=(1, 3))),
        [0, 1, 4, 0, 0, 0],
    )
    assert_allclose(
        refine_pattern(temp, 'zero', dict(roi_time=(0, 0))),
        [0, 0, 0, 0, 0, 0],
    )
    assert_allclose(
        refine_pattern(np.c_[temp, temp].T, 'zero', dict(roi_time=(1, 3))),
        [[0, 1, 4, 0, 0, 0], [0, 1, 4, 0, 0, 0]],
    )
    assert_allclose(
        refine_pattern(temp, 'peak-mean', dict(roi_time=(2, 3))),
        [0, 0.8, 3.8, 0.8, 0, 0],
    )
    assert_allclose(
        refine_pattern(-temp, 'peak-mean', dict(roi_time=(2, 3))),
        [0, -0.8, -3.8, -0.8, 0, 0],
    )
    assert_allclose(
        refine_pattern(temp, 'peak-mean', dict(roi_time=(3, 4))),
        [0, 0.5, 3.5, 0.5, 0, 0],
    )
    assert_allclose(
        refine_pattern(np.c_[temp, temp].T, 'peak-mean',
                       dict(roi_time=(3, 4))),
        [[0, 0.5, 3.5, 0.5, 0, 0], [0, 0.5, 3.5, 0.5, 0, 0]],
    )
    assert_allclose(
        refine_pattern(temp, 'thres',
                       dict(roi_time=(0, 5), baseline_time=(0, 1))),
        [0, 1, 4, 1, 0, 0]
    )
    assert_allclose(
        refine_pattern(temp, 'thres',
                       dict(roi_time=(0, 5), baseline_time=(0, 2))),
        [0, 0, 4, 0, 0, 0]
    )
    assert_allclose(
        refine_pattern(temp, 'thres',
                       dict(roi_time=(2, 4), baseline_time=(0, 1))),
        [0, 0, 4, 1, 0, 0]
    )

    # Invalid inputs
    assert_raises(ValueError, refine_pattern, temp, 'zero', dict())
    assert_raises(ValueError, refine_pattern, temp, 'peak-mean', dict())
    assert_raises(ValueError, refine_pattern, temp, 'thres', dict())
    assert_raises(ValueError, refine_pattern, temp, 'peak-mean',
                  dict(roi_time=(3, 3)))
    assert_raises(ValueError, refine_pattern, temp, 'peak-mean',
                  dict(roi_time=(3, 2)))
