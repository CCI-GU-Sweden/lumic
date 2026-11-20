"""Known residual patterns, independent SVD and large-image layout contracts."""

import numpy as np
import pytest

from lumic.evaluation import (residual_pca, residual_pca_from_residuals,
                              residual_pca_from_factors)


def test_known_two_patterns_recover_spatial_scores_and_separate_mean_bias():
    x=np.repeat([-2.,0.,2.],2).reshape(3,2)
    y=np.tile([-1.,1.],3).reshape(3,2)
    residuals=np.stack([x+10,y-3,np.full_like(x,7)],axis=-1)
    result=residual_pca_from_residuals(residuals,n_components=2,batch_size=2)
    np.testing.assert_allclose(result.mean_residual,[10,-3,7])
    np.testing.assert_allclose(result.loadings,[[1,0,0],[0,1,0]],atol=1e-12)
    np.testing.assert_allclose(result.scores,np.stack([x,y],axis=-1),atol=1e-12)
    np.testing.assert_allclose(result.explained_variance,[3.2,1.2])
    np.testing.assert_allclose(result.explained_variance_ratio,[3.2/4.4,1.2/4.4])
    reconstructed=result.scores@result.loadings+result.mean_residual
    np.testing.assert_allclose(reconstructed,residuals)
    assert result.spatial_shape==(3,2) and result.n_samples==6


@pytest.mark.parametrize("batch_size",[1,5,65536])
def test_chunked_pca_agrees_with_independent_direct_svd(batch_size):
    residuals=np.random.default_rng(19).normal(size=(4,6,7))
    original=residuals.copy()
    flat=residuals.reshape(-1,7)
    centred=flat-flat.mean(axis=0)
    _,singular,vt=np.linalg.svd(centred,full_matrices=False)
    result=residual_pca_from_residuals(residuals,n_components=3,batch_size=batch_size)
    np.testing.assert_allclose(result.explained_variance,singular[:3]**2/23,rtol=1e-12)
    np.testing.assert_allclose(result.explained_variance_ratio,singular[:3]**2/np.sum(singular**2),rtol=1e-12)
    np.testing.assert_allclose(result.loadings.T@result.loadings,vt[:3].T@vt[:3],atol=1e-12)
    np.testing.assert_allclose(result.scores.reshape(-1,3).var(axis=0,ddof=1),result.explained_variance,rtol=1e-12)
    np.testing.assert_allclose(result.covariance,np.cov(flat,rowvar=False),atol=1e-12)
    np.testing.assert_array_equal(residuals,original)


def test_factor_image_residual_and_channel_first_interfaces_agree():
    rng=np.random.default_rng(37)
    observed=rng.normal(size=(3,5,7))
    abundance=rng.normal(size=(3,5,2))
    spectra=rng.normal(size=(2,7))
    prediction=abundance@spectra
    expected=residual_pca(observed,prediction,n_components=4,batch_size=3)
    versions=[
        residual_pca_from_residuals(observed-prediction,n_components=4),
        residual_pca_from_factors(observed,abundance,spectra,n_components=4),
        residual_pca_from_factors(np.moveaxis(observed,-1,0),abundance.reshape(-1,2),
                                  spectra,n_components=4,spectral_axis=0),
        residual_pca(np.moveaxis(observed,-1,0),np.moveaxis(prediction,-1,0),
                     n_components=4,spectral_axis=0),
    ]
    for actual in versions:
        np.testing.assert_allclose(actual.mean_residual,expected.mean_residual,atol=1e-12)
        np.testing.assert_allclose(actual.explained_variance,expected.explained_variance,atol=1e-12)
        np.testing.assert_allclose(actual.scores,expected.scores,atol=1e-12)


def test_unsigned_subtraction_retains_negative_residuals_and_inputs():
    observed=np.arange(24,dtype=np.uint16).reshape(2,3,4)
    reconstructed=observed+np.array([2,3,4,5],dtype=np.uint16)
    a,b=observed.copy(),reconstructed.copy()
    result=residual_pca(observed,reconstructed,n_components=2)
    np.testing.assert_array_equal(result.mean_residual,[-2,-3,-4,-5])
    np.testing.assert_array_equal(result.scores,np.zeros((2,3,2)))
    np.testing.assert_array_equal(observed,a)
    np.testing.assert_array_equal(reconstructed,b)


@pytest.mark.parametrize("offset",[0.,1.e9])
def test_zero_variance_has_zero_ratios_and_scores_not_nan(offset):
    result=residual_pca_from_residuals(np.full((2,3,4),offset),n_components=3,batch_size=2)
    np.testing.assert_array_equal(result.explained_variance,np.zeros(3))
    np.testing.assert_array_equal(result.explained_variance_ratio,np.zeros(3))
    np.testing.assert_array_equal(result.scores,np.zeros((2,3,3)))


def test_large_mean_does_not_erase_small_residual_variation():
    pattern=np.tile([-2.,-1.,1.,2.],3)
    residuals=np.stack([1.e9+pattern,np.full(12,1.e9)],axis=-1)
    result=residual_pca_from_residuals(residuals,n_components=1,batch_size=4)
    np.testing.assert_allclose(result.explained_variance,[np.var(pattern,ddof=1)])
    np.testing.assert_allclose(result.scores[:,0],pattern)


def test_statistics_only_avoids_score_maps_and_preserves_variance():
    residuals=np.random.default_rng(5).normal(size=(3,2,4))
    result=residual_pca_from_residuals(residuals,n_components=2,return_scores=False)
    assert result.scores is None
    np.testing.assert_allclose(result.covariance,np.cov(residuals.reshape(-1,4),rowvar=False))


@pytest.mark.parametrize("kwargs",[
    {"n_components":0},{"n_components":True},{"n_components":5},
    {"batch_size":0},{"batch_size":1.5},{"spectral_axis":3},
    {"spectral_axis":True},{"return_scores":1},
])
def test_invalid_settings_raise_clear_errors(kwargs):
    with pytest.raises(ValueError):
        residual_pca_from_residuals(np.ones((3,2,4)),**{"n_components":2,**kwargs})


@pytest.mark.parametrize("data",[
    np.ones((1,4)),np.empty((0,4)),np.ones(4),
    np.full((3,4),np.nan),np.ones((3,4),dtype=complex),
])
def test_unusable_inputs_are_rejected(data):
    with pytest.raises(ValueError):
        residual_pca_from_residuals(data,n_components=1)


def test_mismatched_reconstruction_and_factor_shapes_fail():
    with pytest.raises(ValueError,match="identical shapes"):
        residual_pca(np.ones((3,4)),np.ones((4,3)),n_components=1)
    with pytest.raises(ValueError,match="abundances must match"):
        residual_pca_from_factors(np.ones((3,4)),np.ones((2,2)),np.ones((2,4)),n_components=1)
    with pytest.raises(ValueError,match="real finite"):
        residual_pca_from_factors(np.ones((3,4)),np.full((3,2),np.inf),np.ones((2,4)),n_components=1)
