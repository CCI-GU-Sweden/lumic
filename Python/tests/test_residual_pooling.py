"""Selection contracts and descriptive spatial metrics for the bounded pilot."""

import numpy as np
import pytest

from evaluations._residual_pooling import addition_pools, candidate_masks, spatial_diagnostics


def test_opening_removes_isolated_peak_but_retains_supported_region():
    scores=np.zeros((9,9))
    scores[2:7,2:7]=10
    scores[0,8]=100  # Largest residual, but spatially isolated.
    masks=candidate_masks(scores,np.ones_like(scores,dtype=bool),quantile=.7)
    assert masks['high'][0,8]
    assert not masks['spatial'][0,8]
    assert masks['spatial'][4,4]
    assert np.all(~masks['spatial'] | masks['high'])
    assert masks['labels'].max()==1


def test_hull_points_do_not_create_holes_before_opening_and_are_not_added_twice():
    scores=np.zeros((7,7))
    scores[2:5,2:5]=10
    center=np.ravel_multi_index((3,3),scores.shape)
    eligible=np.ones_like(scores,dtype=bool)
    eligible[3,3]=False
    masks=candidate_masks(scores,eligible,quantile=.8)
    assert masks['spatial'][3,3]  # Opening sees the complete region.
    result=addition_pools(np.array([center,center]),eligible,masks,10,2026)
    assert result['actual_budget']==4  # Surviving cross minus its existing hull center.
    assert result['unavailable']==[]
    for pool in result['pools'].values():
        assert len(pool)==5
        assert center in pool
        assert len(np.unique(pool))==len(pool)
    spatial_additions=np.setdiff1d(result['pools']['spatial_residual'],[center])
    assert masks['spatial'].flat[spatial_additions].all()


def test_random_additions_repeat_and_never_sample_outside_candidates():
    eligible=np.ones((8,8),dtype=bool)
    hull=np.array([0,63])
    masks={'high':np.indices((8,8))[0]<4,'spatial':np.indices((8,8))[0]<2}
    first=addition_pools(hull,eligible,masks,5,10)
    second=addition_pools(hull,eligible,masks,5,10)
    for name,pool in first['pools'].items():
        np.testing.assert_array_equal(pool,second['pools'][name])
        assert len(pool)==7
        assert set(hull)<=set(pool)
        if name!='uniform':
            mask=masks['high' if name=='high_residual' else 'spatial']
            assert mask.flat[np.setdiff1d(pool,hull)].all()
    np.testing.assert_array_equal(eligible,np.ones((8,8),dtype=bool))


def test_uninformative_residuals_skip_guidance_without_replacing_it():
    eligible=np.ones((5,5),dtype=bool)
    masks=candidate_masks(np.ones((5,5)),eligible,informative=False)
    result=addition_pools(np.array([0]),eligible,masks,4,1)
    assert masks['reason']=='residual_below_floor'
    assert result['unavailable']==['high_residual','spatial_residual']
    assert list(result['pools'])==['uniform']
    assert result['actual_budget']==4


def test_no_spatial_support_is_unavailable_not_filled_with_isolated_pixels():
    scores=np.zeros((5,5))
    scores[1,1]=10
    masks=candidate_masks(scores,np.ones_like(scores,dtype=bool))
    result=addition_pools(np.array([0]),np.ones_like(scores,dtype=bool),masks,4,1)
    assert result['unavailable']==['spatial_residual']
    assert result['actual_budget']==1
    np.testing.assert_array_equal(result['pools']['high_residual'],[0,6])


def test_spatial_metric_detects_signed_alternation_and_ignores_near_zero_signal_errors():
    checker=1-2*(np.indices((8,8)).sum(axis=0)%2)
    observed=np.full((8,8,2),100.)
    residuals=np.stack([checker,2*checker],axis=-1)
    stats=spatial_diagnostics(observed,residuals)
    assert stats['residual_above_floor']
    np.testing.assert_allclose(stats['neighbor_correlation'],-1,atol=1e-12)
    scaled=spatial_diagnostics(observed*10,residuals*10)
    np.testing.assert_allclose(scaled['relative_l2'],stats['relative_l2'])
    np.testing.assert_allclose(scaled['neighbor_correlation'],stats['neighbor_correlation'])
    np.testing.assert_allclose(scaled['mean_rms'],10*stats['mean_rms'])
    negligible=spatial_diagnostics(observed,residuals*1e-12)
    assert not negligible['residual_above_floor']
    assert np.isnan(negligible['neighbor_correlation'])
    assert np.isnan(negligible['rms_signal_correlation'])


def test_constant_residual_offset_has_magnitude_but_no_centered_spatial_correlation():
    stats=spatial_diagnostics(np.full((4,4,1),100.),np.ones((4,4,1)))
    assert stats['mean_rms']==1
    assert np.isnan(stats['neighbor_correlation'])


@pytest.mark.parametrize('quantile,radius',[(1,1),(-.1,1),(np.nan,1),(.8,-1),(.8,.5),(.8,True)])
def test_invalid_segmentation_parameters_rejected(quantile,radius):
    with pytest.raises(ValueError):
        candidate_masks(np.ones((3,3)),np.ones((3,3),dtype=bool),quantile=quantile,opening_radius=radius)


def test_invalid_hull_index_and_spatial_shapes_rejected():
    masks={'high':np.ones((3,3),dtype=bool),'spatial':np.ones((3,3),dtype=bool)}
    with pytest.raises(ValueError):
        addition_pools(np.array([9]),np.ones((3,3),dtype=bool),masks,2,1)
    with pytest.raises(ValueError):
        spatial_diagnostics(np.ones((3,3,2)),np.ones((3,3)))
