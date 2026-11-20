"""Selection preserves original pixel locations and explains exclusions."""

import numpy as np
import pytest

from lumic import select_analysis_pixels, unfold_spec_data
from lumic.utils import remove_low_intensity
from evaluations._residual_workflow import convex_hull_pool


def test_asymmetric_image_indices_ties_and_legacy_median_agree():
    values=np.array([[1,2,2],[3,4,5]])
    image=np.stack([values,values+2,values+4])
    original=image.copy()
    selection=select_analysis_pixels(image,median_quantile=.4)
    assert selection.median_threshold==4
    np.testing.assert_array_equal(selection.indices,[1,2,3,4,5])
    np.testing.assert_array_equal(selection.mask,[[False,True,True],[True,True,True]])
    with pytest.warns(FutureWarning,match="select_analysis_pixels"):
        legacy=remove_low_intensity(unfold_spec_data(image),.4)
    np.testing.assert_array_equal(unfold_spec_data(image)[selection.indices],legacy)
    np.testing.assert_array_equal(image,original)
    assert selection.exclusion_counts["below_median"]==1
    # The old interface still returns a separate retained-data copy.
    legacy[:]=0
    np.testing.assert_array_equal(image,original)


def test_invalid_zero_and_optional_exclusions_are_disjoint():
    # Spatial rows have six different input/selection situations.
    pixels=np.array([[np.nan,2],[3,np.inf],[-1,4],[0,0],[1,1],[4,4]])
    image=pixels.T.reshape(2,2,3)
    selection=select_analysis_pixels(image,median_quantile=.5)
    np.testing.assert_array_equal(selection.indices,[4,5])
    assert selection.median_threshold==1
    assert selection.exclusion_counts==dict(nonfinite=2,negative=1,zero_signal=1,below_median=0)
    coverage=selection.mask.astype(int)+sum(m.astype(int) for m in selection.exclusion_masks.values())
    np.testing.assert_array_equal(coverage,np.ones((2,3)))
    disabled=select_analysis_pixels(image)
    assert disabled.median_threshold is None
    np.testing.assert_array_equal(disabled.indices,[4,5])


def test_simulation_axis_preserves_spatial_index_order():
    image=np.arange(1,19).reshape(3,2,3)
    a=select_analysis_pixels(image,median_quantile=.5)
    b=select_analysis_pixels(np.moveaxis(image,0,-1),median_quantile=.5,spectral_axis=-1)
    np.testing.assert_array_equal(a.indices,b.indices)
    np.testing.assert_array_equal(a.mask,b.mask)


def test_all_unusable_pixels_return_explicit_empty_selection():
    image=np.zeros((3,2,3))
    result=select_analysis_pixels(image,median_quantile=.2)
    assert not result.indices.size and not result.mask.any()
    assert result.exclusion_counts["zero_signal"]==6
    with pytest.raises(ValueError,match="too few retained"):
        convex_hull_pool(image,result,npc=3)
    invalid=select_analysis_pixels(np.full((3,2,3),np.nan),median_quantile=.2)
    assert invalid.median_threshold is None and invalid.exclusion_counts["nonfinite"]==6


def test_hull_indices_map_back_to_the_retained_original_pixels():
    # npc=2 gives known extrema of normalized 1D SVD coordinates.
    image=np.array([[[0,0,1],[0,1,2]],[[0,0,0],[0,1,0]]],dtype=float)
    retained=select_analysis_pixels(image)
    pool=convex_hull_pool(image,retained,npc=2,seed=7)
    assert np.isin(pool["indices"],retained.indices).all()
    assert 4 in pool["indices"] and len(pool["indices"])==2
    assert set(pool["indices"])&{2,5}
    np.testing.assert_array_equal(pool["eligible"],retained.mask)


@pytest.mark.parametrize("q",[-.1,1.1,np.nan,True,"bad"])
def test_invalid_quantiles_fail_even_without_usable_input(q):
    with pytest.raises(ValueError,match="quantile"):
        select_analysis_pixels(np.zeros((3,2,3)),median_quantile=q)


@pytest.mark.parametrize("axis",[3,-4,True,.5])
def test_invalid_spectral_axes_are_rejected(axis):
    with pytest.raises(ValueError,match="spectral_axis"):
        select_analysis_pixels(np.ones((3,2,3)),spectral_axis=axis)


def test_legacy_median_only_retains_zeros_and_warns_at_caller():
    pixels=np.array([[0,0],[1,1],[2,2]])
    with pytest.warns(FutureWarning,match="select_analysis_pixels") as warning:
        old=remove_low_intensity(pixels,0)
    assert warning[0].filename==__file__
    assert len(old)==3
    new=select_analysis_pixels(pixels.T.reshape(2,1,3),median_quantile=0)
    np.testing.assert_array_equal(new.indices,[1,2])
