"""Public residual selection: spatial support, provenance and seeded sampling."""

import numpy as np
import pytest

from lumic.evaluation import select_residual_pixels


def patterns():
    scores=np.zeros((14,14,2))
    scores[1:6,1:6,0]=10
    scores[8:13,8:13,0]=-10
    scores[...,1]=scores[...,0]
    return scores


def test_both_signs_original_coordinates_exclusions_and_repeatability(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    scores=patterns();before=scores.copy()
    eligible=np.ones((14,14),dtype=bool);eligible[3,3]=False
    existing=np.array([31,32,31])
    result=select_residual_pixels(scores,pcs=[1],eligible=eligible,existing_indices=existing,
                                  per_sign_budget=5,seed=2026)
    repeat=select_residual_pixels(scores,pcs=[1],eligible=eligible,existing_indices=existing,
                                  per_sign_budget=5,seed=2026)
    np.testing.assert_array_equal(result.indices,repeat.indices)
    assert len(result.indices)==10 and len(np.unique(result.indices))==10
    assert not np.isin(result.indices,existing).any() and 3*14+3 not in result.indices
    values=scores[...,0].ravel()[result.indices]
    assert np.count_nonzero(values>0)==5 and np.count_nonzero(values<0)==5
    for name,group in result.groups.items():
        assert group["selected"]==len(group["indices"])
        assert result.candidate_masks[name].ravel()[group["indices"]].all()
    np.testing.assert_array_equal(scores,before)
    assert not eligible[3,3] and eligible[2,3]  # Input mask was not changed.
    np.testing.assert_array_equal(existing,[31,32,31])
    assert not list(tmp_path.iterdir())  # Calculation does not save files.


def test_opening_removes_isolated_peak_but_pool_holes_do_not_erase_support():
    scores=np.zeros((9,9,1));scores[2:5,2:5,0]=10;scores[7,7,0]=100
    centre=3*9+3
    result=select_residual_pixels(scores,pcs=[1],existing_indices=[centre],
        per_sign_budget=30,score_quantile=.8,opening_radius=1,seed=1)
    assert result.threshold_masks["pc1_positive"][7,7]
    assert not result.opened_masks["pc1_positive"][7,7]
    assert result.opened_masks["pc1_positive"][3,3]
    assert not result.candidate_masks["pc1_positive"][3,3]
    np.testing.assert_array_equal(result.indices,[2*9+3,3*9+2,3*9+4,4*9+3])


def test_overlapping_pcs_do_not_duplicate_samples_and_report_shortfalls():
    result=select_residual_pixels(patterns(),pcs=[1,2],per_sign_budget=100,seed=8)
    assert len(result.indices)==42  # Each 5x5 region opens to 21 pixels.
    assert result.groups["pc1_positive"]["selected"]==21
    assert result.groups["pc2_positive"]["selected"]==0
    assert result.groups["pc2_positive"]["reason"]=="overlap_exhausted"
    assert result.groups["pc2_positive"]["candidate_count"]==21


@pytest.mark.parametrize("empty_eligibility",[False,True])
def test_empty_regions_have_no_unrelated_replacement_pixels(empty_eligibility):
    scores=np.zeros((5,6,1)) if not empty_eligibility else np.ones((5,6,1))
    eligible=np.full((5,6),not empty_eligibility,dtype=bool)
    result=select_residual_pixels(scores,pcs=[1],eligible=eligible,seed=3)
    assert not result.indices.size
    assert all(group["selected"]==0 for group in result.groups.values())
    assert all(not mask.any() for mask in result.candidate_masks.values())


def test_no_seed_records_entropy_that_replays_the_selection():
    result=select_residual_pixels(patterns(),pcs=[2],per_sign_budget=4,seed=None)
    replay=select_residual_pixels(patterns(),pcs=[2],per_sign_budget=4,seed=result.settings["seed_entropy"])
    np.testing.assert_array_equal(result.indices,replay.indices)
    assert result.settings["seed"] is None


def test_quantile_includes_opposite_sign_zeros_and_retains_threshold_ties():
    scores=np.array([[[1.],[2.],[3.],[-1.]],[[0.],[0.],[0.],[0.]]])
    result=select_residual_pixels(scores,pcs=[1],score_quantile=.5,
                                  per_sign_budget=10,opening_radius=0,seed=2)
    assert result.groups["pc1_positive"]["threshold"]==0
    np.testing.assert_array_equal(result.indices,[0,1,2,3])
    assert not result.threshold_masks["pc1_positive"][1].any()


@pytest.mark.parametrize("kwargs",[
    {"pcs":[0]},{"pcs":[3]},{"pcs":[1,1]},{"pcs":[True]},
    {"per_sign_budget":0},{"opening_radius":-1},{"score_quantile":1},
    {"seed":-1},{"seed":True},{"existing_indices":[196]},
    {"existing_indices":[1.5]},{"eligible":np.ones((14,14),dtype=int)},
])
def test_invalid_selection_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        select_residual_pixels(patterns(),**{"pcs":[1],**kwargs})
