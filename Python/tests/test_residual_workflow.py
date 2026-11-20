"""User-selected residual PCs retain full-image indexing and pool provenance."""

import json

import numpy as np
import pytest

from evaluations._residual_workflow import expand_pool
from evaluations.evaluate_real_extra_component import select_additions
from lumic.evaluation import select_residual_pixels


def example_scores():
    scores=np.zeros((14,14,3))
    scores[:5,:5,1]=10
    scores[8:13,8:13,1]=-10
    scores[5:8,:,0]=100  # A different PC must not influence selecting PC2.
    return scores


def test_user_selected_pc_keeps_both_signs_existing_pixels_and_provenance(tmp_path):
    scores=example_scores()
    pool=dict(indices=np.array([0,100]),eligible=np.ones((14,14),dtype=bool))
    diagnostics=dict(scores=scores,source_indices=pool["indices"].copy(),source_folder=tmp_path/"previous_fit")
    result=expand_pool(pool,diagnostics,pcs=[2],per_sign_budget=5,output_dir=tmp_path/"selection")
    added=result["added_indices"]
    assert len(added)==10 and len(result["indices"])==12
    assert np.isin(pool["indices"],result["indices"]).all()
    assert not np.isin(added,pool["indices"]).any()
    values=scores[...,1].ravel()[added]
    assert np.count_nonzero(values>0)==5 and np.count_nonzero(values<0)==5
    saved=json.loads((tmp_path/"selection/selection_settings.json").read_text())
    assert saved["pcs"]==[2]
    assert set(saved["groups"])=={"pc2_positive","pc2_negative"}
    selection=select_residual_pixels(scores,pcs=[2],eligible=pool["eligible"],
        existing_indices=pool["indices"],per_sign_budget=5,seed=2026)
    repeat=expand_pool(pool,diagnostics,selection=selection,output_dir=tmp_path/"repeat")
    np.testing.assert_array_equal(repeat["indices"],result["indices"])


@pytest.mark.parametrize("pcs",[[],[0],[4],[1,1],[True],[1.5]])
def test_invalid_user_pc_numbers_are_rejected(pcs):
    with pytest.raises(ValueError,match="one-based PC"):
        select_additions(example_scores(),np.ones((14,14),dtype=bool),2026,pcs=pcs)


def test_stale_pool_diagnostics_cannot_guide_another_round(tmp_path):
    pool=dict(indices=np.array([0,100]),eligible=np.ones((14,14),dtype=bool))
    diagnostics=dict(scores=example_scores(),source_indices=np.array([0]),source_folder=tmp_path)
    with pytest.raises(ValueError,match="current pool"):
        expand_pool(pool,diagnostics,pcs=[1],output_dir=tmp_path/"unused")
    assert not (tmp_path/"unused").exists()
