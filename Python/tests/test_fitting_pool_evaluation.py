"""Pool membership rules that make the numerical comparison interpretable."""

import numpy as np
import pytest
import json

from evaluations.evaluate_fitting_pool_expansion import expanded_pools


def test_pool_expansion_preserves_hull_and_nested_distinct_additions():
    pools = expanded_pools(10, np.array([7, 2, 7]), (0, 2, 5, 20), 2026)
    np.testing.assert_array_equal(pools[0], [2, 7])
    assert [len(p) for p in pools.values()] == [2, 4, 7, 10]
    for small, large in zip(list(pools.values()), list(pools.values())[1:]):
        assert set(small) <= set(large)
    np.testing.assert_array_equal(pools[20], np.arange(10))
    repeat = expanded_pools(10, np.array([2, 7]), (0, 2, 5, 20), 2026)
    for budget in pools:
        np.testing.assert_array_equal(pools[budget], repeat[budget])


@pytest.mark.parametrize("hull,budgets", [([-1], (2,)), ([10], (2,)),
    ([1.5], (2,)), ([], (2,)), ([1], (-1,)), ([1], (2, 1)), ([1], (2, 2))])
def test_invalid_pool_definitions_rejected(hull, budgets):
    with pytest.raises(ValueError):
        expanded_pools(10, np.array(hull), budgets, 0)


def test_iteration_comparison_rejects_changed_pixel_pools(tmp_path):
    from evaluations.evaluate_fitting_pool_robustness import compare_iteration_caps

    old_dir = tmp_path / "iterations_500"
    old_dir.mkdir()
    selection = {"blur_fwhm_um": 0, "pool": "hull", "sampling_seed": -1,
                 "added_budget": 0, "fit_indices": [1, 2],
                 "initial_spectra": [[1, 0]], "selected_start_indices": [1]}
    old = {"max_iterations": 500, "fit_selection": [selection]}
    new = {"max_iterations": 1000,
           "fit_selection": [{**selection, "fit_indices": [1, 3]}]}
    (old_dir / "settings.json").write_text(json.dumps(old))
    (tmp_path / "settings.json").write_text(json.dumps(new))
    with pytest.raises(ValueError, match="fit_indices"):
        compare_iteration_caps(tmp_path)
