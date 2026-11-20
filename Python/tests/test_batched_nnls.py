import numpy as np
import pytest
from scipy.optimize import nnls

from lumic import mcr_als, positive_least_squares, unmixing


@pytest.mark.parametrize("n_components", [1, 3, 4, 6, 7])
@pytest.mark.parametrize("batch_size", [7, 1000])
def test_auto_solver_matches_independent_nnls_for_noisy_mixtures(
    n_components, batch_size
):
    rng = np.random.default_rng(2026)
    components = rng.uniform(0, 1, (n_components, 16))
    counts = rng.uniform(0, 100, (23, n_components))
    counts[::2, -1] = 0
    data = counts @ components + rng.normal(0, 2, (23, 16))
    # Include a zero observation and a negative observation: both can have
    # constrained optima on a boundary rather than in the full component set.
    data[0] = 0
    data[1] = -1
    expected = np.array([nnls(components.T, row)[0] for row in data])

    actual = positive_least_squares(
        data, components, batch_size=batch_size, solver="auto"
    )

    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)


def test_auto_solver_handles_components_and_observations_on_different_scales():
    components = np.array([[1, 0, 0.5, 0.2], [0.1, 1, 0.2, 0.5]])
    components *= np.array([1e-6, 1e6])[:, None]
    expected = np.array([[1e6, 1e-6], [2e-3, 0], [0, 3e2]])
    data = expected @ components

    actual = positive_least_squares(data, components, solver="auto")

    np.testing.assert_allclose(actual @ components, data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_auto_solver_matches_original_for_image_data_types(dtype):
    rng = np.random.default_rng(3)
    components = rng.uniform(0, 1, (3, 16)).astype(np.float32)
    data = rng.uniform(0, 1000, (10, 16)).astype(dtype)

    original = positive_least_squares(data, components, solver="nnls")
    accelerated = positive_least_squares(data, components, solver="auto")

    np.testing.assert_allclose(accelerated, original, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("kind", ["zero", "duplicate", "near_duplicate"])
def test_auto_solver_reconstructs_degenerate_component_matrices(kind):
    first = np.array([1.0, 0.0, 0.5, 0.2])
    second = np.array([0.1, 1.0, 0.2, 0.5])
    third = {
        "zero": np.zeros(4),
        "duplicate": first,
        "near_duplicate": first + np.array([0, 0, 1e-10, 0]),
    }[kind]
    components = np.array([first, second, third])
    data = np.array([[1, 2, 0], [0, 0, 0], [3, 0, 0]]) @ components

    actual = positive_least_squares(data, components, solver="auto")
    baseline = positive_least_squares(data, components, solver="nnls")

    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
    np.testing.assert_allclose(actual @ components, baseline @ components, atol=1e-12)
    np.testing.assert_allclose(actual @ components, data, atol=1e-9)
    if kind == "zero":
        np.testing.assert_array_equal(actual[:, 2], 0)


def test_auto_solver_returns_zero_for_all_zero_components():
    actual = positive_least_squares(
        np.ones((3, 8)), np.zeros((4, 8)), solver="auto"
    )
    np.testing.assert_array_equal(actual, np.zeros((3, 4)))


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_als_batch_solver_matches_original_factors_and_fit_history(normalization):
    rng = np.random.default_rng(14)
    spectra = rng.uniform(0.05, 1, (4, 16))
    counts = np.vstack((np.eye(4), rng.uniform(0, 3, (20, 4))))
    data = counts @ spectra
    starts = 0.9 * spectra + 0.1 * np.roll(spectra, 1, axis=1)

    baseline = mcr_als(data, starts, nIter=30, normL=normalization, solver="nnls")
    accelerated = mcr_als(data, starts, nIter=30, normL=normalization, solver="auto")

    for expected, actual in zip(baseline, accelerated):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_als_progress_callback_includes_iteration_that_converges():
    spectra = np.array([[1.0, 0, 1], [0, 1.0, 1]])
    data = np.array([[1, 0], [0, 1], [1, 1]]) @ spectra
    updates = []

    _, _, history, diagnostics = mcr_als(
        data, spectra, nIter=20, normL="L1", tol_percent=0.1,
        return_diagnostics=True, iteration_callback=updates.append,
    )

    assert diagnostics.stop_reason == "converged"
    assert updates == list(range(1, diagnostics.iterations_completed + 1))
    assert len(history) == len(updates) + 1


def test_positive_least_squares_rejects_unknown_solver():
    with pytest.raises(ValueError, match="solver must be"):
        positive_least_squares(np.ones((2, 3)), np.ones((1, 3)), solver="typo")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_auto_solver_rejects_nonfinite_observations(invalid_value):
    data = np.ones((2, 3))
    data[0, 0] = invalid_value
    with pytest.raises(ValueError):
        positive_least_squares(data, np.ones((1, 3)), solver="auto")


def test_preparation_does_not_reuse_stale_spectra_between_fitting_calls():
    rng = np.random.default_rng(401)
    spectra = rng.uniform(.05, 1, (4, 12))
    data = rng.uniform(0, 5, (37, 4)) @ spectra
    first = positive_least_squares(data, spectra, batch_size=7, solver="auto")
    # Mutate the SAME input object, as a caller may update its references.
    # A cache tied to object identity rather than the call would now be wrong.
    spectra[1] *= 4
    second = positive_least_squares(data, spectra, batch_size=7, solver="auto")
    expected = np.array([nnls(spectra.T, row)[0] for row in data])
    np.testing.assert_allclose(second, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(second[:, 1], first[:, 1] / 4, rtol=1e-10, atol=1e-10)


def test_multibatch_preparation_preserves_vanished_and_zero_observations():
    rng = np.random.default_rng(402)
    spectra = rng.uniform(.05, 1, (4, 12))
    spectra[2] = 0
    expected = rng.uniform(0, 5, (37, 4))
    expected[:, 2] = 0
    expected[[0, 7, 14, 36]] = 0  # Include batch boundaries and a partial batch.
    data = expected @ spectra
    result = positive_least_squares(data, spectra, batch_size=7, solver="auto")
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_failed_preparation_retains_legacy_solver(monkeypatch):
    spectra = np.array([[1., 0., 1.], [0., 1., 1.]])
    data = np.array([[2., 1., 3.], [1., 0., 1.], [0., 0., 0.]])
    expected = positive_least_squares(data, spectra, batch_size=1, solver="nnls")

    def fail_preparation(*args, **kwargs):
        raise np.linalg.LinAlgError("unusable decomposition")

    monkeypatch.setattr(unmixing, "_prepare_small_batch", fail_preparation)
    result = positive_least_squares(data, spectra, batch_size=1, solver="auto")
    np.testing.assert_allclose(result, expected)


def test_tall_basis_streams_without_retaining_all_subset_matrices():
    from lumic._nnls import prepare_small_batch, solve_prepared_batch
    rng = np.random.default_rng(403)
    spectra = rng.uniform(.05, 1, (4, 20000))
    expected = np.array([[3., 1., 0., 2.], [0., 0., 0., 0.], [1., 0., 2., 4.]])
    # Request reuse for a tall basis, as in a spectral update with many pixels.
    # Its retained matrices would exceed the budget. Stream rather than
    # allocating them, while still solving the same nonnegative problem.
    prepared = prepare_small_batch(spectra, cache_subsets=True)
    assert prepared.subsets is None
    result = solve_prepared_batch(expected @ spectra, prepared)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
