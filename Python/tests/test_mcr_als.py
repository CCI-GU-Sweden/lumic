import numpy as np
import pytest
from lumic import lof, mcr_als, normalize_components, unmixing


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_reconstructs_known_mixture(normalization):
    spectra = np.array([[2.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    concentrations = np.array(
        [[1.0, 2.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    mixed_data = concentrations @ spectra

    fitted_concentrations, fitted_spectra, residuals = mcr_als(
        mixed_data, spectra, nIter=2, normL=normalization
    )
    reconstructed = fitted_concentrations @ fitted_spectra

    assert fitted_concentrations.shape == concentrations.shape
    assert fitted_spectra.shape == spectra.shape
    assert residuals.shape == (3,)
    assert np.all(fitted_concentrations >= 0)
    assert np.all(fitted_spectra >= 0)
    if normalization == "L1":
        np.testing.assert_allclose(np.sum(fitted_spectra, axis=1), 1, atol=1e-12)
    elif normalization == "L2":
        np.testing.assert_allclose(
            np.linalg.norm(fitted_spectra, axis=1), 1, atol=1e-12
        )
    np.testing.assert_allclose(reconstructed, mixed_data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        residuals[-1], lof(mixed_data, reconstructed, method="sqr"), atol=1e-12
    )


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_keeps_absent_component_at_zero(normalization):
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    mixed_data = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]
    )

    concentrations, fitted_spectra, residuals = mcr_als(
        mixed_data, spectra, nIter=2, normL=normalization
    )

    assert np.isfinite(concentrations).all()
    assert np.isfinite(fitted_spectra).all()
    assert np.isfinite(residuals).all()
    np.testing.assert_allclose(concentrations[:, 1], 0, atol=1e-12)
    np.testing.assert_allclose(fitted_spectra[1], 0, atol=1e-12)
    np.testing.assert_allclose(concentrations @ fitted_spectra, mixed_data, atol=1e-12)


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_rejects_zero_initial_spectrum(normalization):
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mixed_data = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]
    )

    with pytest.raises(ValueError, match="Initial spectra must be nonzero"):
        mcr_als(mixed_data, spectra, nIter=2, normL=normalization)


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_improves_imperfect_spectra_over_50_iterations(normalization):
    true_spectra = np.array(
        [[2.0, 0.2, 1.8, 0.1], [0.1, 1.2, 0.3, 1.4]]
    )
    concentrations = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [0.4, 1.5]]
    )
    mixed_data = concentrations @ true_spectra
    starting_spectra = np.array(
        [[1.8, 0.3, 1.6, 0.2], [0.2, 1.0, 0.4, 1.2]]
    )

    fitted_concentrations, fitted_spectra, residuals = mcr_als(
        mixed_data, starting_spectra, nIter=50, normL=normalization
    )

    assert residuals.shape == (51,)
    assert np.isfinite(residuals).all()
    assert np.all(np.diff(residuals) <= 1e-12)
    assert residuals[-1] < residuals[0] / 1000
    np.testing.assert_allclose(
        residuals[-1],
        lof(mixed_data, fitted_concentrations @ fitted_spectra, method="sqr"),
        atol=1e-12,
    )


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_recovers_known_spectra_from_imperfect_start(normalization):
    true_spectra = np.array(
        [[2.0, 0.2, 1.8, 0.1], [0.1, 1.2, 0.3, 1.4]]
    )
    concentrations = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [0.4, 1.5]]
    )
    mixed_data = concentrations @ true_spectra
    starting_spectra = np.array(
        [[1.8, 0.3, 1.6, 0.2], [0.2, 1.0, 0.4, 1.2]]
    )

    # Pure-component pixels make the known spectra recoverable. Compare unit-L2
    # rows so the test measures spectral shape, not arbitrary factor scaling.
    true_unit_spectra = normalize_components(true_spectra, do_norm="L2")
    starting_unit_spectra = normalize_components(starting_spectra, do_norm="L2")
    true_spectra_norm = np.linalg.norm(true_unit_spectra)
    initial_global_error = (
        np.linalg.norm(starting_unit_spectra - true_unit_spectra)
        / true_spectra_norm
    )
    initial_channel_error = np.max(
        np.abs(starting_unit_spectra - true_unit_spectra)
    )

    _, fitted_spectra, _ = mcr_als(
        mixed_data, starting_spectra, nIter=50, normL=normalization
    )
    fitted_unit_spectra = normalize_components(fitted_spectra, do_norm="L2")
    fitted_global_error = (
        np.linalg.norm(fitted_unit_spectra - true_unit_spectra) / true_spectra_norm
    )
    fitted_channel_error = np.max(np.abs(fitted_unit_spectra - true_unit_spectra))

    assert fitted_global_error < initial_global_error / 100
    assert fitted_channel_error < initial_channel_error / 100


def test_mcr_als_rejects_concentration_start():
    mixed_data = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]]
    )
    concentrations = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )

    with pytest.raises(ValueError, match="Initial concentration estimates are not supported"):
        mcr_als(mixed_data, concentrations, nIter=2, normL="L2")


def test_mcr_als_rejects_mismatched_spectral_channels():
    mixed_data = np.ones((4, 3))
    spectra = np.ones((2, 4))

    with pytest.raises(ValueError, match="Initial spectra must have the same number of channels"):
        mcr_als(mixed_data, spectra, nIter=2, normL="L2")


@pytest.mark.parametrize("iterations", [-1, 1.5])
def test_mcr_als_rejects_invalid_iteration_count(iterations):
    mixed_data = np.ones((4, 3))
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="nIter must be a non-negative integer"):
        mcr_als(mixed_data, spectra, nIter=iterations, normL="L2")


def test_mcr_als_rejects_zero_data():
    mixed_data = np.zeros((4, 3))
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="mixData must contain a nonzero signal"):
        mcr_als(mixed_data, spectra, nIter=2, normL="L2")


@pytest.mark.parametrize("invalid_input", ["data", "spectra"])
def test_mcr_als_requires_two_dimensional_inputs(invalid_input):
    mixed_data = np.ones((4, 3))
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    if invalid_input == "data":
        mixed_data = mixed_data[0]
    else:
        spectra = spectra[0]

    with pytest.raises(ValueError, match="must be 2D arrays"):
        mcr_als(mixed_data, spectra, nIter=2, normL="L2")


def test_mcr_als_allows_zero_iterations():
    spectra = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    mixed_data = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])

    concentrations, fitted_spectra, residuals = mcr_als(
        mixed_data, spectra, nIter=0, normL="L2"
    )

    assert residuals.shape == (1,)
    np.testing.assert_allclose(concentrations @ fitted_spectra, mixed_data, atol=1e-12)
    np.testing.assert_allclose(residuals, 0, atol=1e-12)


def test_mcr_als_reports_convergence_for_an_exact_fit():
    spectra = np.eye(2)
    data = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])

    concentrations, fitted_spectra, history, diagnostics = mcr_als(
        data, spectra, nIter=20, normL="L1", tol_percent=0.1,
        return_diagnostics=True,
    )

    assert diagnostics.stop_reason == "converged"
    assert diagnostics.iterations_completed < 20
    assert len(history) == diagnostics.iterations_completed + 1
    assert diagnostics.best_iteration == 1
    np.testing.assert_allclose(
        diagnostics.best_relative_l2, history[diagnostics.best_iteration]
    )
    np.testing.assert_allclose(concentrations @ fitted_spectra, data, atol=1e-12)


def test_mcr_als_reports_maximum_iterations_when_stopping_is_not_reached():
    spectra = np.array([[1.0, 0.1], [0.1, 1.0]])
    data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    concentrations, fitted_spectra, history, diagnostics = mcr_als(
        data, spectra, nIter=0, normL="L1", tol_percent=0.1,
        return_diagnostics=True,
    )

    assert diagnostics.stop_reason == "max_iterations"
    assert diagnostics.iterations_completed == 0
    assert diagnostics.best_iteration == 0
    assert len(history) == 1
    np.testing.assert_allclose(
        diagnostics.best_relative_l2,
        lof(data, concentrations @ fitted_spectra, method="sqr"),
    )


def test_mcr_als_small_worsening_is_not_called_convergence(monkeypatch):
    data = np.eye(2)
    spectra = np.eye(2)
    responses = iter((np.eye(2), np.diag([1.01, 1.0]), np.eye(2)))
    monkeypatch.setattr(unmixing, "positive_least_squares", lambda **kwargs: next(responses))

    concentrations, fitted_spectra, history, diagnostics = mcr_als(
        data, spectra, nIter=1, normL=False, tol_percent=100.0,
        return_diagnostics=True,
    )

    assert history[1] > history[0]
    assert diagnostics.stop_reason == "max_iterations"
    assert diagnostics.best_iteration == 0
    np.testing.assert_array_equal(concentrations, np.eye(2))
    np.testing.assert_array_equal(fitted_spectra, np.eye(2))
    np.testing.assert_allclose(
        diagnostics.best_relative_l2,
        lof(data, concentrations @ fitted_spectra, method="sqr"),
    )


@pytest.mark.parametrize(
    ("tolerance", "expected_reason"),
    [(50.0, "converged"), (1.0, "max_iterations")],
)
def test_mcr_als_uses_relative_improvement_and_maximum(monkeypatch, tolerance, expected_reason):
    data = np.eye(2)
    starting_spectra = 0.9 * np.eye(2)
    responses = iter((
        0.9 * np.eye(2), 0.95 * np.eye(2), 0.95 * np.eye(2)
    ))
    monkeypatch.setattr(unmixing, "positive_least_squares", lambda **kwargs: next(responses))

    concentrations, spectra, history, diagnostics = mcr_als(
        data, starting_spectra, nIter=1, normL=False,
        tol_percent=tolerance, return_diagnostics=True,
    )

    assert diagnostics.stop_reason == expected_reason
    assert diagnostics.iterations_completed == diagnostics.best_iteration == 1
    np.testing.assert_allclose(history, [0.19, 0.0975])
    np.testing.assert_allclose(
        diagnostics.best_relative_l2,
        lof(data, concentrations @ spectra, method="sqr"),
    )


@pytest.mark.parametrize("tolerance", [-0.1, 0.0, np.nan, np.inf, True])
def test_mcr_als_rejects_invalid_convergence_tolerance(tolerance):
    with pytest.raises(ValueError, match="tol_percent"):
        mcr_als(
            np.eye(2), np.eye(2), nIter=2, normL=False,
            tol_percent=tolerance,
        )


@pytest.mark.parametrize("normalization", [False, "L1", "L2"])
def test_mcr_als_fits_mixtures_without_pure_component_pixels(normalization):
    known_spectra = np.array(
        [[2.0, 0.2, 1.8, 0.1], [0.1, 1.2, 0.3, 1.4]]
    )
    # Each pixel contains both components; none provides a pure spectrum.
    # Check reconstruction only: a good fit does not guarantee unique recovery
    # of the original signatures when the data has no pure-component pixels.
    concentrations = np.array(
        [[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [0.35, 0.65], [0.72, 0.28]]
    )
    mixed_data = concentrations @ known_spectra
    starting_spectra = np.array(
        [[1.8, 0.3, 1.6, 0.2], [0.2, 1.0, 0.4, 1.2]]
    )

    fitted_concentrations, fitted_spectra, residuals = mcr_als(
        mixed_data, starting_spectra, nIter=50, normL=normalization
    )
    reconstructed = fitted_concentrations @ fitted_spectra

    assert np.all(concentrations > 0)
    assert fitted_concentrations.shape == concentrations.shape
    assert fitted_spectra.shape == known_spectra.shape
    assert residuals.shape == (51,)
    assert np.isfinite(fitted_concentrations).all()
    assert np.isfinite(fitted_spectra).all()
    assert np.isfinite(residuals).all()
    assert np.all(fitted_concentrations >= 0)
    assert np.all(fitted_spectra >= 0)
    np.testing.assert_allclose(reconstructed, mixed_data, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        residuals[-1], lof(mixed_data, reconstructed, method="sqr"), atol=1e-12
    )
