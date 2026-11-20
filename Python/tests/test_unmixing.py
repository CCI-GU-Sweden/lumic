import numpy as np
import pytest

import lumic
from lumic import positive_least_squares, unmixing


def test_existing_unmixing_imports_are_preserved():
    for name in ("positive_least_squares", "mcr_als", "MCRALSDiagnostics",
                 "normalize_components", "lof"):
        assert getattr(lumic, name) is getattr(unmixing, name)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_positive_least_squares_recovers_known_mixture(batch_size):
    components = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    expected_abundances = np.array([[2.0, 3.0], [4.0, 0.0], [0.0, 1.0]])
    mixed_data = np.array([[2.0, 3.0, 5.0], [4.0, 0.0, 4.0], [0.0, 1.0, 1.0]])

    estimated_abundances = positive_least_squares(
        mixed_data, components, do_norm=False, batch_size=batch_size
    )

    assert estimated_abundances.shape == expected_abundances.shape
    assert np.all(estimated_abundances >= 0)
    np.testing.assert_allclose(
        estimated_abundances, expected_abundances, rtol=1e-12, atol=1e-12
    )


def test_positive_least_squares_retries_failed_nnls_batch_with_bvls(monkeypatch):
    components = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    expected_abundances = np.array([[2.0, 3.0], [4.0, 0.0], [0.0, 1.0]])
    mixed_data = np.array([[2.0, 3.0, 5.0], [4.0, 0.0, 4.0], [0.0, 1.0, 1.0]])

    def fail_nnls(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    monkeypatch.setattr(unmixing.LinearRegression, "fit", fail_nnls)

    with pytest.warns(RuntimeWarning, match="BVLS fallback was used"):
        estimated_abundances = positive_least_squares(
            mixed_data, components, do_norm=False, batch_size=3
        )

    np.testing.assert_allclose(
        estimated_abundances, expected_abundances, rtol=1e-10, atol=1e-10
    )


def test_positive_least_squares_marks_unresolved_bvls_observations_as_nan(monkeypatch):
    components = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    mixed_data = np.array([[2.0, 3.0, 5.0]])

    def fail_nnls(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    def fail_bvls(*args, **kwargs):
        return type("FailedFit", (), {"success": False, "x": np.zeros(2)})()

    monkeypatch.setattr(unmixing.LinearRegression, "fit", fail_nnls)
    monkeypatch.setattr(unmixing, "lsq_linear", fail_bvls)

    with pytest.warns(RuntimeWarning, match=r"observation index/indices \[0\]"):
        estimated_abundances = positive_least_squares(
            mixed_data, components, do_norm=False, batch_size=1
        )

    assert np.isnan(estimated_abundances).all()
