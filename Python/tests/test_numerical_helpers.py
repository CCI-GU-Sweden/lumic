import numpy as np
import pytest

from lumic import lof, normalize_components


def test_lof_matches_known_residuals():
    observed = np.array([1.0, 2.0])
    predicted = np.array([1.0, 1.0])

    assert lof(observed, predicted, method="abs") == pytest.approx(1 / 3)
    assert lof(observed, predicted, method="sqr") == pytest.approx(np.sqrt(1 / 5))


def test_lof_is_zero_for_exact_prediction():
    observed = np.array([1.0, 2.0])

    assert lof(observed, observed, method="abs") == 0
    assert lof(observed, observed, method="sqr") == 0


def test_normalize_components_l1():
    components = np.array([[3.0, 4.0], [5.0, 12.0]])
    expected = np.array([[3 / 7, 4 / 7], [5 / 17, 12 / 17]])

    np.testing.assert_allclose(normalize_components(components, do_norm="L1"), expected)


def test_normalize_components_l2():
    components = np.array([[3.0, 4.0], [5.0, 12.0]])
    expected = np.array([[3 / 5, 4 / 5], [5 / 13, 12 / 13]])

    np.testing.assert_allclose(normalize_components(components, do_norm="L2"), expected)


def test_normalize_components_disabled_preserves_values():
    components = np.array([[3.0, 4.0], [5.0, 12.0]])

    np.testing.assert_array_equal(normalize_components(components, do_norm=False), components)
