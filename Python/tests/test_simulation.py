import numpy as np
import pytest

from examples import simulate_fpbase_mixtures
from examples import visualize_three_fluorophore_mixture
from examples import visualize_ternary_mixture
from lumic.simulation import (
    make_abundance_maps,
    make_mixture_gradient,
    resample_spectrum,
    simulate_linear_mixture,
)


def test_resample_spectrum_interpolates_and_uses_zero_outside_source_range():
    result = resample_spectrum([500, 510, 520], [0, 2, 0], [490, 500, 505, 510, 515, 520, 530])

    np.testing.assert_allclose(result, [0, 0, 1, 2, 1, 0, 0])


def test_resample_spectrum_preserves_input_intensity_scale():
    result = resample_spectrum([500, 510], [4, 8], [500, 505, 510])

    np.testing.assert_allclose(result, [4, 6, 8])


def test_make_abundance_maps_preserves_pixelwise_mixture_fractions():
    fractions = np.array(
        [
            [[0.7, 0.3], [0.6, 0.4]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.55, 0.45], [0.45, 0.55]],
        ]
    )
    total = np.array([[1, 2], [3, 4], [5, 6]])

    abundances = make_abundance_maps(fractions, (3, 2), total)

    np.testing.assert_allclose(abundances.sum(axis=-1), total)
    np.testing.assert_allclose(abundances / total[..., None], fractions)
    assert np.max(abundances / total[..., None]) < 0.8


def test_two_component_mixture_gradient_runs_from_first_pure_to_second_pure():
    abundances = make_mixture_gradient(("EGFP", "mCherry"), shape=(10, 10))
    flattened = abundances.reshape(-1, 2)

    assert abundances.shape == (10, 10, 2)
    np.testing.assert_allclose(flattened[:, 0], np.linspace(1.0, 0.0, 100))
    np.testing.assert_allclose(flattened[:, 1], np.linspace(0.0, 1.0, 100))
    np.testing.assert_array_equal(flattened[0], [1.0, 0.0])
    np.testing.assert_array_equal(flattened[-1], [0.0, 1.0])


def test_multi_component_gradient_covers_simplex_compositions():
    abundances = make_mixture_gradient(
        ("EGFP", "mCherry", "BFP"), shape=(2, 3), n_steps=2
    )
    unique_compositions = np.unique(abundances.reshape(-1, 3), axis=0)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(unique_compositions, np.unique(expected, axis=0))
    np.testing.assert_allclose(abundances.sum(axis=-1), 1.0)


def test_mixture_gradient_rejects_lattice_larger_than_image():
    with pytest.raises(ValueError, match="requires 6"):
        make_mixture_gradient(("A", "B", "C"), shape=(2, 2), n_steps=2)


def test_linear_mixture_matches_known_two_component_image_and_retains_truth():
    abundances = np.array([[[2.0, 3.0], [4.0, 1.0]]])
    spectra = np.array([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    wavelengths = np.array([500.0, 510.0, 520.0])

    dataset = simulate_linear_mixture(
        abundances, spectra, wavelengths, fluorophore_names=("EGFP", "mCherry")
    )

    np.testing.assert_allclose(
        dataset.image, [[[2.0, 6.0, 7.0], [4.0, 2.0, 9.0]]]
    )
    np.testing.assert_array_equal(dataset.abundances, abundances)
    np.testing.assert_array_equal(dataset.spectra, spectra)
    np.testing.assert_array_equal(dataset.wavelengths, wavelengths)
    assert dataset.fluorophore_names == ("EGFP", "mCherry")


def test_fpbase_example_maps_curves_to_lsm980_channels_offline(monkeypatch):
    curves = {
        "EGFP": ([400.0, 700.0], [1.0, 0.0]),
        "mCherry": ([400.0, 700.0], [0.0, 1.0]),
    }
    monkeypatch.setattr(
        simulate_fpbase_mixtures,
        "fetch_emission_spectrum",
        lambda name: curves[name],
    )

    dataset = simulate_fpbase_mixtures.main()

    assert dataset.image.shape == (3, 2, 32)
    assert dataset.abundances.shape == (3, 2, 2)
    assert dataset.wavelengths.shape == (32,)
    assert np.max(dataset.abundances) == pytest.approx(0.7)
    assert dataset.fluorophore_names == ("EGFP", "mCherry")


def test_three_fluorophore_preview_saves_png_without_network(monkeypatch, tmp_path):
    curves = {
        "mTagBFP2": ([400.0, 700.0], [1.0, 0.0]),
        "EGFP": ([400.0, 700.0], [0.5, 0.5]),
        "mCherry": ([400.0, 700.0], [0.0, 1.0]),
    }
    monkeypatch.setattr(
        visualize_three_fluorophore_mixture,
        "fetch_emission_spectrum",
        lambda name: curves[name],
    )
    output_path = tmp_path / "preview.png"

    saved_path = visualize_three_fluorophore_mixture.create_preview(output_path)

    assert saved_path == output_path
    assert output_path.is_file()


def test_ternary_abundances_have_pure_vertices_and_zero_background():
    abundances, inside = visualize_ternary_mixture.build_ternary_abundances()
    vertices = visualize_ternary_mixture.VERTICES.astype(int)

    assert abundances.shape == (401, 401, 3)
    for component, (x, y) in enumerate(vertices):
        assert inside[y, x]
        np.testing.assert_array_equal(abundances[y, x], np.eye(3)[component])
    np.testing.assert_allclose(abundances[inside].sum(axis=-1), 1.0)
    np.testing.assert_array_equal(abundances[~inside], 0)

    # Zero abundances must also produce zero spectral signal outside the triangle.
    dataset = simulate_linear_mixture(abundances, np.eye(3), [450, 510, 610])
    np.testing.assert_array_equal(dataset.image[~inside], 0)


def test_ternary_preview_saves_png_without_network(monkeypatch, tmp_path):
    curves = {
        "mTagBFP2": ([400.0, 700.0], [1.0, 0.0]),
        "EGFP": ([400.0, 700.0], [0.5, 0.5]),
        "mCherry": ([400.0, 700.0], [0.0, 1.0]),
    }
    monkeypatch.setattr(
        visualize_ternary_mixture,
        "fetch_emission_spectrum",
        lambda name: curves[name],
    )
    output_path = tmp_path / "ternary.png"

    saved_path = visualize_ternary_mixture.create_preview(output_path)

    assert saved_path == output_path
    assert output_path.is_file()


@pytest.mark.parametrize(
    ("abundances", "spectra", "wavelengths", "message"),
    [
        (np.ones((2, 3)), np.ones((1, 4)), np.arange(4), "abundances"),
        (np.ones((1, 2, 2)), np.ones((1, 4)), np.arange(4), "spectra"),
        (np.ones((1, 2, 1)), np.ones((1, 4)), np.arange(3), "wavelengths"),
    ],
)
def test_linear_mixture_rejects_inconsistent_shapes(abundances, spectra, wavelengths, message):
    with pytest.raises(ValueError, match=message):
        simulate_linear_mixture(abundances, spectra, wavelengths)
