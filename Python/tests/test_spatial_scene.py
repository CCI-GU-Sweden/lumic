import numpy as np
import pytest

from examples import visualize_spatial_scene
from lumic import positive_least_squares
from lumic.simulation import (
    Ellipse,
    GaussianPSF,
    Line,
    Rectangle,
    SamplingGrid,
    Scene2D,
    SpectralDetector,
    load_offline_emission_spectrum,
    render_scene,
    simulate_noisy_image,
)


def test_shapes_add_counts_for_multiple_fluorophores():
    scene = Scene2D(
        ("EBFP2", "EGFP", "mCherry"),
        (
            Ellipse((0.5, 0.5), (0.3, 0.2), {"EBFP2": 500, "EGFP": 200}),
            Rectangle((0.5, 0.5), (0.2, 0.2), {"EGFP": 300}),
            Line((0.2, 0.5), (0.8, 0.5), 0.1, {"mCherry": 400}),
        ),
    )
    result = render_scene(scene, SamplingGrid((10, 10)))

    # The line and both filled shapes meet at this scan position. Their
    # count levels add independently in component order.
    np.testing.assert_allclose(result.source_counts[4, 4], [500, 500, 400])
    np.testing.assert_array_equal(result.source_counts[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(result.blurred_counts, result.source_counts)


def test_broad_shape_has_same_per_pixel_counts_at_two_scan_spacings():
    scene = Scene2D(
        ("EGFP",), (Ellipse((1.0, 1.0), (0.8, 0.8), {"EGFP": 500}),)
    )
    coarse = render_scene(scene, SamplingGrid((10, 10), pixel_size_um=0.2))
    fine = render_scene(scene, SamplingGrid((17, 17), pixel_size_um=0.12))

    assert coarse.source_counts[5, 5, 0] == 500
    assert fine.source_counts[8, 8, 0] == 500
    assert fine.source_counts.sum() > coarse.source_counts.sum()

    psf = GaussianPSF({"EGFP": 0.2})
    blurred_coarse = render_scene(scene, coarse.grid, psf)
    blurred_fine = render_scene(scene, fine.grid, psf)
    assert blurred_coarse.blurred_counts[5, 5, 0] == pytest.approx(500, abs=1e-3)
    assert blurred_fine.blurred_counts[8, 8, 0] == pytest.approx(500, abs=1e-3)


def test_gaussian_psf_recovers_thin_line_between_scan_positions():
    scene = Scene2D(
        ("EGFP",), (Line((0.5, 0.2), (0.5, 0.8), 0.02, {"EGFP": 500}),)
    )
    grid = SamplingGrid((10, 10), pixel_size_um=0.1)
    result = render_scene(scene, grid, GaussianPSF({"EGFP": 0.2}))

    assert result.source_counts[4, 4, 0] == 0
    assert 0 < result.blurred_counts[4, 4, 0] < 500
    assert result.blurred_counts[4, 4, 0] > result.blurred_counts[0, 0, 0]


def test_psf_includes_an_object_just_outside_the_field_of_view():
    scene = Scene2D(
        ("EGFP",), (Ellipse((-0.05, 0.5), (0.04, 0.04), {"EGFP": 500}),)
    )
    result = render_scene(
        scene, SamplingGrid((10, 10)), GaussianPSF({"EGFP": 0.2})
    )

    assert not np.any(result.source_counts)
    assert result.blurred_counts[4, 0, 0] > 0


def test_confocal_psf_width_tracks_wavelength_and_na():
    excitation = {"EBFP2": 405, "mCherry": 561}
    emission = {"EBFP2": 448, "mCherry": 610}
    low_na = GaussianPSF.approximate_confocal(0.7, excitation, emission)
    high_na = GaussianPSF.approximate_confocal(1.4, excitation, emission)

    assert low_na.fwhm_um["mCherry"] > low_na.fwhm_um["EBFP2"]
    assert high_na.fwhm_um["EBFP2"] == pytest.approx(
        low_na.fwhm_um["EBFP2"] / 2
    )


def test_rendered_three_component_scene_feeds_offline_detector_simulation():
    names = ("EBFP2", "EGFP", "mCherry")
    scene = Scene2D(
        names,
        (Rectangle((0.5, 0.5), (1.0, 1.0),
                   {"EBFP2": 500, "EGFP": 250, "mCherry": 100}),),
    )
    rendered = render_scene(scene, SamplingGrid((10, 10)))
    spectra = [load_offline_emission_spectrum(name) for name in names]
    result = simulate_noisy_image(
        rendered.blurred_counts, spectra,
        SpectralDetector.from_profile("980_32ch"),
        fluorophore_names=names, rng=np.random.default_rng(1),
    )

    assert result.expected_counts.shape == (10, 10, 32)
    np.testing.assert_allclose(result.expected_counts.sum(axis=-1), 850)
    assert result.ideal.fluorophore_names == names
    assert len(result.source_spectra) == 3


def test_known_spectra_recover_blurred_spatial_count_maps():
    names = ("EBFP2", "EGFP", "mCherry")
    scene = Scene2D(
        names,
        (
            Ellipse((0.6, 0.8), (0.45, 0.4), {"EBFP2": 500}),
            Rectangle((1.0, 0.8), (0.6, 0.7), {"EGFP": 400}),
            Line((0.3, 1.3), (1.5, 0.3), 0.08, {"mCherry": 300}),
        ),
    )
    rendered = render_scene(
        scene, SamplingGrid((18, 18)),
        GaussianPSF({"EBFP2": 0.14, "EGFP": 0.16, "mCherry": 0.18}),
    )
    spectra = [load_offline_emission_spectrum(name) for name in names]
    detector_result = simulate_noisy_image(
        rendered.blurred_counts, spectra,
        SpectralDetector.from_profile("980_32ch"),
        fluorophore_names=names, rng=np.random.default_rng(2),
    )
    observed = detector_result.expected_counts.reshape(-1, 32)
    recovered = positive_least_squares(
        observed, detector_result.ideal.spectra, batch_size=100
    ).reshape(18, 18, 3)

    assert np.max(np.count_nonzero(rendered.blurred_counts > 1, axis=-1)) >= 2
    np.testing.assert_allclose(
        recovered, rendered.blurred_counts, rtol=1e-8, atol=1e-7
    )


def test_scene_and_renderer_reject_unresolved_components():
    with pytest.raises(ValueError, match="unknown fluorophore"):
        Scene2D(("EGFP",), (Ellipse((0.5, 0.5), (0.2, 0.2), {"mCherry": 500}),))
    scene = Scene2D(("EGFP",), (Ellipse((0.5, 0.5), (0.2, 0.2), {"EGFP": 500}),))
    with pytest.raises(ValueError, match="missing fluorophore"):
        render_scene(scene, SamplingGrid((10, 10)), GaussianPSF({"mCherry": 0.2}))
    with pytest.raises(ValueError, match="offline emission spectrum"):
        load_offline_emission_spectrum("UnknownProtein")


def test_spatial_example_saves_visual_preview_and_detector_truth(tmp_path):
    output_path = tmp_path / "scene.png"
    spatial, detector_result = visualize_spatial_scene.create_preview(output_path)

    assert output_path.is_file()
    assert spatial.source_counts.shape == (160, 160, 3)
    assert np.any(spatial.source_counts != spatial.blurred_counts)
    assert detector_result.expected_counts.shape == (160, 160, 32)
    np.testing.assert_allclose(
        detector_result.expected_counts.sum(axis=-1),
        spatial.blurred_counts.sum(axis=-1),
    )
