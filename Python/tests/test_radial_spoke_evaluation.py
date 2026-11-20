"""Focused checks for the radial-spoke benchmark's scientific controls."""

import numpy as np

from evaluations.evaluate_radial_spoke_blur import (
    BACKGROUND_COUNTS,
    FLUOROPHORES,
    build_scene,
    detector_spectra,
    mixture_statistics,
    spoke_labels,
)
from lumic.simulation import SamplingGrid, SpectralDetector, render_scene


def test_spoke_assignment_is_seeded_and_includes_all_three_fluorophores():
    first = spoke_labels()
    np.testing.assert_array_equal(first, spoke_labels())
    assert set(first) == set(range(len(FLUOROPHORES)))


def test_purity_count_excludes_background_only_pixels():
    # Background is present in every pixel. A line pixel can be nearly pure,
    # but a background-only pixel must not count as a pure fluorophore.
    counts = np.array([[[0, 0, 0, 24], [900, 0, 0, 24],
                        [50, 50, 0, 24], [0.01, 0, 0, 24]]])
    stats = mixture_statistics(counts)
    assert stats["near_pure_95pct_including_bg"] == 1
    assert stats["near_pure_95pct_fluor_only"] == 1
    assert stats["fluor_signal_pixels"] == 2
    np.testing.assert_allclose(stats["max_fluor_fraction_including_bg"], 900 / 924)


def test_scene_background_and_detector_spectra_have_four_components():
    scene = build_scene()
    grid = SamplingGrid((8, 8), pixel_size_um=0.12)
    spatial = render_scene(scene, grid)
    np.testing.assert_array_equal(spatial.source_counts[..., -1], BACKGROUND_COUNTS)
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = detector_spectra(detector)
    assert spectra.shape == (4, 32)
    np.testing.assert_allclose(spectra.sum(axis=1), 1)
