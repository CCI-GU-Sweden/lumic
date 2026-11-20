"""Check row/coordinate correspondence independently of inverse reshaping."""
from itertools import permutations

import numpy as np
import pytest

from lumic import SpectralImage, UnfoldedImage
from lumic import unfold_spec_data, fold_flat_spec_data, fold_abundance_matrix


def test_all_axis_orders_keep_spectra_and_component_coordinates():
    # Different coordinate contributions reveal both swapped axes and wrong
    # row order. No reconstructed-pixel oracle uses the folding implementation.
    canonical_axes = "CTZYX"
    sizes = dict(C=3, T=2, Z=2, Y=2, X=4)
    indices = np.indices(tuple(sizes[a] for a in canonical_axes))
    canonical = sum(index * weight for index, weight in zip(indices, (10000, 1000, 100, 10, 1)))
    for ordering in permutations(canonical_axes):
        axes = "".join(ordering)
        image = SpectralImage(canonical.transpose([canonical_axes.index(a) for a in axes]), axes)
        unfolded = image.unfold()
        assert isinstance(unfolded, UnfoldedImage)
        assert unfolded.source_axes == axes
        observation_axes = axes.replace("C", "")
        observation_shape = tuple(sizes[a] for a in observation_axes)
        # Define two component values per pixel, independent of channel count.
        abundances = np.empty((np.prod(observation_shape), 2), dtype=int)
        for row, position in enumerate(np.ndindex(observation_shape)):
            coordinates = dict(zip(observation_axes, position))
            for c in range(sizes["C"]):
                canonical_position = tuple(c if a == "C" else coordinates[a] for a in canonical_axes)
                assert unfolded.data[row, c] == canonical[canonical_position], axes
            abundances[row] = [row, row + 100]
        maps = unfolded.fold_abundances(abundances, component_names=["A", "B"])
        assert isinstance(maps, SpectralImage)
        assert maps.kind == "abundance"
        assert maps.axes == axes.replace("C", "K")
        assert maps.component_names == ("A", "B")
        for row, position in enumerate(np.ndindex(observation_shape)):
            coordinates = dict(zip(observation_axes, position))
            for k in range(2):
                output_position = tuple(k if a == "K" else coordinates[a] for a in maps.axes)
                assert maps.data[output_position] == abundances[row, k], axes
        np.testing.assert_array_equal(unfolded.fold_spectra().data, image.data)
        reconstructed = unfolded.fold_spectra(unfolded.data + 0.25)
        assert reconstructed.axes == axes
        assert reconstructed.kind == "reconstructed"
        np.testing.assert_array_equal(reconstructed.data, image.data + 0.25)


def test_metadata_preserved_but_not_assigned_to_components():
    data = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
    image = SpectralImage(data, "CYX", wavelengths_nm=[450, 500, 600],
                          detection_bands_nm=[[440, 460], [490, 510], [590, 610]],
                          pixel_sizes_um={"X": .1, "Y": .2},
                          provenance={"path": "example.czi", "selection": {"Z": 2}})
    unfolded = image.unfold()
    restored = unfolded.fold_spectra()
    assert restored.data.dtype == np.uint16
    np.testing.assert_array_equal(restored.wavelengths_nm, image.wavelengths_nm)
    np.testing.assert_array_equal(restored.detection_bands_nm, image.detection_bands_nm)
    assert restored.pixel_sizes_um == image.pixel_sizes_um
    assert restored.provenance == image.provenance
    maps = unfolded.fold_abundances(np.ones((8, 2)))
    assert maps.pixel_sizes_um == image.pixel_sizes_um
    assert maps.provenance == image.provenance
    assert maps.component_names is None
    assert maps.wavelengths_nm is None
    assert maps.detection_bands_nm is None
    unfolded.wavelengths_nm[0] = 400
    assert image.wavelengths_nm[0] == 450  # Metadata is copied.


@pytest.mark.parametrize("axes,shape", [("CYX", (1, 1, 4)), ("TZCYX", (1, 1, 1, 2, 1)),
                                      ("YXC", (2, 4, 3))])
def test_singletons_and_noncontiguous_arrays(axes, shape):
    # Slicing makes the last axis non-contiguous without changing its labels.
    larger_shape = (*shape[:-1], shape[-1] * 2)
    data = np.arange(np.prod(larger_shape)).reshape(larger_shape)[..., ::2]
    image = SpectralImage(data, axes)
    restored = image.unfold().fold_spectra()
    assert restored.axes == axes
    assert restored.data.shape == shape
    np.testing.assert_array_equal(restored.data, data)


def test_cyx_results_agree_with_legacy_helpers():
    data = np.arange(30).reshape(3, 2, 5)
    unfolded = SpectralImage(data, "CYX").unfold()
    np.testing.assert_array_equal(unfolded.data, unfold_spec_data(data))
    np.testing.assert_array_equal(unfolded.fold_spectra().data,
                                  fold_flat_spec_data(unfolded.data, data.shape))
    abundance = np.arange(20).reshape(10, 2)
    np.testing.assert_array_equal(unfolded.fold_abundances(abundance).data,
                                  fold_abundance_matrix(abundance, data.shape))


@pytest.mark.parametrize("matrix", [np.zeros((7, 3)), np.zeros((8, 2)), np.zeros((2, 4, 3))])
def test_spectral_folding_rejects_wrong_matrix_shape(matrix):
    unfolded = SpectralImage(np.zeros((3, 2, 4)), "CYX").unfold()
    with pytest.raises(ValueError):
        unfolded.fold_spectra(matrix)


@pytest.mark.parametrize("matrix", [np.zeros((7, 2)), np.zeros((8, 0)), np.zeros((2, 4, 2))])
def test_abundance_folding_rejects_wrong_matrix_shape(matrix):
    unfolded = SpectralImage(np.zeros((3, 2, 4)), "CYX").unfold()
    with pytest.raises(ValueError):
        unfolded.fold_abundances(matrix)


@pytest.mark.parametrize("axes,shape", [("CYX", (3, 2, 0)), ("CYX", (3, 2)),
                                      ("CCX", (3, 2, 4)), ("CYX", (3, 2, 1.5))])
def test_invalid_source_layout_rejected(axes, shape):
    with pytest.raises(ValueError):
        UnfoldedImage(np.zeros((8, 3)), axes, shape)


def test_direct_construction_requires_matching_matrix_and_metadata():
    with pytest.raises(ValueError, match="source layout"):
        UnfoldedImage(np.zeros((7, 3)), "CYX", (3, 2, 4))
    with pytest.raises(ValueError, match="wavelengths_nm"):
        UnfoldedImage(np.zeros((8, 3)), "CYX", (3, 2, 4), wavelengths_nm=[500])
    with pytest.raises(TypeError, match="NumPy"):
        UnfoldedImage([[1, 2]], "CYX", (2, 1, 1))


@pytest.mark.parametrize("names,error", [(["A"], ValueError), (["A", 1], ValueError), ("AB", TypeError)])
def test_invalid_component_names(names, error):
    unfolded = SpectralImage(np.zeros((3, 2, 4)), "CYX").unfold()
    with pytest.raises(error):
        unfolded.fold_abundances(np.zeros((8, 2)), component_names=names)


def test_abundance_image_requires_component_axes():
    with pytest.raises(ValueError, match="axes"):
        SpectralImage(np.zeros((2, 3, 4)), "CYX", kind="abundance")


def test_image_kind_is_immutable_and_round_trip_preserves_it():
    from dataclasses import FrozenInstanceError

    for kind in ("measured", "reconstructed"):
        image = SpectralImage(np.zeros((3, 2, 4)), "CYX", kind=kind)
        unfolded = image.unfold()
        assert unfolded.source_kind == kind
        assert unfolded.fold_spectra().kind == kind
        assert unfolded.fold_spectra(unfolded.data).kind == "reconstructed"
        with pytest.raises(FrozenInstanceError):
            image.kind = "abundance"
    maps = unfolded.fold_abundances(np.zeros((8, 2)))
    assert maps.kind == "abundance"
    with pytest.raises(FrozenInstanceError):
        maps.kind = "measured"
    with pytest.raises(ValueError, match="not abundance"):
        maps.unfold()


@pytest.mark.parametrize("kind,axes,metadata", [
    ("unknown", "CYX", {}),
    ("abundance", "KYX", {"wavelengths_nm": [500, 600]}),
    ("abundance", "KYX", {"detection_bands_nm": [[490, 510], [590, 610]]}),
    ("measured", "KYX", {}),
    ("reconstructed", "KYX", {}),
    ("measured", "CYX", {"component_names": ["A", "B"]}),
])
def test_kind_requires_appropriate_axes_and_metadata(kind, axes, metadata):
    with pytest.raises(ValueError):
        SpectralImage(np.zeros((2, 3, 4)), axes, kind=kind, **metadata)


def test_existing_solver_can_unmix_and_restore_a_permuted_volume():
    from lumic import positive_least_squares

    spectra = np.array([[1., .5, 0.], [0., .25, 1.]])
    # Ground-truth abundance maps are KZYX. Neither folding nor unfolding is used
    # to generate the measurements or the expected output coordinates.
    truth = np.arange(1, 25, dtype=float).reshape(2, 2, 2, 3)
    detector = np.einsum("kzyx,kc->zyxc", truth, spectra)
    image = SpectralImage(detector, "ZYXC", wavelengths_nm=[450, 550, 650])
    unfolded = image.unfold()
    abundance = positive_least_squares(unfolded.data, spectra, do_norm=False)
    maps = unfolded.fold_abundances(abundance, component_names=["A", "B"])
    assert maps.axes == "ZYXK"
    assert maps.kind == "abundance"
    np.testing.assert_allclose(maps.data, np.moveaxis(truth, 0, -1), atol=1e-12)
    reconstruction = unfolded.fold_spectra(abundance @ spectra)
    assert reconstruction.axes == image.axes
    assert reconstruction.kind == "reconstructed"
    np.testing.assert_allclose(reconstruction.data, detector, atol=1e-12)
