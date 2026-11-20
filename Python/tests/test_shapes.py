import numpy as np

from lumic import fold_abundance_matrix, fold_flat_spec_data, unfold_spec_data


def test_unfold_spec_data_preserves_pixel_and_channel_order():
    # Image axes are [channel, y, x]: four channels, three rows, two columns.
    image = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[11, 12], [13, 14], [15, 16]],
            [[21, 22], [23, 24], [25, 26]],
            [[31, 32], [33, 34], [35, 36]],
        ]
    )
    expected_pixels = np.array(
        [
            [1, 11, 21, 31],
            [2, 12, 22, 32],
            [3, 13, 23, 33],
            [4, 14, 24, 34],
            [5, 15, 25, 35],
            [6, 16, 26, 36],
        ]
    )

    np.testing.assert_array_equal(unfold_spec_data(image), expected_pixels)


def test_fold_flat_spec_data_reconstructs_spectral_image():
    image = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[11, 12], [13, 14], [15, 16]],
            [[21, 22], [23, 24], [25, 26]],
            [[31, 32], [33, 34], [35, 36]],
        ]
    )

    pixels = unfold_spec_data(image)

    np.testing.assert_array_equal(fold_flat_spec_data(pixels, image.shape), image)


def test_fold_abundance_matrix_places_components_at_pixel_positions():
    abundances = np.array(
        [
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50],
            [6, 60],
        ]
    )
    expected_maps = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[10, 20], [30, 40], [50, 60]],
        ]
    )

    np.testing.assert_array_equal(
        fold_abundance_matrix(abundances, spec_data_shape=(4, 3, 2)),
        expected_maps,
    )
