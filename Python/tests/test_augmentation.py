import numpy as np

from lumic import augmentation_convolution


def test_identity_kernel_preserves_spectral_image():
    image = np.arange(24, dtype=float).reshape(2, 3, 4)
    identity = np.array([[1.0]])

    augmented = augmentation_convolution(image, kernels=(identity,))

    np.testing.assert_array_equal(augmented, image)


def test_four_neighbour_kernel_normalizes_and_appends_channels():
    image = np.zeros((2, 5, 5), dtype=float)
    image[0, 2, 2] = 4.0
    image[1, 2, 2] = 8.0
    identity = np.array([[1.0]])
    four_neighbours = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )

    augmented = augmentation_convolution(image, kernels=(identity, four_neighbours))

    expected = np.zeros((4, 5, 5), dtype=float)
    expected[0, 2, 2] = 4.0
    expected[1, 2, 2] = 8.0
    for y, x in ((1, 2), (2, 1), (2, 3), (3, 2)):
        expected[2, y, x] = 1.0
        expected[3, y, x] = 2.0

    np.testing.assert_array_equal(augmented, expected)
