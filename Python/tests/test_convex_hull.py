import numpy as np

from lumic import essential_spectral_pixles


def test_convex_hull_selects_pure_spectra_buried_among_mixtures():
    pure_spectra = np.array(
        [
            [5.0, 1.0, 0.2, 0.1, 0.1],
            [0.1, 0.2, 5.0, 1.0, 0.1],
            [0.1, 0.1, 0.2, 1.0, 5.0],
        ]
    )

    # Create all strictly interior ternary mixtures with proportions in steps of 1/41.
    proportions = np.array(
        [
            [first, second, 41 - first - second]
            for first in range(1, 40)
            for second in range(1, 41 - first)
        ],
        dtype=float,
    ) / 41
    mixture_rows = proportions @ pure_spectra
    assert mixture_rows.shape == (780, pure_spectra.shape[1])

    # Insert the pure spectra at separated, non-edge positions in the data.
    pure_indices = [17, 391, 760]
    data_rows = mixture_rows.tolist()
    for index, spectrum in zip(pure_indices, pure_spectra):
        data_rows.insert(index, spectrum.tolist())
    mixed_data = np.asarray(data_rows)

    important_spectra, vertex_indices = essential_spectral_pixles(mixed_data, npc=3)

    assert set(vertex_indices) == set(pure_indices)
    assert important_spectra.shape == pure_spectra.shape
    np.testing.assert_array_equal(important_spectra, mixed_data[vertex_indices])
