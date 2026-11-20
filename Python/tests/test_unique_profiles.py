import numpy as np

from lumic import unique_profiles


def test_unique_profiles_selects_pure_spectra_from_candidate_pool():
    pure_spectra = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    mixture_weights = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.4, 0.4],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
            [0.3, 0.3, 0.4],
            [0.3, 0.4, 0.3],
            [0.3, 0.5, 0.2],
            [0.4, 0.2, 0.4],
            [0.4, 0.3, 0.3],
            [0.4, 0.4, 0.2],
            [0.5, 0.2, 0.3],
            [0.5, 0.3, 0.2],
        ]
    )
    mixture_rows = mixture_weights @ pure_spectra

    # These are candidate profiles (rows); all mixtures have positive weight
    # from each component, so only the inserted rows are pure spectra.
    pure_indices = [2, 8, 12]
    candidate_rows = mixture_rows.tolist()
    for index, spectrum in zip(pure_indices, pure_spectra):
        candidate_rows.insert(index, spectrum.tolist())
    candidate_pool = np.asarray(candidate_rows)

    # unique_profiles expects variables in columns, hence the transpose.
    profiles, selected_indices = unique_profiles(candidate_pool.T, nr=3, f=0.5)

    assert set(selected_indices) == set(pure_indices)
    assert profiles.shape == pure_spectra.shape
    expected_unit_spectra = pure_spectra / np.linalg.norm(
        pure_spectra, axis=1, keepdims=True
    )
    np.testing.assert_allclose(profiles, expected_unit_spectra, atol=1e-12)
