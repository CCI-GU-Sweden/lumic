"""Create a tiny 3 x 2 noiseless EGFP/mCherry LSM980 spectral image.

Run ``python -m examples.simulate_fpbase_mixtures`` from the repository's
Python directory with ``unmixing-env`` active. This example contacts FPbase;
the unit tests use fixed spectra and stay offline.
"""

from pathlib import Path

import numpy as np

from lumic.fpbase import fetch_emission_spectrum
from lumic.simulation import (
    make_abundance_maps,
    resample_spectrum,
    simulate_linear_mixture,
)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    labels_path = (
        repo_root / "Python" / "data" / "io-data" / "980" / "all-channels-labels.txt"
    )
    wavelengths = np.fromstring(labels_path.read_text(), sep=",")

    fluorophores = ("EGFP", "mCherry")
    spectra = []
    for name in fluorophores:
        source_wavelengths, source_intensities = fetch_emission_spectrum(name)
        spectra.append(
            resample_spectrum(source_wavelengths, source_intensities, wavelengths)
        )

    # Every pixel is mixed, and each component contributes at most 70%.
    fractions = np.array(
        [
            [[0.7, 0.3], [0.6, 0.4]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.55, 0.45], [0.45, 0.55]],
        ]
    )
    abundances = make_abundance_maps(fractions, shape=(3, 2))
    dataset = simulate_linear_mixture(
        abundances,
        np.asarray(spectra),
        wavelengths,
        fluorophore_names=fluorophores,
    )
    return dataset


if __name__ == "__main__":
    result = main()
    print(f"Image shape (y, x, channel): {result.image.shape}")
    print(f"Fluorophores: {', '.join(result.fluorophore_names)}")
