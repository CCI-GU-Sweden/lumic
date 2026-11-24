"""Save, reload, and use detector-sampled reference spectra.

From Python/, run python -m examples.spectrum_collection_workflow spectra.json.
The example uses three bundled FPbase curves and an ideal 32-channel detector.
"""

import argparse
from pathlib import Path

import numpy as np

from lumic import SpectrumCollection
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum
from lumic.unmixing import positive_least_squares


def run_workflow(output_path):
    """Write the reference JSON and return the reloaded collection and fitted abundances."""
    names = ("EBFP2", "EGFP", "mCherry")
    detector = SpectralDetector.from_profile("980_32ch")

    # Each matrix row holds the integrated emission for one fluorophore.
    curves = [load_offline_emission_spectrum(name) for name in names]
    sampled = np.stack([detector.sample(wavelengths, emission) for wavelengths, emission in curves])
    edges = detector.bin_edges
    collection = SpectrumCollection(
        sampled,
        detector.channel_centers,
        detection_bands_nm=np.column_stack((edges[:-1], edges[1:])),
        names=names,
        row_provenance=[
            {"source": "FPbase", "bundled_file": f"{name.lower()}_emission.csv"}
            for name in names
        ],
        provenance={
            "detector_profile": detector.name,
            "sampling": "Integration over ideal detector bands",
            "normalization": None,
        },
    )
    collection.save_json(output_path)
    restored = SpectrumCollection.load_json(output_path)

    # The two noiseless observations have known abundances in component order.
    known = np.array([[0.50, 0.25, 0.25], [0.00, 0.25, 0.75]])
    observations = known @ sampled
    direct_fit = positive_least_squares(observations, sampled)
    restored_fit = positive_least_squares(observations, restored.values)
    np.testing.assert_allclose(restored_fit, direct_fit, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(restored_fit, known, rtol=1e-10, atol=1e-12)
    return restored, restored_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="Path for the reference spectra JSON")
    args = parser.parse_args()
    collection, fitted = run_workflow(args.output)
    print(f"Saved {len(collection)} reference spectra to {args.output}")
    print(f"Fitted abundances in component order ({', '.join(collection.names)})")
    print(fitted.round(6))


if __name__ == "__main__":
    main()
