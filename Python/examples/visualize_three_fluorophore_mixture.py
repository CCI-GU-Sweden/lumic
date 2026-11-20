"""Save an RGB preview of a 20 x 20 BFP/EGFP/mCherry mixture.

Run ``python -m examples.visualize_three_fluorophore_mixture`` from the
repository's Python directory. The script fetches the latest emission curves
from FPbase and writes the preview under ``Python/data``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from lumic.fpbase import fetch_emission_spectrum
from lumic.simulation import (
    make_mixture_gradient,
    resample_spectrum,
    simulate_linear_mixture,
)
from lumic.utils import RGB_from_spec, norm8bit


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Python" / "data"
WAVELENGTH_LABELS = DATA_DIR / "io-data" / "980" / "all-channels-labels.txt"
DEFAULT_OUTPUT = DATA_DIR / "visual-check-bfp-egfp-mcherry.png"
FLUOROPHORES = ("mTagBFP2", "EGFP", "mCherry")


def create_preview(output_path=DEFAULT_OUTPUT):
    """Fetch spectra, simulate the mixture, and save its RGB preview."""
    wavelengths = np.fromstring(WAVELENGTH_LABELS.read_text(), sep=",")
    spectra = []
    for name in FLUOROPHORES:
        source_wavelengths, source_intensities = fetch_emission_spectrum(name)
        spectra.append(
            resample_spectrum(source_wavelengths, source_intensities, wavelengths)
        )

    abundances = make_mixture_gradient(FLUOROPHORES, shape=(20, 20))
    dataset = simulate_linear_mixture(
        abundances,
        np.asarray(spectra),
        wavelengths,
        fluorophore_names=FLUOROPHORES,
    )

    # RGB_from_spec converts its channel averages to integers. Scale the raw
    # FPbase float intensities to 8-bit first so those values are not truncated.
    channels = norm8bit(np.moveaxis(dataset.image, -1, 0))
    rgb = RGB_from_spec(channels, do_norm=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = Image.fromarray(rgb).resize((600, 600), Image.Resampling.NEAREST)
    preview.save(output_path)
    return output_path


if __name__ == "__main__":
    saved_path = create_preview()
    print(f"Saved RGB preview to {saved_path}")
