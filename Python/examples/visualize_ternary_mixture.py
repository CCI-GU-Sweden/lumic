"""Save a ternary RGB diagram for BFP, EGFP, and mCherry mixtures.

Run ``python -m examples.visualize_ternary_mixture`` from the repository's
Python directory. The example fetches FPbase emission spectra and saves a PNG
under ``Python/data``. Pixels outside the triangle have zero spectral signal;
they are displayed as white in the PNG.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic.fpbase import fetch_emission_spectrum
from lumic.simulation import resample_spectrum, simulate_linear_mixture
from lumic.utils import RGB_from_spec, norm8bit


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Python" / "data"
WAVELENGTH_LABELS = DATA_DIR / "io-data" / "980" / "all-channels-labels.txt"
DEFAULT_OUTPUT = DATA_DIR / "visual-check-ternary-bfp-egfp-mcherry.png"
FLUOROPHORES = ("mTagBFP2", "EGFP", "mCherry")
IMAGE_SIZE = 401

# Vertex coordinates are (x, y) in the square image. Their spacing makes an
# approximately equilateral triangle and leaves room for the labels.
VERTICES = np.array([[200.0, 20.0], [10.0, 349.0], [390.0, 349.0]])


def build_ternary_abundances():
    """Return per-pixel fractions and the mask of pixels inside the triangle."""
    y, x = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    top, left, right = VERTICES

    # Barycentric coordinates give the amount of each fluorophore at a pixel.
    # The three values sum to one inside the triangle, and each vertex is pure.
    bfp = (left[1] - y) / (left[1] - top[1])
    mcherry = (x - left[0] - (top[0] - left[0]) * bfp) / (right[0] - left[0])
    egfp = 1.0 - bfp - mcherry
    fractions = np.stack((bfp, egfp, mcherry), axis=-1)

    inside = np.all(fractions >= -1e-12, axis=-1)
    abundances = np.where(inside[..., np.newaxis], np.clip(fractions, 0, 1), 0)
    return abundances, inside


def create_preview(output_path=DEFAULT_OUTPUT):
    """Simulate the ternary spectral image and save its labeled RGB diagram."""
    wavelengths = np.fromstring(WAVELENGTH_LABELS.read_text(), sep=",")
    spectra = []
    for name in FLUOROPHORES:
        source_wavelengths, source_intensities = fetch_emission_spectrum(name)
        spectra.append(
            resample_spectrum(source_wavelengths, source_intensities, wavelengths)
        )

    abundances, inside = build_ternary_abundances()
    dataset = simulate_linear_mixture(
        abundances,
        np.asarray(spectra),
        wavelengths,
        fluorophore_names=FLUOROPHORES,
    )

    # RGB_from_spec converts channel averages to integers. Scale the raw
    # FPbase floats first, as in the earlier three-fluorophore example.
    channels = norm8bit(np.moveaxis(dataset.image, -1, 0))
    rgb = RGB_from_spec(channels, do_norm=True)
    rgb[~inside] = 255

    fig = Figure(figsize=(7, 7), dpi=150)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(rgb, interpolation="nearest")

    # Lines of constant component fraction are parallel to the opposite side.
    for component in range(3):
        others = [index for index in range(3) if index != component]
        for fraction in (0.25, 0.5, 0.75):
            ends = [
                fraction * VERTICES[component]
                + (1 - fraction) * VERTICES[other]
                for other in others
            ]
            ax.plot(*np.asarray(ends).T, color="white", alpha=0.45, linewidth=0.8)

    outline = VERTICES[[0, 1, 2, 0]]
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2)
    ax.text(200, 7, "mTagBFP2 (100%)", ha="center", va="center", fontsize=11)
    ax.text(10, 371, "EGFP (100%)", ha="left", va="center", fontsize=11)
    ax.text(390, 371, "mCherry (100%)", ha="right", va="center", fontsize=11)
    ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
    ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    return output_path


if __name__ == "__main__":
    saved_path = create_preview()
    print(f"Saved ternary RGB preview to {saved_path}")
