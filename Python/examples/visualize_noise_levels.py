"""Save a four-panel preview of increasing noise in one spectral scene.

Run ``python -m examples.visualize_noise_levels`` from ``Python``. This uses
the FPbase emission curves saved in ``lumic/simulation/fpbase_data``, so it works offline.
Every panel uses the same component fractions and detector; only the expected
counts per pixel change. Brightness is adjusted for those count changes, making
the increasing relative noise visible without changing the underlying mixture.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic.simulation import (
    SpectralDetector,
    load_offline_emission_spectrum,
    simulate_noisy_image,
)
from lumic.utils import RGB_from_spec


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "visual-check-noise-levels.png"
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry", "mNeptune")
COUNT_LEVELS = (4000, 1000, 250, 60)
GAUSSIAN_SIGMA = 3.0  # Detector counts per channel, at every count level.
IMAGE_SIZE = 100


def _component_fractions():
    """Build a smooth color field, repeated exactly in every noise panel."""
    y, x = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    x /= IMAGE_SIZE - 1
    y /= IMAGE_SIZE - 1
    # Each corner is pure in one fluorophore. Every panel repeats this map.
    return np.stack(
        ((1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y), axis=-1
    )


def _rgb_preview(spectral_image, counts_per_pixel, display_gain):
    """Apply the same RGB conversion and display gain to every panel."""
    # Dividing by expected counts equalizes brightness, while preserving the
    # relative shot and read noise. Clip only for display, not in the data.
    display_channels = (
        np.moveaxis(spectral_image / counts_per_pixel, -1, 0) * display_gain
    )
    rgb = RGB_from_spec(display_channels, do_norm=False)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def create_preview(output_path=DEFAULT_OUTPUT):
    """Simulate four count levels and save their RGB images side by side."""
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = [load_offline_emission_spectrum(name) for name in FLUOROPHORES]
    fractions = _component_fractions()
    rng = np.random.default_rng(2026)

    # Establish one fixed display gain from the clean, brightest scene.
    first = simulate_noisy_image(
        fractions * COUNT_LEVELS[0], spectra, detector,
        fluorophore_names=FLUOROPHORES,
        gaussian_sigma=GAUSSIAN_SIGMA, rng=rng,
    )
    # RGB_from_spec converts averages to integers, so apply the float display
    # scale first. Use the peak clean bin as a shared reference for all panels.
    display_gain = 230.0 / np.max(first.expected_counts / COUNT_LEVELS[0])

    fig = Figure(figsize=(12, 3.5), dpi=150)
    axes = fig.subplots(1, len(COUNT_LEVELS))
    for index, (ax, count_level) in enumerate(zip(axes, COUNT_LEVELS)):
        result = first if index == 0 else simulate_noisy_image(
            fractions * count_level, spectra, detector,
            fluorophore_names=FLUOROPHORES,
            gaussian_sigma=GAUSSIAN_SIGMA, rng=rng,
        )
        ax.imshow(_rgb_preview(result.noisy_image, count_level, display_gain))
        ax.set_title(f"{count_level:,} counts / pixel")
        ax.axis("off")

    fig.suptitle("Increasing relative noise as photon counts fall")
    fig.text(
        0.5, 0.01,
        f"Same mixture · Poisson counts + Gaussian σ={GAUSSIAN_SIGMA:g} counts/channel · brightness normalized for comparison",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.82, wspace=0.08)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    return output_path


if __name__ == "__main__":
    saved_path = create_preview()
    print(f"Saved noise-level preview to {saved_path}")
