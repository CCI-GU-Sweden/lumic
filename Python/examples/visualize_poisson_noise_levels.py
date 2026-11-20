"""Save spectral curves showing increasing relative Poisson noise.

Run ``python -m examples.visualize_poisson_noise_levels`` from ``Python``.
This offline example mixes saved FPbase EGFP and mCherry spectra, samples the
mixture with the LSM 980 32-channel detector profile, then lowers the expected
counts per pixel. Each panel shows the clean expected curve, one noisy pixel,
and the 10th–90th percentile across many pixels with the same mixture.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic.simulation import (
    SpectralDetector,
    load_offline_emission_spectrum,
    simulate_noisy_image,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUTPUT = DATA_DIR / "visual-check-poisson-noise-curves.png"
FLUOROPHORES = ("EGFP", "mCherry")
COUNT_LEVELS = (10000, 1000, 100, 30)
N_REPLICATES = 256


def create_preview(output_path=DEFAULT_OUTPUT):
    """Simulate replicate pixels and save the noise-level plot."""
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = [load_offline_emission_spectrum(name) for name in FLUOROPHORES]
    rng = np.random.default_rng(20260930)

    fig = Figure(figsize=(11, 4.3), dpi=150)
    axes = fig.subplots(1, len(COUNT_LEVELS), sharex=True, sharey=True)

    for ax, counts_per_pixel in zip(axes, COUNT_LEVELS):
        # All replicate pixels have the same 50:50 mixture and expected total
        # counts. Only independent Poisson sampling varies between pixels.
        component_counts = np.full(
            (1, N_REPLICATES, len(FLUOROPHORES)), counts_per_pixel / 2.0
        )
        result = simulate_noisy_image(
            component_counts,
            spectra,
            detector,
            fluorophore_names=FLUOROPHORES,
            gaussian_sigma=0.0,
            rng=rng,
        )

        expected_curve = result.expected_counts[0, 0] / counts_per_pixel
        noisy_curves = result.noisy_image[0] / counts_per_pixel
        lower, upper = np.percentile(noisy_curves, (10, 90), axis=0)
        example_curve = noisy_curves[0]
        wavelengths = detector.channel_centers

        ax.fill_between(
            wavelengths, lower, upper, color="tab:orange", alpha=0.22,
            label="10th–90th percentile across pixels",
        )
        ax.plot(
            wavelengths, expected_curve, color="black", linewidth=1.5,
            label="Clean expected spectrum",
        )
        ax.plot(
            wavelengths, example_curve, color="tab:orange", linewidth=0.9,
            label="One noisy pixel",
        )
        ax.set_title(f"{counts_per_pixel:,} expected counts / pixel")
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Fraction of total expected counts")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=3, frameon=False,
    )
    fig.suptitle("Poisson noise grows relative to signal as counts fall", y=0.99)
    fig.text(
        0.5, 0.01,
        "Same 50:50 EGFP/mCherry mixture in every panel · Poisson noise only · 256 replicate pixels",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.80, wspace=0.12)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    return output_path


if __name__ == "__main__":
    saved_path = create_preview()
    print(f"Saved Poisson-noise curve preview to {saved_path}")
