"""Render a three-fluorophore scene with and without approximate optical blur.

Run ``python -m examples.visualize_spatial_scene`` from ``Python``. The example
uses the saved FPbase emission curves and the ideal LSM 980 32-channel detector,
so it works offline. The PNG shows component count maps before and after blur;
the returned detector image remains available for later unmixing evaluations.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic.simulation import (
    Ellipse,
    GaussianPSF,
    Line,
    Rectangle,
    SamplingGrid,
    Scene2D,
    SpectralDetector,
    load_offline_emission_spectrum,
    render_scene,
    simulate_noisy_image,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "visual-check-spatial-scene.png"
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
COLORS = ("Blues", "Greens", "Reds")


def build_scene():
    """Make broad pure regions, overlaps, and structures thinner than the PSF."""
    return Scene2D(
        FLUOROPHORES,
        (
            Ellipse((4.2, 5.0), (3.0, 2.2), {"EBFP2": 900}, angle_deg=20),
            Rectangle((8.0, 7.8), (4.2, 7.0), {"EGFP": 700}, angle_deg=25),
            Ellipse((12.6, 9.4), (2.4, 3.5), {"mCherry": 850}, angle_deg=-15),
            # One object carries two fluorophores; the maps retain both truths.
            Ellipse((9.8, 11.8), (1.3, 0.9),
                    {"EGFP": 350, "mCherry": 450}),
            Line((3.0, 13.8), (13.7, 4.0), 0.06, {"EBFP2": 800}),
            Line((3.5, 14.6), (14.2, 4.8), 0.06, {"EGFP": 650}),
            Line((4.1, 15.2), (14.8, 5.4), 0.06, {"mCherry": 750}),
        ),
    )


def _false_color(component_counts):
    """Map the known blue, green, red component counts to display colors."""
    scale = 1100.0
    return np.clip(
        component_counts[..., [2, 1, 0]] / scale,
        0, 1,
    )


def create_preview(output_path=DEFAULT_OUTPUT):
    """Save a comparison PNG and return the spatial and detector results."""
    scene = build_scene()
    grid = SamplingGrid((160, 160), pixel_size_um=0.1)
    # This is an ideal small-pinhole approximation. A measured FWHM can replace
    # it with GaussianPSF({name: measured_width_um, ...}).
    psf = GaussianPSF.approximate_confocal(
        1.3,
        {"EBFP2": 405, "EGFP": 488, "mCherry": 561},
        {"EBFP2": 448, "EGFP": 509, "mCherry": 610},
    )
    spatial = render_scene(scene, grid, psf)

    spectra = tuple(load_offline_emission_spectrum(name) for name in FLUOROPHORES)
    detector_result = simulate_noisy_image(
        spatial.blurred_counts,
        spectra,
        SpectralDetector.from_profile("980_32ch"),
        fluorophore_names=FLUOROPHORES,
        rng=np.random.default_rng(2026),
    )

    fig = Figure(figsize=(13, 6.3), dpi=150)
    axes = fig.subplots(2, 4)
    for row, maps in enumerate((spatial.source_counts, spatial.blurred_counts)):
        axes[row, 0].imshow(_false_color(maps), interpolation="nearest")
        axes[row, 0].set_title(
            "Scene before blur" if row == 0 else "After optical blur"
        )
        for component, name in enumerate(FLUOROPHORES):
            axes[row, component + 1].imshow(
                maps[..., component], cmap=COLORS[component],
                vmin=0, vmax=1100, interpolation="nearest",
            )
            axes[row, component + 1].set_title(name)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Spatial scene: broad objects, shared labels, and thin nearby lines")
    widths_nm = "/".join(
        f"{psf.fwhm_um[name] * 1000:.0f}" for name in FLUOROPHORES
    )
    fig.text(
        0.5, 0.01,
        f"100 nm scan spacing · NA 1.3 · ideal PSF FWHM (B/G/R): {widths_nm} nm · same count scale",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.07, top=0.91,
                        wspace=0.04, hspace=0.13)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    return spatial, detector_result


if __name__ == "__main__":
    spatial, detector_result = create_preview()
    print(f"Saved spatial preview to {DEFAULT_OUTPUT}")
    print(f"Detector image shape: {detector_result.expected_counts.shape}")
