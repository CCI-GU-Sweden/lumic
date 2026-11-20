"""Show spectral distinctness in image coordinates and hull spectral space.

Run from Python with ``python -m examples.visualize_spectral_uniqueness``.
Uses offline FPbase spectra and the approximate LSM 980 detector profile.
"""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure

from lumic import spectral_uniqueness_map, spectral_uniqueness_rgb
from lumic.simulation import (
    Ellipse, GaussianPSF, SamplingGrid, Scene2D, SpectralDetector,
    load_offline_emission_spectrum, render_scene, simulate_linear_mixture,
)
from lumic.utils import RGB_from_spec


# User settings: physical lengths in micrometres, brightness in counts/pixel.
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
CIRCLES = (("EGFP", (6.2, 6.4)), ("EBFP2", (12.2, 6.4)),
           ("mCherry", (9.2, 11.6)))
RADIUS_UM = 5.2
COUNTS_PER_PIXEL = 850
IMAGE_SHAPE = (180, 180)
PIXEL_SIZE_UM = 0.1
DETECTOR_PROFILE = "980_32ch"
NPC = 3  # First coordinate is dropped, leaving a 2D spectral scatter plot.
MIN_TOTAL_INTENSITY = 50
# (label, PSF FWHM in micrometres, brightness multiplier)
CASES = (("No blur", 0.0, 1.0),
         ("No blur, 4× brighter", 0.0, 4.0),
         ("Blur FWHM 0.8 µm", 0.8, 1.0))
DEFAULT_OUTPUT = (Path(__file__).resolve().parents[1] / "data"
                  / "visual-check-spectral-uniqueness.png")


def create_scene():
    """Arrange three circles with overlapping central regions."""
    return Scene2D(FLUOROPHORES, tuple(
        Ellipse(center, (RADIUS_UM, RADIUS_UM), {name: COUNTS_PER_PIXEL})
        for name, center in CIRCLES
    ))


def create_spectral_image(scene, grid, detector, spectra, fwhm, brightness):
    """Render component counts and combine their sampled detector spectra."""
    psf = None if fwhm == 0 else GaussianPSF({name: fwhm for name in FLUOROPHORES})
    spatial = render_scene(scene, grid, psf, oversample=3)
    return simulate_linear_mixture(
        spatial.blurred_counts * brightness, spectra, detector.channel_centers,
        fluorophore_names=FLUOROPHORES,
    )


def run_analysis(spectral_image, reference=None):
    """Map normalized SVD distance using measured pixels, without ground truth."""
    return spectral_uniqueness_map(
        spectral_image, spectral_axis=-1, npc=NPC,
        min_total_intensity=MIN_TOTAL_INTENSITY, reference=reference,
    )


def create_preview(output_path=DEFAULT_OUTPUT):
    """Simulate, score, then show maps and their corresponding spectral points."""
    if NPC != 3:
        raise ValueError("this 2D scatter example requires NPC=3")
    scene = create_scene()
    grid = SamplingGrid(IMAGE_SHAPE, pixel_size_um=PIXEL_SIZE_UM)
    detector = SpectralDetector.from_profile(DETECTOR_PROFILE)
    sampled = np.array([
        detector.sample(*load_offline_emission_spectrum(name))
        for name in FLUOROPHORES
    ])
    spectra = sampled / sampled.sum(axis=1, keepdims=True)
    results = []
    reference = None
    for label, fwhm, brightness in CASES:
        simulation = create_spectral_image(scene, grid, detector, spectra,
                                           fwhm, brightness)
        analysis = run_analysis(simulation.image, reference)
        # A shared basis and center make all displayed scores comparable.
        if reference is None:
            reference = analysis
        results.append((label, simulation, analysis))

    maximum_score = max(float(np.nanmax(result.scores)) for _, _, result in results)
    maximum_counts = max(float(np.max(result.total_intensity)) for _, _, result in results)
    all_points = np.concatenate([result.coordinates[result.valid_mask]
                                 for _, _, result in results])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum(upper - lower, 0.01) * 0.08
    fig = Figure(figsize=(20, 4 * len(results)), dpi=135)
    axes = fig.subplots(len(results), 5, squeeze=False)
    score_colors = colormaps["magma"].copy()
    score_colors.set_bad("#505050")
    for row, (label, simulation, result) in enumerate(results):
        counts_ax, rgb_ax, score_ax, direction_ax, space_ax = axes[row]
        counts = counts_ax.imshow(result.total_intensity, cmap="gray",
                                  vmin=0, vmax=maximum_counts)
        counts_ax.set_title(f"{label}\nTotal detector counts")
        fig.colorbar(counts, ax=counts_ax, fraction=0.046, pad=0.03)
        rgb_ax.imshow(RGB_from_spec(np.moveaxis(simulation.image, -1, 0), do_norm=True))
        rgb_ax.set_title("Spectral RGB (display-scaled)")
        score_ax.set_facecolor("#252525")
        scores = score_ax.imshow(np.ma.masked_invalid(result.scores), cmap=score_colors,
                                 vmin=0, vmax=maximum_score)
        score_ax.set_title("Distance from spectral center\nGray pixels are excluded")
        fig.colorbar(scores, ax=score_ax, fraction=0.046, pad=0.03)
        direction_rgb = spectral_uniqueness_rgb(result, max_distance=maximum_score)
        direction_ax.imshow(direction_rgb)
        direction_ax.set_title("Hue = direction · color strength = distance\nBlack = center; gray = excluded")
        # A wheel connects image hues to the two displayed spectral axes.
        wheel_ax = direction_ax.inset_axes([0.70, 0.01, 0.28, 0.28])
        y, x = np.mgrid[-1:1:81j, -1:1:81j]
        radius = np.hypot(x, y)
        wheel = hsv_to_rgb(np.stack((np.mod(np.arctan2(y, x) / (2*np.pi), 1),
                                     np.minimum(radius, 1), np.minimum(radius, 1)), axis=-1))
        wheel[radius > 1] = 1
        wheel_ax.imshow(wheel, origin="lower", extent=(-1, 1, -1, 1))
        wheel_ax.set(xlabel="+x →", ylabel="+y ↑", xticks=[], yticks=[])
        wheel_ax.tick_params(labelsize=6)
        points = result.coordinates[result.valid_mask]
        space_ax.scatter(points[:, 0], points[:, 1], c=direction_rgb[result.valid_mask],
                         s=4, alpha=0.35, rasterized=True)
        space_ax.scatter(*result.center_coordinates, marker="+", c="tab:red",
                         s=100, linewidths=2, label="Reference center")
        space_ax.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]),
                     ylim=(lower[1] - margin[1], upper[1] + margin[1]),
                     xlabel="SVD coordinate 2 / coordinate 1",
                     ylabel="SVD coordinate 3 / coordinate 1")
        space_ax.set_title("Same pixels in hull spectral space")
        space_ax.legend(fontsize=8)
        space_ax.grid(alpha=0.2)
        for ax in (counts_ax, rgb_ax, score_ax, direction_ax):
            ax.axis("off")
    fig.suptitle("Spectral uniqueness: distance in the normalized SVD space", fontsize=16)
    fig.text(0.5, 0.94,
             f"Shared basis and center from the first image · retain pixels ≥ "
             f"{MIN_TOTAL_INTENSITY} counts · no noise or unmixing fit", ha="center")
    fig.text(0.5, 0.015,
             "A distant spectrum is distinct from the reference population; "
             "the score does not establish purity.", ha="center")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.07, top=0.89,
                        wspace=0.4, hspace=0.35)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    return results, output_path


if __name__ == "__main__":
    _, path = create_preview()
    print(f"Saved spectral uniqueness preview to {path}")
