"""Render three overlapping fluorescent circles in a square field of view.

Run ``python -m examples.visualize_overlapping_circles`` from ``Python``.
The saved PNG compares the ideal component maps, an approximate confocal-PSF
blurred version, and the detector image converted with ``RGB_from_spec``. The
circles carry EBFP2, EGFP, and mCherry.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic import (
    Spectrum,
    essential_spectral_pixles,
    load_spectra_json,
    mcr_als,
    normalize_components,
    positive_least_squares,
    save_spectra_json,
    unique_profiles,
)
from lumic.evaluation import concentration_error_maps, evaluate_factors
from lumic.simulation import (
    Ellipse,
    GaussianPSF,
    SamplingGrid,
    Scene2D,
    SpectralDetector,
    load_offline_emission_spectrum,
    render_scene,
    simulate_noisy_image,
)
from lumic.utils import RGB_from_spec


# ---------------------------------------------------------------------------
# User-editable example settings
# Circle centres, radius, and scan spacing are in micrometres. The count level
# is the expected signal per scan pixel inside a broad, uniform circle region.
# Change these values to create a different scene or imaging setup.
# ---------------------------------------------------------------------------
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
# Each tuple is (fluorophore name, circle centre (x, y)). The display colors
# below assume EBFP2, EGFP, and mCherry are ordered blue, green, and red.
CIRCLE_LAYOUT_UM = (
    ("EGFP", (6.2, 6.4)),
    ("EBFP2", (12.2, 6.4)),
    ("mCherry", (9.2, 11.6)),
)
CIRCLE_RADIUS_UM = 5.2
CIRCLE_COUNTS_PER_PIXEL = 850
IMAGE_SHAPE = (180, 180)  # rows, columns
PIXEL_SIZE_UM = 0.1
OBJECTIVE_NA = 1.3
EXCITATION_NM = {"EBFP2": 405, "EGFP": 488, "mCherry": 561}
EMISSION_NM = {"EBFP2": 448, "EGFP": 509, "mCherry": 610}
DETECTOR_PROFILE = "980_32ch"
MCR_ITERATIONS = 500
MCR_TOL_PERCENT = 0.1
HULL_COMPONENTS = len(FLUOROPHORES)
HULL_UNIQUE_PROFILES_F_PERCENT = 0.5

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "visual-check-overlapping-circles.png"
)
DEFAULT_RECOVERY_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "visual-check-overlapping-circles-recovery.png"
)
DEFAULT_SPECTRA_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "output"
    / "overlapping-circles-spectra"
)
COLORS = ("Blues", "Greens", "Reds")


def build_scene():
    """Arrange three equal circles so their overlaps meet near the centre."""
    return Scene2D(
        FLUOROPHORES,
        tuple(
            Ellipse(
                center_um,
                (CIRCLE_RADIUS_UM, CIRCLE_RADIUS_UM),
                {name: CIRCLE_COUNTS_PER_PIXEL},
            )
            for name, center_um in CIRCLE_LAYOUT_UM
        ),
    )


def _false_color(component_counts):
    """Display EBFP2, EGFP, and mCherry as blue, green, and red channels."""
    return np.clip(
        component_counts[..., [2, 1, 0]] / CIRCLE_COUNTS_PER_PIXEL, 0, 1
    )


def run_blind_unmixing(spectral_image, detector):
    """Run hull selection, unique-profile selection, and MCR-ALS on an image.

    Simulation truth is deliberately not an argument here. This function
    returns the raw factor estimates in their arbitrary model order; matching
    them to fluorophore labels happens later in ``compare_to_simulation``.
    """
    spectral_image = np.asarray(spectral_image, dtype=float)
    if spectral_image.ndim != 3 or spectral_image.shape[-1] != detector.n_channels:
        raise ValueError(
            "spectral_image must have shape (y, x, detector channel)"
        )
    pixels = spectral_image.reshape(-1, detector.n_channels)
    totals = pixels.sum(axis=1)
    # Gaussian blur leaves very small tails around the circles. Exclude values
    # below numerical relevance so they do not dominate hull normalization.
    valid_pixels = totals > totals.max() * 1e-10
    fit_pixels = pixels[valid_pixels]

    # The legacy hull helper uses NumPy's global RNG through TruncatedSVD.
    # Fix and restore its state so repeated runs produce the same figure
    # without changing random state for the caller.
    random_state = np.random.get_state()
    np.random.seed(2026)
    try:
        candidates, hull_indices = essential_spectral_pixles(
            fit_pixels, npc=HULL_COMPONENTS
        )
        starting_spectra, selected_indices = unique_profiles(
            candidates.T,
            nr=HULL_COMPONENTS,
            f=HULL_UNIQUE_PROFILES_F_PERCENT,
        )
    finally:
        np.random.set_state(random_state)
    starting_spectra = normalize_components(starting_spectra, do_norm="L1")
    initial_counts = positive_least_squares(
        fit_pixels, starting_spectra, do_norm=False
    )
    try:
        fitted_counts, fitted_spectra, history, diagnostics = mcr_als(
            fit_pixels,
            starting_spectra,
            nIter=MCR_ITERATIONS,
            normL="L1",
            tol_percent=MCR_TOL_PERCENT,
            return_diagnostics=True,
        )
        status = (
            f"{diagnostics.iterations_completed} updates, "
            f"best={diagnostics.best_iteration}, {diagnostics.stop_reason}"
        )
    except RuntimeError as error:
        # Preserve a useful selection-only view if an individual NNLS update
        # cannot complete; do not label it as an optimized result.
        status = (
            f"MCR-ALS failed; showing initialization "
            f"({type(error).__name__}: {error})"
        )
        fitted_counts = initial_counts
        fitted_spectra = starting_spectra
        history = np.asarray([], dtype=float)

    return {
        "initial_counts": initial_counts,
        "initial_spectra": starting_spectra,
        "fitted_counts": fitted_counts,
        "fitted_spectra": fitted_spectra,
        "fit_pixels": fit_pixels,
        "valid_pixel_mask": valid_pixels.reshape(spectral_image.shape[:2]),
        "history": history,
        "status": status,
        "valid_pixel_count": int(valid_pixels.sum()),
        "hull_candidate_count": len(hull_indices),
        "selected_indices": selected_indices,
    }


def _save_comparison_figure(path, spatial, detector, blind_result, comparison):
    """Plot blind estimates beside simulation truth after factor matching."""
    fig = Figure(figsize=(13, 11), dpi=145)
    axes = fig.subplots(4, len(FLUOROPHORES))
    initial = comparison["initial"]
    fitted = comparison["fitted"]
    initial_spectra = initial.estimated_spectra
    recovered_spectra = fitted.estimated_spectra
    wavelengths = detector.channel_centers
    recovered_maps = comparison["recovered_maps"]
    true_spectra = comparison["true_spectra"]
    errors = concentration_error_maps(
        spatial.blurred_counts, true_spectra, recovered_maps, recovered_spectra
    ).signed

    for component, name in enumerate(FLUOROPHORES):
        axis = axes[0, component]
        axis.plot(
            wavelengths, true_spectra[component], color="black",
            label="Known spectrum",
        )
        axis.plot(
            wavelengths, initial_spectra[component], color="tab:blue",
            linestyle=":", label="Hull + unique profiles",
        )
        axis.plot(
            wavelengths, recovered_spectra[component], color="tab:orange",
            linestyle="--", label="After MCR-ALS",
        )
        axis.set_title(f"{name} spectrum")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Fraction of detector counts")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)

        true_map = spatial.blurred_counts[..., component]
        recovered_map = recovered_maps[..., component]
        difference = errors[..., component]
        max_counts = max(float(true_map.max()), float(recovered_map.max()))
        max_difference = max(float(np.abs(difference).max()), 1e-6)
        for axis, values, title, limits, cmap in (
            (axes[1, component], true_map, f"Known {name} counts", (0, max_counts), "viridis"),
            (axes[2, component], recovered_map, f"Recovered {name} counts", (0, max_counts), "viridis"),
            (axes[3, component], difference, "Recovered − known", (-max_difference, max_difference), "coolwarm"),
        ):
            image = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Blind spectral recovery from the simulated detector image",
        fontsize=13,
    )
    fig.text(
        0.5, 0.92,
        f"{blind_result['status']} · "
        f"{blind_result['valid_pixel_count']:,} fitted pixels · "
        f"spectral rel. L2 {fitted.spectral_relative_l2_global:.2g} · "
        f"abundance RMSE {fitted.abundance_rmse_global:.2g} counts · "
        f"reconstruction rel. L2 {fitted.reconstruction_relative_l2:.2g}",
        ha="center", fontsize=8,
    )
    fig.text(
        0.5, 0.015,
        f"Starting spectra selected from detector pixels only · "
        f"{blind_result['hull_candidate_count']} convex-hull candidates · "
        "truth shown only for comparison",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.87,
                        wspace=0.35, hspace=0.38)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path


def compare_to_simulation(blind_result, spatial, spectral_result, detector,
                          output_path=DEFAULT_RECOVERY_OUTPUT):
    """Match blind factors to known fluorophores and measure recovery errors.

    The ground-truth spectra and maps enter the workflow here, after
    ``run_blind_unmixing`` has finished. They are used only to match anonymous
    estimated factors to fluorophore names, calculate errors, and make the
    comparison figure.
    """
    valid_pixels = blind_result["valid_pixel_mask"].ravel()
    observed_pixels = spectral_result.expected_counts.reshape(
        -1, detector.n_channels
    )[valid_pixels]
    true_counts = spatial.blurred_counts.reshape(
        -1, len(FLUOROPHORES)
    )[valid_pixels]
    true_spectra = spectral_result.ideal.spectra

    # Save blind estimates before matching to simulation truth. Anonymous
    # names make it clear that the solver did not identify fluorophores.
    output_dir = DEFAULT_SPECTRA_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_objects = [
        Spectrum(values, detector.channel_centers, name=f"Initial component {index + 1}",
                 provenance={"method": "convex hull and unique profiles",
                             "normalization": "L1", "source": "simulated detector pixels"})
        for index, values in enumerate(blind_result["initial_spectra"])
    ]
    fitted_objects = [
        Spectrum(values, detector.channel_centers, name=f"Blind component {index + 1}",
                 provenance={"method": "MCR-ALS", "status": blind_result["status"],
                             "source": "simulated detector pixels"})
        for index, values in enumerate(blind_result["fitted_spectra"])
    ]
    true_objects = [
        Spectrum(values, detector.channel_centers, name=name,
                 provenance={"source": "offline FPbase simulation truth"})
        for name, values in zip(FLUOROPHORES, true_spectra)
    ]
    save_spectra_json(initial_objects, output_dir / "initial-blind-spectra.json")
    save_spectra_json(fitted_objects, output_dir / "fitted-blind-spectra.json")
    save_spectra_json(true_objects, output_dir / "simulation-truth-spectra.json")
    # Demonstrate the public loader and retain objects for downstream plotting.
    loaded_fitted_spectra = load_spectra_json(output_dir / "fitted-blind-spectra.json")
    print(f"Saved and reloaded {len(loaded_fitted_spectra)} blind spectra in {output_dir}")

    # Evaluation aligns component order only now, after blind fitting.
    initial = evaluate_factors(
        observed_pixels,
        true_counts,
        true_spectra,
        blind_result["initial_counts"],
        blind_result["initial_spectra"],
    )
    fitted = evaluate_factors(
        observed_pixels,
        true_counts,
        true_spectra,
        blind_result["fitted_counts"],
        blind_result["fitted_spectra"],
    )

    recovered_maps = np.zeros_like(spatial.blurred_counts).reshape(
        -1, len(FLUOROPHORES)
    )
    recovered_maps[valid_pixels] = fitted.estimated_abundances
    comparison = {
        "initial": initial,
        "fitted": fitted,
        "recovered_maps": recovered_maps.reshape(spatial.blurred_counts.shape),
        "true_spectra": true_spectra,
    }
    figure_path = _save_comparison_figure(
        output_path, spatial, detector, blind_result, comparison
    )
    return comparison, figure_path


def create_scene_preview(output_path=DEFAULT_OUTPUT):
    """Simulate the scene and save component-map and detector-RGB previews.

    Returns the spatial render, simulated spectral image, and detector. The
    spectral image can be passed to ``run_blind_unmixing`` without passing
    any simulation truth.
    """
    scene = build_scene()
    grid = SamplingGrid(IMAGE_SHAPE, pixel_size_um=PIXEL_SIZE_UM)
    psf = GaussianPSF.approximate_confocal(
        OBJECTIVE_NA,
        EXCITATION_NM,
        EMISSION_NM,
    )
    spatial = render_scene(scene, grid, psf)
    detector = SpectralDetector.from_profile(DETECTOR_PROFILE)
    spectra = tuple(load_offline_emission_spectrum(name) for name in FLUOROPHORES)
    spectral_result = simulate_noisy_image(
        spatial.blurred_counts,
        spectra,
        detector,
        fluorophore_names=FLUOROPHORES,
        rng=np.random.default_rng(2026),
    )
    detector_channels = np.moveaxis(spectral_result.expected_counts, -1, 0)
    # RGB_from_spec converts channel means to integers; scale first so low
    # expected counts are not truncated before its 8-bit display normalization.
    display_gain = 10_000.0 / max(float(detector_channels.max()), 1.0)
    detector_rgb = RGB_from_spec(
        detector_channels * display_gain, do_norm=True
    )

    fig = Figure(figsize=(14, 5.4), dpi=160, facecolor="black")
    axes = fig.subplots(1, 3)
    for ax, maps, title in zip(
        axes[:2],
        (spatial.source_counts, spatial.blurred_counts),
        ("Ideal component maps", "After approximate optical blur"),
    ):
        ax.imshow(_false_color(maps), interpolation="nearest", origin="upper")
        ax.set_title(title, color="white", pad=10)
        ax.set_facecolor("black")
        ax.axis("off")
    axes[2].imshow(detector_rgb, interpolation="nearest", origin="upper")
    axes[2].set_title("980 32-channel detector → RGB_from_spec", color="white", pad=10)
    axes[2].set_facecolor("black")
    axes[2].axis("off")

    widths_nm = "/".join(
        f"{psf.fwhm_um[name] * 1000:.0f}" for name in FLUOROPHORES
    )
    fig.suptitle(
        "Three overlapping fluorescent circles", color="white", fontsize=15
    )
    fig.text(
        0.5, 0.035,
        f"EBFP2 (blue) · EGFP (green) · mCherry (red) · "
        f"{PIXEL_SIZE_UM * 1000:g} nm scan spacing · NA {OBJECTIVE_NA:g} · "
        f"PSF FWHM: {widths_nm} nm · "
        "offline FPbase spectra · ideal detector counts",
        ha="center",
        color="white",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.85, wspace=0.04)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="black")
    return spatial, spectral_result, detector


def main():
    """Run the example in visible, human-readable stages."""
    print("1/3: Create spatial sample and simulate the detector image")
    spatial, spectral_result, detector = create_scene_preview()
    print(f"Saved overlapping-circles preview to {DEFAULT_OUTPUT}")

    print("2/3: Blindly unmix the detector image (no true spectra supplied)")
    blind_result = run_blind_unmixing(
        spectral_result.expected_counts, detector
    )
    print(
        f"Selected {len(blind_result['selected_indices'])} starting spectra "
        f"from {blind_result['hull_candidate_count']} convex-hull candidates"
    )
    print(blind_result["status"])

    print("3/3: Compare blind estimates with simulation ground truth")
    comparison, recovery_path = compare_to_simulation(
        blind_result,
        spatial,
        spectral_result,
        detector,
    )
    print(f"Saved blind-recovery comparison to {recovery_path}")
    print(
        f"Spectral relative L2: "
        f"{comparison['fitted'].spectral_relative_l2_global:.3g}; "
        f"abundance RMSE: {comparison['fitted'].abundance_rmse_global:.3g} counts"
    )


if __name__ == "__main__":
    main()
