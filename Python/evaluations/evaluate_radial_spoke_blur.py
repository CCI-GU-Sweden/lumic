"""Noiseless four-component radial-spoke blur experiment.

Run from Python with ``python -m evaluations.evaluate_radial_spoke_blur``.
The true spectra and maps are used only after blind selection/fitting.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.spatial import QhullError
from tqdm.auto import tqdm

from lumic import (
    essential_spectral_pixles,
    mcr_als,
    normalize_components,
    positive_least_squares,
    unique_profiles,
)
from lumic.evaluation import concentration_error_maps, evaluate_factors
from lumic.simulation import (
    GaussianPSF,
    Line,
    Rectangle,
    SamplingGrid,
    Scene2D,
    SpectralDetector,
    load_offline_emission_spectrum,
    render_scene,
    simulate_linear_mixture,
)


# User settings. All lengths are micrometres; counts are expected detector
# counts per scan pixel. FWHM is a controlled stress-test parameter, not a
# measured objective response. The zero-width case applies no PSF.
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
BACKGROUND_NAME = "broad_500nm_background"
COMPONENTS = (*FLUOROPHORES, BACKGROUND_NAME)
IMAGE_SHAPE = (64, 64)
PIXEL_SIZE_UM = 0.12
N_SPOKES = 12
SPOKE_LENGTH_UM = 3.2
SPOKE_WIDTH_UM = 0.18
SPOKE_COUNTS = 900.0
BACKGROUND_COUNTS = 24.0
BACKGROUND_CENTER_NM = 500.0
BACKGROUND_SIGMA_NM = 105.0
SPOKE_SEED = 2026
BLUR_FWHM_UM = (0.0, 0.30, 0.65, 1.20, 2.00)
PURITY_THRESHOLDS = (0.90, 0.95, 0.99)
MAX_FIT_PIXELS = 512
MCR_ITERATIONS = 4000
MCR_TOL_PERCENT = 0.1
UNIQUE_PROFILES_F_PERCENT = 0.5

RESULTS = Path(__file__).resolve().parent / "results" / "radial_spoke_blur"


def spoke_labels(n_spokes=N_SPOKES, seed=SPOKE_SEED):
    """Assign balanced labels randomly, ensuring every fluorophore appears."""
    if n_spokes < len(FLUOROPHORES):
        raise ValueError("need at least one spoke per fluorophore")
    labels = np.arange(n_spokes) % len(FLUOROPHORES)
    return np.random.default_rng(seed).permutation(labels)


def build_scene():
    """Create thin spokes over a weak, spatially uniform fourth component."""
    field_width = IMAGE_SHAPE[1] * PIXEL_SIZE_UM
    field_height = IMAGE_SHAPE[0] * PIXEL_SIZE_UM
    center = (field_width / 2, field_height / 2)
    shapes = [Rectangle(center, (50.0, 50.0), {BACKGROUND_NAME: BACKGROUND_COUNTS})]
    for index, label in enumerate(spoke_labels()):
        angle = 2 * np.pi * index / N_SPOKES
        endpoint = (
            center[0] + SPOKE_LENGTH_UM * np.cos(angle),
            center[1] + SPOKE_LENGTH_UM * np.sin(angle),
        )
        shapes.append(Line(center, endpoint, SPOKE_WIDTH_UM,
                           {FLUOROPHORES[label]: SPOKE_COUNTS}))
    return Scene2D(COMPONENTS, tuple(shapes))


def detector_spectra(detector):
    """Sample three saved FPbase curves and one synthetic broad background."""
    curves = [load_offline_emission_spectrum(name) for name in FLUOROPHORES]
    wavelengths = np.arange(400.0, 721.0)
    intensities = np.exp(
        -0.5 * ((wavelengths - BACKGROUND_CENTER_NM) / BACKGROUND_SIGMA_NM) ** 2
    )
    curves.append((wavelengths, intensities))
    sampled = np.array([detector.sample(*curve) for curve in curves])
    return sampled / sampled.sum(axis=1, keepdims=True)


def mixture_statistics(counts, thresholds=PURITY_THRESHOLDS,
                       min_fluor_counts=BACKGROUND_COUNTS):
    """Measure purity where fluorophore signal is at least the BG count level."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or counts.shape[-1] != len(COMPONENTS):
        raise ValueError("counts must have shape (y, x, four components)")
    total = counts.sum(axis=-1)
    fluor_total = counts[..., :len(FLUOROPHORES)].sum(axis=-1)
    # Blurred Gaussian tails can be nonzero almost everywhere. Requiring at
    # least the background count level avoids calling negligible tails pure.
    fluor_foreground = fluor_total >= min_fluor_counts
    # Count only fluorophore-dominated pixels. Background-only pixels are
    # spectrally pure BG, but cannot show whether any fluorophore stays pure.
    all_fraction = np.divide(counts[..., :len(FLUOROPHORES)].max(axis=-1),
                             total, out=np.zeros_like(total),
                             where=fluor_foreground)
    fluor_fraction = np.divide(counts[..., :len(FLUOROPHORES)].max(axis=-1),
                               fluor_total, out=np.zeros_like(total),
                               where=fluor_foreground)
    stats = {
        "pixels": int(total.size),
        "min_fluor_counts": min_fluor_counts,
        "fluor_signal_pixels": int(fluor_foreground.sum()),
        "max_fluor_fraction_including_bg": float(all_fraction.max()),
        "max_fraction_fluor_only": float(fluor_fraction.max()),
    }
    for threshold in thresholds:
        key = f"near_pure_{round(100 * threshold):02d}pct"
        stats[f"{key}_including_bg"] = int(
            (fluor_foreground & (all_fraction >= threshold)).sum()
        )
        stats[f"{key}_fluor_only"] = int(
            (fluor_foreground & (fluor_fraction >= threshold)).sum()
        )
    return stats


def _blind_unmix(image, iteration_callback=None):
    """Find starts from detector pixels, refine on a fixed subset, map all pixels."""
    pixels = image.reshape(-1, image.shape[-1])
    state = np.random.get_state()
    np.random.seed(SPOKE_SEED)  # TruncatedSVD inside the legacy hull helper.
    try:
        candidates, hull_indices = essential_spectral_pixles(
            pixels, npc=len(COMPONENTS)
        )
        starts, selected_indices = unique_profiles(
            candidates.T, nr=len(COMPONENTS), f=UNIQUE_PROFILES_F_PERCENT
        )
    finally:
        np.random.set_state(state)
    starts = normalize_components(starts, do_norm="L1")
    chosen = np.random.default_rng(SPOKE_SEED).choice(
        len(pixels), min(MAX_FIT_PIXELS, len(pixels)), replace=False
    )
    # Always include the actual selected starts, but cap the ordinary sample.
    # The number of hull vertices grows sharply as blur smooths the image;
    # including every vertex would change the fit-set size across conditions.
    selected_pixels = hull_indices[np.asarray(selected_indices, dtype=int)]
    chosen = np.unique(np.concatenate((chosen, selected_pixels)))
    fit_pixels = pixels[chosen]
    initial_counts = positive_least_squares(pixels, starts, do_norm=False)
    _, fitted_spectra, history, diagnostics = mcr_als(
        fit_pixels, starts, nIter=MCR_ITERATIONS, normL="L1",
        tol_percent=MCR_TOL_PERCENT, return_diagnostics=True,
        iteration_callback=iteration_callback,
    )
    # Project the recovered spectra over the entire image, including pixels
    # not used for the iterative spectrum refinement.
    fitted_counts = positive_least_squares(pixels, fitted_spectra, do_norm=False)
    return {
        "initial_counts": initial_counts,
        "initial_spectra": starts,
        "fitted_counts": fitted_counts,
        "fitted_spectra": fitted_spectra,
        "history": history,
        "diagnostics": diagnostics,
        "hull_candidates": len(hull_indices),
        "selected_indices": selected_indices,
        "fit_pixels": len(chosen),
    }


def _case_figure(path, fwhm, spatial, spectra, start, final, status):
    """Show true/selected/fitted spectra and true/recovered abundance maps."""
    fig = Figure(figsize=(15, 11), dpi=130)
    axes = fig.subplots(4, len(COMPONENTS))
    wavelengths = SpectralDetector.from_profile("980_32ch").channel_centers
    # Keep the failed-run display as zero recovery, not a successful fit.
    recovered = (final.estimated_abundances.reshape(spatial.blurred_counts.shape)
                 if final is not None else np.zeros_like(spatial.blurred_counts))
    recovered_spectra = final.estimated_spectra if final is not None else spectra
    errors = concentration_error_maps(
        spatial.blurred_counts, spectra, recovered, recovered_spectra
    ).signed
    for k, name in enumerate(COMPONENTS):
        ax = axes[0, k]
        ax.plot(wavelengths, spectra[k], "k-", label="True")
        if start is not None:
            ax.plot(wavelengths, start.estimated_spectra[k], ":", color="tab:blue",
                    label="Selected start")
        if final is not None:
            ax.plot(wavelengths, final.estimated_spectra[k], "--", color="tab:orange",
                    label="After MCR-ALS")
        ax.set_title(name)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Count fraction")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
        true_map = spatial.blurred_counts[..., k]
        recovered_map = (final.estimated_abundances[:, k].reshape(IMAGE_SHAPE)
                         if final is not None else np.zeros(IMAGE_SHAPE))
        delta = errors[..., k]
        scale = max(float(true_map.max()), float(recovered_map.max()), 1.0)
        diff_scale = max(float(np.abs(delta).max()), 1.0)
        for row, values, title, limits, cmap in (
            (1, true_map, "True counts", (0, scale), "viridis"),
            (2, recovered_map, "Recovered counts", (0, scale), "viridis"),
            (3, delta, "Recovered − true", (-diff_scale, diff_scale), "coolwarm"),
        ):
            axis = axes[row, k]
            img = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(img, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle(f"Radial spokes · PSF FWHM {fwhm:g} µm · noiseless", fontsize=15)
    fig.text(0.5, 0.925, status, ha="center", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.88,
                        wspace=0.35, hspace=0.35)
    fig.savefig(path, facecolor="white")


def _overview_figure(path, cases):
    fig = Figure(figsize=(14, 8), dpi=140)
    axes = fig.subplots(2, 3)
    fwhms = [case["blur_fwhm_um"] for case in cases]
    for ax, key, title in (
        (axes[0, 0], "near_pure_95pct_including_bg", "Pixels ≥95% one fluorophore, BG included"),
        (axes[0, 1], "near_pure_95pct_fluor_only", "Pixels ≥95% one fluorophore, BG excluded"),
        (axes[0, 2], "final_spectral_relative_l2_global", "Spectral shape error"),
        (axes[1, 0], "final_abundance_rmse_global", "Abundance RMSE (counts)"),
        (axes[1, 1], "final_reconstruction_relative_l2", "Reconstruction relative L2"),
        (axes[1, 2], "hull_candidates", "Convex-hull candidate pixels"),
    ):
        ax.plot(fwhms, [case[key] for case in cases], "o-")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PSF FWHM (µm)")
        ax.grid(alpha=0.25)
    fig.suptitle("Four-component radial-spoke blur experiment · no noise")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.89,
                        wspace=0.3, hspace=0.35)
    fig.savefig(path, facecolor="white")


def save_reports(output_dir, rows):
    """Write tables and overview from completed case rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _overview_figure(output_dir / "overview.png", rows)
    with (output_dir / "summary.md").open("w") as file:
        file.write("# Noiseless radial-spoke blur benchmark\n\n")
        file.write(f"Three fluorophores: {', '.join(FLUOROPHORES)}; background: "
                   f"broad Gaussian spectrum centered at {BACKGROUND_CENTER_NM:g} nm "
                   f"(σ={BACKGROUND_SIGMA_NM:g} nm). Spokes contribute "
                   f"{SPOKE_COUNTS:g} counts; the uniform background contributes "
                   f"{BACKGROUND_COUNTS:g} counts per pixel. Detector: idealized "
                   "LSM 980 32-channel profile. No noise was added.\n\n")
        file.write("The zero-blur case is an unblurred control. 'Near pure' "
                   "counts fluorophore-dominated pixels while including the "
                   "background in the denominator. Foreground statistics require "
                   f"at least {BACKGROUND_COUNTS:g} fluorophore counts per pixel; "
                   "background-only pixels and negligible blur tails are excluded. "
                   "A low reconstruction error does not establish correct "
                   "spectra or abundances. Fitted factor error may be larger "
                   "than the starting error, so both are shown. Failed runs "
                   f"have missing accuracy metrics. MCR-ALS has a limit of "
                   f"{MCR_ITERATIONS} iterations; convergence status reports "
                   "whether its stopping criterion was met.\n\n")
        file.write("| PSF FWHM (µm) | ≥95% fluorophore pixels (BG included) | "
                   "Start spectral L2 | Fitted spectral L2 | Start abundance "
                   "RMSE | Fitted abundance RMSE | Reconstruction L2 | "
                   "Iterations | Status |\n")
        file.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            file.write(
                f"| {row['blur_fwhm_um']:g} | "
                f"{row['near_pure_95pct_including_bg']} | "
                f"{row['initial_spectral_relative_l2_global']:.3g} | "
                f"{row['final_spectral_relative_l2_global']:.3g} | "
                f"{row['initial_abundance_rmse_global']:.3g} | "
                f"{row['final_abundance_rmse_global']:.3g} | "
                f"{row['final_reconstruction_relative_l2']:.3g} | "
                f"{row['iterations']} | "
                f"{row['status']} |\n"
            )


def main(output_dir=RESULTS):
    """Generate the scene once, then vary only optical blur."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    grid = SamplingGrid(IMAGE_SHAPE, pixel_size_um=PIXEL_SIZE_UM)
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = detector_spectra(detector)
    rows = []
    for fwhm in BLUR_FWHM_UM:
        psf = None if fwhm == 0 else GaussianPSF({name: fwhm for name in COMPONENTS})
        spatial = render_scene(scene, grid, psf, oversample=3)
        ideal = simulate_linear_mixture(
            spatial.blurred_counts, spectra, detector.channel_centers,
            fluorophore_names=COMPONENTS,
        )
        stats = mixture_statistics(spatial.blurred_counts)
        row = {"blur_fwhm_um": fwhm, **stats,
               "background_counts": BACKGROUND_COUNTS,
               "spoke_counts": SPOKE_COUNTS,
               "spoke_seed": SPOKE_SEED}
        initial = final = None
        try:
            with tqdm(total=MCR_ITERATIONS,
                      desc=f"PSF FWHM {fwhm:g} µm", unit="iter") as progress:
                blind = _blind_unmix(ideal.image,
                                     iteration_callback=lambda _: progress.update(1))
                progress.set_postfix_str(blind["diagnostics"].stop_reason)
            observed = ideal.image.reshape(-1, detector.n_channels)
            truth = spatial.blurred_counts.reshape(-1, len(COMPONENTS))
            initial = evaluate_factors(
                observed, truth, spectra,
                blind["initial_counts"], blind["initial_spectra"]
            )
            final = evaluate_factors(
                observed, truth, spectra,
                blind["fitted_counts"], blind["fitted_spectra"]
            )
            diagnostics = blind["diagnostics"]
            row.update({
                "status": diagnostics.stop_reason,
                "iterations": diagnostics.iterations_completed,
                "best_iteration": diagnostics.best_iteration,
                "hull_candidates": blind["hull_candidates"],
                "fit_pixels": blind["fit_pixels"],
                "initial_spectral_relative_l2_global": initial.spectral_relative_l2_global,
                "final_spectral_relative_l2_global": final.spectral_relative_l2_global,
                "initial_abundance_rmse_global": initial.abundance_rmse_global,
                "final_abundance_rmse_global": final.abundance_rmse_global,
                "final_reconstruction_relative_l2": final.reconstruction_relative_l2,
            })
            for k, name in enumerate(COMPONENTS):
                row[f"{name}_spectral_error"] = final.spectral_relative_l2_per_component[k]
                row[f"{name}_abundance_rmse"] = final.abundance_rmse_per_component[k]
        except (RuntimeError, ValueError, np.linalg.LinAlgError, QhullError) as error:
            # This is an exploratory benchmark. A selection or solver failure
            # is an observed result, and its metrics remain missing.
            row.update({"status": f"failed: {type(error).__name__}: {error}",
                        "iterations": "", "best_iteration": "",
                        "hull_candidates": "", "fit_pixels": "",
                        "initial_spectral_relative_l2_global": float("nan"),
                        "final_spectral_relative_l2_global": float("nan"),
                        "initial_abundance_rmse_global": float("nan"),
                        "final_abundance_rmse_global": float("nan"),
                        "final_reconstruction_relative_l2": float("nan")})
            for name in COMPONENTS:
                row[f"{name}_spectral_error"] = float("nan")
                row[f"{name}_abundance_rmse"] = float("nan")
        _case_figure(output_dir / f"case_{fwhm:.2f}um.png", fwhm,
                     spatial, spectra, initial, final, row["status"])
        rows.append(row)
        tqdm.write(f"  {row['status']}; near-pure fluorophore ≥95%: "
                   f"{row['near_pure_95pct_including_bg']}")

    save_reports(output_dir, rows)
    print(f"Saved results to {output_dir}")
    return rows


if __name__ == "__main__":
    main()
