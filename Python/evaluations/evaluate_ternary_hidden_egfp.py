"""Evaluate data-driven recovery when EGFP is hidden in ternary mixtures.

Run ``python -m evaluations.evaluate_ternary_hidden_egfp`` from ``Python``.
The experiment is noiseless and uses the saved FPbase emission curves offline.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic import (
    essential_spectral_pixles,
    mcr_als,
    normalize_components,
    positive_least_squares,
    unique_profiles,
)
from lumic.evaluation import concentration_error_maps, evaluate_factors
from lumic.simulation import SpectralDetector, simulate_linear_mixture

from evaluations._component_reports import (
    component_metric_rows,
    save_component_metrics,
    save_component_recovery_figure,
)


# EBFP2 and mCherry have pure pixels along the bottom edge of each image.
# EGFP is titrated from a pure-pixel control to mixtures with no pure EGFP.
COMPONENTS = ("EBFP2", "EGFP", "mCherry")
HIDDEN_COMPONENT = "EGFP"
MAX_HIDDEN_FRACTIONS = (1.0, 0.95, 0.75, 0.50, 0.25)
IMAGE_SHAPE = (20, 20)
TOTAL_COUNTS_PER_PIXEL = 1000.0
MCR_ITERATIONS = 500
MCR_TOL_PERCENT = 0.1
UNIQUE_PROFILES_F_PERCENT = 0.5

PYTHON_DIR = Path(__file__).resolve().parents[1]
SPECTRA_DIR = PYTHON_DIR / "lumic" / "simulation" / "fpbase_data"
OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "ternary_hidden_egfp"
)


def _load_spectra():
    """Read the three offline curves and normalize their binned counts."""
    detector = SpectralDetector.from_profile("980_32ch")
    sampled = []
    for name in COMPONENTS:
        curve = np.loadtxt(
            SPECTRA_DIR / f"{name.lower()}_emission.csv",
            delimiter=",", skiprows=1,
        )
        sampled.append(detector.sample(curve[:, 0], curve[:, 1]))
    sampled = np.asarray(sampled)
    if np.any(sampled.sum(axis=1) <= 0):
        raise ValueError("Each chosen spectrum must have signal in the detector range")
    return detector, sampled / sampled.sum(axis=1, keepdims=True)


def _make_component_counts(max_egfp_fraction):
    """Map a square grid over a triangular composition region.

    The bottom edge spans pure EBFP2 to pure mCherry. Moving up increases
    EGFP to the requested maximum while retaining the EBFP2:mCherry ratio.
    Consequently, every level below 100% contains no pure EGFP pixel.
    """
    rows, columns = IMAGE_SHAPE
    y, x = np.indices(IMAGE_SHAPE, dtype=float)
    # ``imshow`` displays row zero at the top, so reverse the grid coordinate
    # to keep the two-component edge at the bottom of the plotted image.
    egfp_fraction = max_egfp_fraction * (rows - 1 - y) / (rows - 1)
    remaining_fraction = 1 - egfp_fraction
    mCherry_fraction = x / (columns - 1)
    fractions = np.stack((
        remaining_fraction * (1 - mCherry_fraction),
        egfp_fraction,
        remaining_fraction * mCherry_fraction,
    ), axis=-1)
    return fractions * TOTAL_COUNTS_PER_PIXEL, egfp_fraction


def _evaluate_level(max_egfp_fraction, detector, true_spectra):
    """Run hull selection, unique profiles, and MCR-ALS without access to truth."""
    true_counts, egfp_fraction = _make_component_counts(max_egfp_fraction)
    dataset = simulate_linear_mixture(
        true_counts,
        true_spectra,
        detector.channel_centers,
        fluorophore_names=COMPONENTS,
    )
    pixels = dataset.image.reshape(-1, detector.n_channels)

    # The selection stages see only pixel spectra. Truth is used below only
    # to match component order and quantify the known simulation errors.
    candidates, hull_indices = essential_spectral_pixles(pixels, npc=3)
    starting_spectra, selected_in_hull = unique_profiles(
        candidates.T, nr=3, f=UNIQUE_PROFILES_F_PERCENT
    )
    selected_indices = hull_indices[selected_in_hull]

    # Keep a valid starting-factor estimate so that a failure inside the
    # current NNLS solver still leaves useful, clearly labeled diagnostics.
    starting_spectra = normalize_components(starting_spectra, do_norm="L1")
    starting_counts = positive_least_squares(
        pixels, starting_spectra, do_norm=False
    )
    true_flat_counts = true_counts.reshape(-1, 3)
    initial = evaluate_factors(
        pixels, true_flat_counts, true_spectra,
        starting_counts, starting_spectra,
    )
    initial_fit = initial.reconstruction_relative_l2
    mcr_error = ""
    diagnostics = None
    try:
        fitted_counts, fitted_spectra, fit_history, diagnostics = mcr_als(
            pixels, starting_spectra, nIter=MCR_ITERATIONS, normL="L1",
            tol_percent=MCR_TOL_PERCENT, return_diagnostics=True,
        )
        final = evaluate_factors(
            pixels, true_flat_counts, true_spectra, fitted_counts, fitted_spectra
        )
        mcr_succeeded = True
    except RuntimeError as error:
        # SciPy's NNLS can exhaust its iteration limit for a three-factor
        # update. Do not claim that the selected factors are an optimized fit.
        mcr_error = f"{type(error).__name__}: {error}"
        mcr_succeeded = False
        final = None
        fit_history = np.asarray([initial_fit])
    starting_spectra = initial.estimated_spectra
    displayed_factors = final if final is not None else initial
    fitted_spectra = displayed_factors.estimated_spectra
    fitted_counts = displayed_factors.estimated_abundances.reshape(*IMAGE_SHAPE, 3)

    row = {
        "components": ";".join(COMPONENTS),
        "hidden_component": HIDDEN_COMPONENT,
        "total_counts_per_pixel": TOTAL_COUNTS_PER_PIXEL,
        "image_pixels": int(np.prod(IMAGE_SHAPE)),
        "max_hidden_fraction": max_egfp_fraction,
        "mcr_iterations": MCR_ITERATIONS,
        "mcr_tolerance_percent": MCR_TOL_PERCENT,
        "mcr_completed_iterations": (
            diagnostics.iterations_completed if diagnostics is not None else 0
        ),
        "mcr_best_iteration": (
            diagnostics.best_iteration if diagnostics is not None else ""
        ),
        "mcr_stop_reason": (
            diagnostics.stop_reason if diagnostics is not None else "failed"
        ),
        "mcr_status": "completed" if mcr_succeeded else "failed",
        "mcr_error": mcr_error,
        "hull_candidate_count": len(hull_indices),
        "hull_hidden_fractions": ";".join(
            f"{value:.4f}" for value in egfp_fraction.ravel()[hull_indices]
        ),
        "selected_hidden_fractions": ";".join(
            f"{value:.4f}" for value in egfp_fraction.ravel()[selected_indices]
        ),
        "initial_global_relative_l2": initial.spectral_relative_l2_global,
        "final_global_relative_l2": (
            final.spectral_relative_l2_global if final is not None else float("nan")
        ),
        "initial_max_channel_absolute": initial.max_channel_absolute,
        "final_max_channel_absolute": (
            final.max_channel_absolute if final is not None else float("nan")
        ),
        "max_error_component": (
            COMPONENTS[final.max_error_component_index] if final is not None else ""
        ),
        "max_error_wavelength_nm": (
            float(detector.channel_centers[final.max_error_channel_index])
            if final is not None else float("nan")
        ),
        "fit_relative_l2": displayed_factors.reconstruction_relative_l2,
        "initial_factor_fit_relative_l2": initial_fit,
        "mcr_initial_lack_of_fit": float(fit_history[0]),
        "mcr_final_lack_of_fit": (
            diagnostics.best_relative_l2 if diagnostics is not None else float("nan")
        ),
        "mcr_last_lack_of_fit": float(fit_history[-1]),
    }
    for index, name in enumerate(COMPONENTS):
        row[f"{name}_concentration_rmse_counts"] = float(
            final.abundance_rmse_per_component[index]
            if final is not None else float("nan")
        )
        row[f"initial_{name}_concentration_rmse_counts"] = float(
            initial.abundance_rmse_per_component[index]
        )
    row["hidden_concentration_rmse_fraction_of_total"] = (
        row[f"{HIDDEN_COMPONENT}_concentration_rmse_counts"]
        / TOTAL_COUNTS_PER_PIXEL
    )
    return (
        row, true_counts, fitted_counts, true_spectra, starting_spectra,
        fitted_spectra, fit_history, initial, final,
    )


def _save_case_figure(path, row, true_counts, fitted_counts, true_spectra,
                      starting_spectra, fitted_spectra, wavelengths):
    """Plot spectra and true, recovered, and error maps for all components."""
    errors = concentration_error_maps(
        true_counts, true_spectra, fitted_counts, fitted_spectra
    ).signed
    fig = Figure(figsize=(13, 12), dpi=140)
    axes = fig.subplots(4, 3)
    spectra_to_plot = [
        (true_spectra, "black", "True spectrum", "-"),
        (starting_spectra, "tab:blue", "Convex hull + unique profiles", ":"),
    ]
    if row["mcr_status"] == "completed":
        spectra_to_plot.append(
            (fitted_spectra, "tab:orange",
             f"After MCR-ALS ({row['mcr_best_iteration']} iterations)", "--")
        )
    for component, name in enumerate(COMPONENTS):
        for spectra, color, label, linestyle in spectra_to_plot:
            normalized = normalize_components(spectra, do_norm="L1")
            axes[0, component].plot(
                wavelengths, normalized[component], color=color,
                linestyle=linestyle, linewidth=1.5, label=label,
            )
        axes[0, component].set_title(f"{name} spectrum")
        axes[0, component].set_xlabel("Wavelength (nm)")
        axes[0, component].set_ylabel("Fraction of detector counts")
        axes[0, component].legend(frameon=False, fontsize=8)
        axes[0, component].grid(alpha=0.2)

        true_map = true_counts[..., component]
        fitted_map = fitted_counts[..., component]
        difference = errors[..., component]
        abundance_limit = max(float(true_map.max()), float(fitted_map.max()))
        # Avoid scientific-notation colorbar offsets covering plot titles when
        # the selected start already matches the truth to round-off.
        error_limit = max(float(np.max(np.abs(difference))), 1e-2)
        for ax, data, title in zip(
            axes[component + 1],
            (true_map, fitted_map, difference),
            (f"True {name} counts",
             (f"Recovered {name} counts" if row["mcr_status"] == "completed"
              else f"Starting estimate {name} counts"),
             (f"{name} error: recovered − true" if row["mcr_status"] == "completed"
              else f"{name} error: starting estimate − true")),
        ):
            if "error" in title:
                image = ax.imshow(
                    data, vmin=-error_limit, vmax=error_limit, cmap="coolwarm"
                )
            else:
                image = ax.imshow(data, vmin=0, vmax=abundance_limit, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    error_location = (
        "numerical precision" if row["final_max_channel_absolute"] < 1e-8
        else f"{row['max_error_component']} at {row['max_error_wavelength_nm']:g} nm"
    )
    egfp_rmse = row[f"{HIDDEN_COMPONENT}_concentration_rmse_counts"]
    fig.suptitle(
        f"Maximum EGFP fraction: {row['max_hidden_fraction']:.0%} · noiseless ternary mixture\n"
        + (f"MCR-ALS {row['mcr_stop_reason']} after "
           f"{row['mcr_completed_iterations']} iterations: spectral error (unit L2) "
           f"{row['final_global_relative_l2']:.2%}; largest channel "
           f"{row['final_max_channel_absolute']:.4f} ({error_location}); "
           f"EGFP RMSE: {egfp_rmse:.1f} counts; "
           f"reconstruction relative L2: {row['fit_relative_l2']:.2e}"
           if row["mcr_status"] == "completed" else
           f"MCR-ALS failed ({row['mcr_error']}); showing selected-start fit "
           f"with relative L2 {row['fit_relative_l2']:.2e}"),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.04, top=0.88,
                        hspace=0.52, wspace=0.42)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


def _save_overview(path, rows):
    """Plot spectral, EGFP concentration, and reconstruction errors."""
    levels = np.array([row["max_hidden_fraction"] * 100 for row in rows])
    fig = Figure(figsize=(10, 7), dpi=150)
    axes = fig.subplots(2, 2)
    comparisons = (
        (axes[0, 0], "global_relative_l2", "Global spectral error, unit L2 (%)", 100),
        (axes[0, 1], "max_channel_absolute", "Largest channel error, unit L2", 1),
    )
    for ax, key, label, scale in comparisons:
        ax.plot(levels, [row[f"initial_{key}"] * scale for row in rows],
                "o--", label="Selected candidates")
        ax.plot(levels, [row[f"final_{key}"] * scale for row in rows],
                "o-", label="After MCR-ALS")
        ax.set_ylabel(label)
        ax.legend(frameon=False)

    axes[1, 0].plot(
        levels,
        [row[f"{HIDDEN_COMPONENT}_concentration_rmse_counts"] for row in rows],
        "o-", color="tab:red",
    )
    axes[1, 0].set_ylabel("EGFP concentration RMSE (counts)")
    axes[1, 1].semilogy(
        levels, [max(row["fit_relative_l2"], 1e-16) for row in rows],
        "o-", color="tab:green",
    )
    axes[1, 1].set_ylabel("Reconstruction relative L2 (log scale)")
    for ax in axes.ravel():
        ax.invert_xaxis()
        ax.set_xlabel("Maximum EGFP fraction (%)")
        ax.grid(alpha=0.2)

    fig.suptitle("Recovery as pure EGFP disappears from ternary data")
    fig.text(
        0.5, 0.01,
        "A small reconstruction residual can coexist with inaccurate spectra or concentrations.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.9,
                        hspace=0.38, wspace=0.35)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


def _save_optimization_history(path, histories):
    """Plot relative lack of fit at initialization and after every MCR cycle."""
    fig = Figure(figsize=(9, 5.5), dpi=150)
    ax = fig.subplots()
    for max_fraction, history, status in histories:
        ax.semilogy(
            np.arange(len(history)), np.maximum(history, 1e-16),
            label=f"Maximum EGFP: {max_fraction:.0%} ({status})",
        )
    ax.set_xlabel("MCR-ALS iteration (0 is the selected starting spectra)")
    ax.set_ylabel("Relative L2 lack of fit")
    ax.set_title(f"MCR-ALS fit history (up to {MCR_ITERATIONS} iterations)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.text(
        0.5, 0.01,
        "Values near 10⁻¹⁵ are at floating-point precision; changes there are round-off.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


def main(output_dir=OUTPUT_DIR):
    """Run the noiseless EGFP titration and save tables and figures."""
    if (len(IMAGE_SHAPE) != 2 or min(IMAGE_SHAPE) < 2
            or TOTAL_COUNTS_PER_PIXEL <= 0):
        raise ValueError("Use an image of at least 2 x 2 pixels and positive counts")
    if (not MAX_HIDDEN_FRACTIONS
            or any(not 0 < value <= 1 for value in MAX_HIDDEN_FRACTIONS)
            or len(set(MAX_HIDDEN_FRACTIONS)) != len(MAX_HIDDEN_FRACTIONS)):
        raise ValueError("Titration fractions must be distinct values in (0, 1]")

    detector, true_spectra = _load_spectra()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    component_rows = []
    histories = []
    for index, max_fraction in enumerate(MAX_HIDDEN_FRACTIONS):
        (row, true_counts, fitted_counts, truth, starting, recovered,
         fit_history, initial, final) = _evaluate_level(
            max_fraction, detector, true_spectra
        )
        rows.append(row)
        component_rows.extend(component_metric_rows(
            max_fraction, COMPONENTS, initial, final
        ))
        histories.append((max_fraction, fit_history, row["mcr_stop_reason"]))
        _save_case_figure(
            output_dir / f"case_{index:02d}_{max_fraction:.3f}.png",
            row, true_counts, fitted_counts, truth, starting, recovered,
            detector.channel_centers,
        )

    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6g}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    save_component_metrics(output_dir / "component_metrics.csv", component_rows)

    table = [
        "# Hidden EGFP ternary evaluation", "",
        "Components: EBFP2, EGFP, and mCherry. EBFP2 and mCherry remain pure "
        "along the image's bottom edge. Each pixel has "
        f"{TOTAL_COUNTS_PER_PIXEL:g} expected counts. No noise.", "",
        "| Maximum EGFP fraction | MCR-ALS | Global spectral error | "
        "Largest channel error | EGFP concentration RMSE | Factor reconstruction relative L2 |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        channel_detail = ""
        if np.isfinite(row["final_max_channel_absolute"]) and row["final_max_channel_absolute"] >= 1e-8:
            channel_detail = (
                f" ({row['max_error_component']} at "
                f"{row['max_error_wavelength_nm']:g} nm)"
            )
        if row["mcr_status"] == "completed":
            spectral_error = f"{row['final_global_relative_l2']:.2%}"
            channel_error = f"{row['final_max_channel_absolute']:.4f}{channel_detail}"
            egfp_rmse_text = (
                f"{row[f'{HIDDEN_COMPONENT}_concentration_rmse_counts']:.1f} counts"
            )
        else:
            spectral_error = channel_error = egfp_rmse_text = "—"
        table.append(
            f"| {row['max_hidden_fraction']:.0%} "
            f"| {row['mcr_stop_reason']} ({row['mcr_completed_iterations']}) "
            f"| {spectral_error} "
            f"| {channel_error} "
            f"| {egfp_rmse_text} "
            f"| {row['fit_relative_l2']:.2e} |"
        )
    table.extend([
        "", f"## Components at {MAX_HIDDEN_FRACTIONS[-1]:.0%} maximum EGFP fraction", "",
        "| Component | Start spectral error | Fitted spectral error | "
        "Fitted abundance RMSE | Fitted abundance relative error |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for component_row in component_rows[-len(COMPONENTS):]:
        table.append(
            f"| {component_row['component']} "
            f"| {component_row['initial_spectral_relative_l2']:.2%} "
            f"| {component_row['final_spectral_relative_l2']:.2%} "
            f"| {component_row['final_abundance_rmse_counts']:.1f} counts "
            f"| {component_row['final_abundance_relative_l2']:.2%} |"
        )
    (output_dir / "summary.md").write_text("\n".join(table) + "\n")
    _save_overview(output_dir / "overview.png", rows)
    save_component_recovery_figure(
        output_dir / "component_recovery.png", component_rows,
        COMPONENTS, HIDDEN_COMPONENT,
    )
    _save_optimization_history(output_dir / "optimization_history.png", histories)
    return output_dir


if __name__ == "__main__":
    saved_dir = main()
    print(f"Saved evaluation results to {saved_dir}")
