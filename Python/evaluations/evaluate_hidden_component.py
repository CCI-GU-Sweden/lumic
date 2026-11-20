"""Evaluate data-driven recovery when one binary component is never pure.

Run ``python -m evaluations.evaluate_hidden_component`` from ``Python``.
Only the settings in the block below need changing for a first exploration.
This experiment is noiseless and uses saved FPbase emission curves offline.
"""

import argparse
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
from ._execution import record_evaluation


# Choose two distinct names from EBFP2, EGFP, mCherry, and mNeptune.
# The first protein has a pure pixel in every image. The second is titrated.
PURE_COMPONENT = "EGFP"
HIDDEN_COMPONENT = "mCherry"

# 1.0 is a control with both pure spectra. All other settings have no pure
# pixel of HIDDEN_COMPONENT. The maximum fraction occurs at one image corner.
MAX_HIDDEN_FRACTIONS = (1.0, 0.95, 0.75, 0.50, 0.25)
IMAGE_SHAPE = (20, 20)
TOTAL_COUNTS_PER_PIXEL = 1000.0
MCR_ITERATIONS = 500
MCR_TOL_PERCENT = 0.1
UNIQUE_PROFILES_F_PERCENT = 0.5

PYTHON_DIR = Path(__file__).resolve().parents[1]
SPECTRA_DIR = PYTHON_DIR / "lumic" / "simulation" / "fpbase_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "binary_hidden_component"
AVAILABLE_COMPONENTS = ("EBFP2", "EGFP", "mCherry", "mNeptune")


def _load_spectra():
    """Read raw emission curves, then bin and sum-normalize for count mixing."""
    names = (PURE_COMPONENT, HIDDEN_COMPONENT)
    if names[0] == names[1] or any(name not in AVAILABLE_COMPONENTS for name in names):
        raise ValueError(f"Choose two distinct names from {AVAILABLE_COMPONENTS}")

    detector = SpectralDetector.from_profile("980_32ch")
    sampled = []
    for name in names:
        curve = np.loadtxt(
            SPECTRA_DIR / f"{name.lower()}_emission.csv",
            delimiter=",", skiprows=1,
        )
        sampled.append(detector.sample(curve[:, 0], curve[:, 1]))
    sampled = np.asarray(sampled)
    if np.any(sampled.sum(axis=1) <= 0):
        raise ValueError("Each chosen spectrum must have signal in the detector range")
    return detector, sampled / sampled.sum(axis=1, keepdims=True)


def _make_component_counts(max_hidden_fraction):
    """Make a diagonal titration: pure first component to the hidden limit."""
    rows, columns = IMAGE_SHAPE
    y, x = np.indices(IMAGE_SHAPE, dtype=float)
    hidden_fraction = max_hidden_fraction * (
        x / (columns - 1) + y / (rows - 1)
    ) / 2
    fractions = np.stack((1 - hidden_fraction, hidden_fraction), axis=-1)
    return fractions * TOTAL_COUNTS_PER_PIXEL, hidden_fraction


def _evaluate_level(max_hidden_fraction, detector, true_spectra):
    """Run the full selection and MCR-ALS workflow without giving it truth."""
    true_counts, hidden_fraction = _make_component_counts(max_hidden_fraction)
    dataset = simulate_linear_mixture(
        true_counts,
        true_spectra,
        detector.channel_centers,
        fluorophore_names=(PURE_COMPONENT, HIDDEN_COMPONENT),
    )
    pixels = dataset.image.reshape(-1, detector.n_channels)

    # Candidate selection sees only the spectral image. The known fractions
    # are used below solely to describe which pixels it selected.
    candidates, hull_indices = essential_spectral_pixles(pixels, npc=2)
    starting_spectra, selected_in_hull = unique_profiles(
        candidates.T, nr=2, f=UNIQUE_PROFILES_F_PERCENT
    )
    selected_indices = hull_indices[selected_in_hull]

    starting_spectra = normalize_components(starting_spectra, do_norm="L1")
    starting_counts = positive_least_squares(
        pixels, starting_spectra, do_norm=False
    )
    true_flat_counts = true_counts.reshape(-1, 2)
    initial = evaluate_factors(
        pixels, true_flat_counts, true_spectra,
        starting_counts, starting_spectra,
    )
    fitted_counts, fitted_spectra, fit_history, diagnostics = mcr_als(
        pixels, starting_spectra, nIter=MCR_ITERATIONS, normL="L1",
        tol_percent=MCR_TOL_PERCENT, return_diagnostics=True,
    )
    final = evaluate_factors(
        pixels, true_flat_counts, true_spectra, fitted_counts, fitted_spectra
    )
    starting_spectra = initial.estimated_spectra
    fitted_spectra = final.estimated_spectra
    fitted_counts = final.estimated_abundances.reshape(*IMAGE_SHAPE, 2)

    row = {
        "pure_component": PURE_COMPONENT,
        "hidden_component": HIDDEN_COMPONENT,
        "total_counts_per_pixel": TOTAL_COUNTS_PER_PIXEL,
        "image_pixels": int(np.prod(IMAGE_SHAPE)),
        "max_hidden_fraction": max_hidden_fraction,
        "mcr_iterations": MCR_ITERATIONS,
        "mcr_tolerance_percent": MCR_TOL_PERCENT,
        "mcr_completed_iterations": diagnostics.iterations_completed,
        "mcr_best_iteration": diagnostics.best_iteration,
        "mcr_stop_reason": diagnostics.stop_reason,
        "hull_candidate_count": len(hull_indices),
        "hull_hidden_fractions": ";".join(
            f"{value:.4f}" for value in hidden_fraction.ravel()[hull_indices]
        ),
        "selected_hidden_fractions": ";".join(
            f"{value:.4f}" for value in hidden_fraction.ravel()[selected_indices]
        ),
        "initial_global_relative_l2": initial.spectral_relative_l2_global,
        "final_global_relative_l2": final.spectral_relative_l2_global,
        "initial_max_channel_absolute": initial.max_channel_absolute,
        "final_max_channel_absolute": final.max_channel_absolute,
        "max_error_component": (PURE_COMPONENT, HIDDEN_COMPONENT)[
            final.max_error_component_index
        ],
        "max_error_wavelength_nm": float(
            detector.channel_centers[final.max_error_channel_index]
        ),
        "pure_concentration_rmse_counts": float(final.abundance_rmse_per_component[0]),
        "hidden_concentration_rmse_counts": float(final.abundance_rmse_per_component[1]),
        "hidden_concentration_rmse_fraction_of_total": float(
            final.abundance_rmse_per_component[1] / TOTAL_COUNTS_PER_PIXEL
        ),
        "fit_relative_l2": final.reconstruction_relative_l2,
        "mcr_initial_lack_of_fit": float(fit_history[0]),
        "mcr_final_lack_of_fit": diagnostics.best_relative_l2,
        "mcr_last_lack_of_fit": float(fit_history[-1]),
    }
    return (
        row, true_counts, fitted_counts, true_spectra, starting_spectra,
        fitted_spectra, fit_history, initial, final,
    )


def _save_case_figure(path, row, true_counts, fitted_counts, true_spectra,
                      starting_spectra, fitted_spectra, wavelengths):
    """Show both recovered signatures and both concentration error maps."""
    errors = concentration_error_maps(
        true_counts, true_spectra, fitted_counts, fitted_spectra
    ).signed
    fig = Figure(figsize=(12, 9), dpi=140)
    axes = fig.subplots(3, 3)
    names = (PURE_COMPONENT, HIDDEN_COMPONENT)
    true_plot_spectra = normalize_components(true_spectra, do_norm="L1")
    starting_plot_spectra = normalize_components(starting_spectra, do_norm="L1")
    fitted_plot_spectra = normalize_components(fitted_spectra, do_norm="L1")
    for component, name in enumerate(names):
        ax = axes[0, component]
        ax.plot(
            wavelengths, true_plot_spectra[component], color="black",
            linewidth=1.6, label="True spectrum",
        )
        ax.plot(
            wavelengths, starting_plot_spectra[component], color="tab:blue",
            linestyle=":", linewidth=1.6, label="Convex hull + unique profiles",
        )
        ax.plot(
            wavelengths, fitted_plot_spectra[component], color="tab:orange",
            linestyle="--", linewidth=1.3,
            label=f"After MCR-ALS ({row['mcr_best_iteration']} iterations)",
        )
        ax.set_title(f"{name} spectrum")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction of detector counts")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

    info = axes[0, 2]
    info.axis("off")
    if row["final_max_channel_absolute"] < 1e-8:
        largest_error_location = "numerical precision"
    else:
        largest_error_location = (
            f"{row['max_error_component']} at {row['max_error_wavelength_nm']:g} nm"
        )
    info.text(
        0, 1,
        f"Global spectral error (unit L2): {row['final_global_relative_l2']:.3%}\n"
        f"Largest channel error (unit L2): {row['final_max_channel_absolute']:.4f}\n"
        f"  {largest_error_location}\n"
        f"Hidden concentration RMSE: {row['hidden_concentration_rmse_counts']:.1f} counts\n"
        f"Reconstruction relative L2: {row['fit_relative_l2']:.2e}",
        ha="left", va="top", fontsize=10,
    )

    for component, name in enumerate(names):
        row_axes = axes[component + 1]
        true_map = true_counts[..., component]
        fitted_map = fitted_counts[..., component]
        difference = errors[..., component]
        abundance_limit = max(float(true_map.max()), float(fitted_map.max()))
        error_limit = max(float(np.max(np.abs(difference))), 1e-12)

        for ax, data, title in zip(
            row_axes[:2], (true_map, fitted_map),
            (f"True {name} counts", f"Recovered {name} counts"),
        ):
            image = ax.imshow(data, vmin=0, vmax=abundance_limit, cmap="viridis")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        error_image = row_axes[2].imshow(
            difference, vmin=-error_limit, vmax=error_limit, cmap="coolwarm"
        )
        row_axes[2].set_title(f"{name} error: recovered − true")
        fig.colorbar(error_image, ax=row_axes[2], fraction=0.046, pad=0.04)
        for ax in row_axes:
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")

    fig.suptitle(
        f"Maximum {HIDDEN_COMPONENT} fraction: {row['max_hidden_fraction']:.0%}"
        " · noiseless binary mixture"
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.05, top=0.92,
                        hspace=0.48, wspace=0.42)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


def _save_overview(path, rows):
    """Summarize how spectral and concentration errors change with titration."""
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
        [row["hidden_concentration_rmse_counts"] for row in rows],
        "o-", color="tab:red",
    )
    axes[1, 0].set_ylabel(f"{HIDDEN_COMPONENT} concentration RMSE (counts)")
    axes[1, 1].semilogy(
        levels, [max(row["fit_relative_l2"], 1e-16) for row in rows],
        "o-", color="tab:green",
    )
    axes[1, 1].set_ylabel("Reconstruction relative L2 (log scale)")
    for ax in axes.ravel():
        ax.invert_xaxis()
        ax.set_xlabel(f"Maximum {HIDDEN_COMPONENT} fraction (%)")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Recovery as pure {HIDDEN_COMPONENT} disappears from the data")
    fig.text(
        0.5, 0.01,
        "A small reconstruction residual can coexist with an inaccurate recovered spectrum.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.9,
                        hspace=0.38, wspace=0.35)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


def _save_optimization_history(path, histories):
    """Plot relative lack of fit at initialization and after each MCR-ALS cycle."""
    fig = Figure(figsize=(9, 5.5), dpi=150)
    ax = fig.subplots()
    for max_hidden_fraction, history, reason in histories:
        iterations = np.arange(len(history))
        ax.semilogy(
            iterations, np.maximum(history, 1e-16),
            label=f"Maximum {HIDDEN_COMPONENT}: {max_hidden_fraction:.0%} ({reason})",
        )
    ax.set_xlabel("MCR-ALS iteration (0 is the selected starting spectra)")
    ax.set_ylabel("Relative L2 lack of fit")
    ax.set_title(f"MCR-ALS fit history (up to {MCR_ITERATIONS} iterations)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.text(
        0.5, 0.01,
        "Values near 10⁻¹⁵ are at floating-point precision; changes there are numerical round-off.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(path, facecolor="white", bbox_inches="tight")


@record_evaluation("output_dir")
def main(output_dir=OUTPUT_DIR):
    """Run the noiseless titration and write a CSV plus explanatory figures."""
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
    for index, max_hidden_fraction in enumerate(MAX_HIDDEN_FRACTIONS):
        (row, true_counts, fitted_counts, truth, starting, recovered,
         fit_history, initial, final) = _evaluate_level(
            max_hidden_fraction, detector, true_spectra
        )
        rows.append(row)
        component_rows.extend(component_metric_rows(
            max_hidden_fraction, (PURE_COMPONENT, HIDDEN_COMPONENT), initial, final
        ))
        histories.append((max_hidden_fraction, fit_history, row["mcr_stop_reason"]))
        case_path = output_dir / f"case_{index:02d}_{max_hidden_fraction:.3f}.png"
        _save_case_figure(
            case_path, row, true_counts, fitted_counts, truth, starting,
            recovered, detector.channel_centers,
        )

    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            # Keep the CSV readable while retaining enough precision for
            # comparing small residuals with substantial recovery errors.
            writer.writerow({
                key: f"{value:.6g}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    save_component_metrics(output_dir / "component_metrics.csv", component_rows)
    table = [
        f"# Hidden {HIDDEN_COMPONENT} evaluation",
        "",
        f"Pure component: {PURE_COMPONENT}. Total: {TOTAL_COUNTS_PER_PIXEL:g} "
        "expected counts per pixel. No noise.",
        "",
        "| Maximum hidden fraction | MCR-ALS stop | Global spectral error | Largest channel "
        "error | Hidden concentration RMSE | Reconstruction relative L2 |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["final_max_channel_absolute"] < 1e-8:
            channel_detail = ""
        else:
            channel_detail = (
                f" ({row['max_error_component']} at "
                f"{row['max_error_wavelength_nm']:g} nm)"
            )
        table.append(
            f"| {row['max_hidden_fraction']:.0%} "
            f"| {row['mcr_stop_reason']} ({row['mcr_completed_iterations']}) "
            f"| {row['final_global_relative_l2']:.2%} "
            f"| {row['final_max_channel_absolute']:.4f}{channel_detail} "
            f"| {row['hidden_concentration_rmse_counts']:.1f} counts "
            f"| {row['fit_relative_l2']:.2e} |"
        )
    table.extend([
        "", f"## Components at {MAX_HIDDEN_FRACTIONS[-1]:.0%} maximum hidden fraction", "",
        "| Component | Start spectral error | Fitted spectral error | "
        "Fitted abundance RMSE | Fitted abundance relative error |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for component_row in component_rows[-2:]:
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
        (PURE_COMPONENT, HIDDEN_COMPONENT), HIDDEN_COMPONENT,
    )
    _save_optimization_history(
        output_dir / "optimization_history.png", histories
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    saved_dir = main(parser.parse_args().output_dir)
    print(f"Saved evaluation results to {saved_dir}")
