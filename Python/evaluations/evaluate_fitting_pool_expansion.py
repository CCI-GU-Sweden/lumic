"""Noiseless test of adding image coverage to a complete convex-hull pool.

Run from Python: python -m evaluations.evaluate_fitting_pool_expansion
Selection and ALS never use simulation truth. Truth enters reporting only.
"""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from scipy.spatial import QhullError

from lumic import mcr_als, positive_least_squares
from lumic.evaluation import concentration_error_maps, reconstruction_error_maps
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene
from tqdm.auto import tqdm

from . import evaluate_background_recovery as diagnostic
from . import evaluate_radial_spoke_blur as scene
from ._execution import record_evaluation


# User settings: added counts are non-hull pixels, not total pool sizes.
BLUR_FWHM_UM = (0.0, 0.65)
ADDED_PIXEL_BUDGETS = (64, 256, 512)
SAMPLING_SEEDS = (2026, 2027, 2028)
MAX_ITERATIONS = 1000
TOL_PERCENT = 0.1
SOLVER = "auto"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "fitting_pool_expansion"


def expanded_pools(n_pixels, hull_indices, budgets, seed):
    """Retain all hull pixels and add nested, seeded samples without replacement.

    Inputs are flattened pixel indices; neither spectra nor truth are required.
    Budgets count additional distinct non-hull pixels, capped by availability.
    """
    hull = np.asarray(hull_indices)
    if (not isinstance(n_pixels, (int, np.integer)) or n_pixels <= 0
            or hull.ndim != 1 or not hull.size or hull.dtype.kind not in "iu"
            or np.any(hull < 0) or np.any(hull >= n_pixels)):
        raise ValueError("hull must contain valid integer indices in a nonempty image")
    if (any(not isinstance(n, (int, np.integer)) or n < 0 for n in budgets)
            or list(budgets) != sorted(set(budgets))):
        raise ValueError("budgets must be increasing distinct nonnegative integers")
    hull = np.unique(hull)
    remaining = np.setdiff1d(np.arange(n_pixels), hull)
    shuffled = np.random.default_rng(seed).permutation(remaining)
    return {int(n): np.union1d(hull, shuffled[:n]) for n in budgets}


def _write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def overview(path, rows):
    """Per-component recovery across budgets; median and seed range, not CI."""
    fig = Figure(figsize=(16, 10), dpi=130)
    axes = fig.subplots(4, 4)
    for blur_index, blur in enumerate(BLUR_FWHM_UM):
        for component_index, name in enumerate(scene.COMPONENTS):
            for metric_index, (metric, label) in enumerate(
                    (("spectral_error", "Spectral shape error (unit L2)"),
                     ("abundance_rmse", "Concentration RMSE (counts)"))):
                ax = axes[2 * blur_index + metric_index, component_index]
                entries = [r for r in rows if r["blur_fwhm_um"] == blur
                           and r["component"] == name]
                expanded = [r for r in entries if r["pool"] == "hull_plus_uniform"
                            and r["status"] != "failed"]
                medians, lows, highs = [], [], []
                for budget in ADDED_PIXEL_BUDGETS:
                    values = [r[metric] for r in expanded if r["added_budget"] == budget]
                    medians.append(float(np.median(values)) if values else np.nan)
                    lows.append(min(values) if values else np.nan)
                    highs.append(max(values) if values else np.nan)
                ax.plot(ADDED_PIXEL_BUDGETS, medians, "o-", label="Hull + uniform: median")
                ax.fill_between(ADDED_PIXEL_BUDGETS, lows, highs, alpha=0.2,
                                label="Sampling-seed range")
                for pool, style, label_text in (("hull", "--", "Hull only"),
                                                ("full_image", ":", "Full image")):
                    entry = next(r for r in entries if r["pool"] == pool)
                    if entry["status"] != "failed":
                        ax.axhline(entry[metric], ls=style, label=label_text)
                display_name = "Broad background" if name == scene.BACKGROUND_NAME else name
                ax.set_title(f"{display_name} · blur {blur:g} µm")
                ax.set_xlabel("Additional non-hull pixels")
                ax.set_ylabel(label)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.2)
                if component_index == 0:
                    ax.legend(fontsize=7)
    fig.suptitle(f"Pool expansion · identical starts · {MAX_ITERATIONS}-update cap · noiseless data")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)


def comparison_figure(path, blur, pixels, truth, spectra, starts, results):
    """Inspect one declared seed without choosing its results using truth."""
    keys = ["hull", *(f"plus_{n}" for n in ADDED_PIXEL_BUDGETS), "full_image"]
    fig = Figure(figsize=(19, 12), dpi=120)
    axes = fig.subplots(4, len(keys))
    maps, residuals = {}, {}
    for key in keys:
        if key not in results:
            continue
        factors = results[key]
        maps[key] = concentration_error_maps(truth, spectra,
                    factors.estimated_abundances, factors.estimated_spectra).signed[:, -1]
        residuals[key] = reconstruction_error_maps(pixels,
                    factors.estimated_abundances @ factors.estimated_spectra).rms
    error_limit = max([1.] + [float(np.abs(v).max()) for v in maps.values()])
    rms_limit = max([1e-3] + [float(v.max()) for v in residuals.values()])
    wavelengths = SpectralDetector.from_profile("980_32ch").channel_centers
    for col, key in enumerate(keys):
        title = {"hull": "Hull only", "full_image": "Full image"}.get(
            key, key.replace("plus_", "Hull + "))
        if key not in results:
            for ax in axes[:, col]:
                ax.text(.5, .5, "Run failed", ha="center", transform=ax.transAxes)
                ax.set_axis_off()
            axes[0, col].set_title(title)
            continue
        factors = results[key]
        ax = axes[0, col]
        ax.plot(wavelengths, spectra[-1], "k", label="True")
        ax.plot(wavelengths, starts.estimated_spectra[-1], ":", label="Start")
        ax.plot(wavelengths, factors.estimated_spectra[-1], "--", label="Returned")
        ax.set_title(title + "\nBackground spectrum")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction per channel")
        ax.legend(fontsize=7)
        im = axes[1, col].imshow(maps[key].reshape(scene.IMAGE_SHAPE),
                   cmap="coolwarm", vmin=-error_limit, vmax=error_limit)
        axes[1, col].set_title("Background error (counts)\nRecovered − true; truth required")
        fig.colorbar(im, ax=axes[1, col], fraction=.046)
        im = axes[2, col].imshow(np.maximum(residuals[key], 1e-10).reshape(scene.IMAGE_SHAPE),
                   cmap="magma", norm=LogNorm(vmin=1e-10, vmax=rms_limit))
        axes[2, col].set_title("Residual RMS/channel (counts)\nTruth not required")
        fig.colorbar(im, ax=axes[2, col], fraction=.046)
        axes[3, col].bar(np.arange(4), factors.abundance_rmse_per_component)
        axes[3, col].set_xticks(np.arange(4))
        axes[3, col].set_xticklabels(("BFP", "GFP", "Cherry", "BG"))
        axes[3, col].set_ylabel("Concentration RMSE (counts)")
        for row in (1, 2):
            axes[row, col].axis("off")
    fig.suptitle(f"Fitting-pool coverage · blur {blur:g} µm · sample seed {SAMPLING_SEEDS[0]}")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path)


def recovery_summary(rows):
    """Report comparisons without equating a better fit with correct factors."""
    lines = ["", "## Concentration recovery at the largest added budget", "",
             "Expanded-pool entries are medians across sampling seeds. RMSE is in counts; "
             "the background truth is 24 counts per pixel. These are results under the "
             "prototype iteration cap, not necessarily converged solutions.", "",
             "| Blur µm | Component | Hull RMSE | Expanded median RMSE | Full-image RMSE |",
             "| --- | --- | --- | --- | --- |"]
    for blur in BLUR_FWHM_UM:
        for name in scene.COMPONENTS:
            entries = [r for r in rows if r["blur_fwhm_um"] == blur and r["component"] == name]
            hull = next(r for r in entries if r["pool"] == "hull")
            full = next(r for r in entries if r["pool"] == "full_image")
            expanded = [r["abundance_rmse"] for r in entries
                        if r["pool"] == "hull_plus_uniform"
                        and r["added_budget"] == max(ADDED_PIXEL_BUDGETS)
                        and r["status"] != "failed"]
            median = float(np.median(expanded)) if expanded else np.nan
            lines.append(f"| {blur:g} | {name} | {hull['abundance_rmse']:.4g} | "
                         f"{median:.4g} | {full['abundance_rmse']:.4g} |")
    lines.extend(["", "Changing the pool changes which image profiles influence the shared spectra. "
                  "Interpret the coverage comparison jointly with stopping diagnostics and residuals. "
                  "A causal attribution to background representation alone would require additional "
                  "controlled comparisons; adding pixels changes several aspects of the dataset."])
    return lines


@record_evaluation("output_dir")
def main(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = scene.detector_spectra(detector)
    grid = SamplingGrid(scene.IMAGE_SHAPE, pixel_size_um=scene.PIXEL_SIZE_UM)
    rows, regions, selections, initial_rows = [], [], [], []
    saved_factors = {}
    settings = {"blur_fwhm_um": BLUR_FWHM_UM, "added_pixel_budgets": ADDED_PIXEL_BUDGETS,
                "sampling_seeds": SAMPLING_SEEDS, "max_iterations": MAX_ITERATIONS,
                "tol_percent": TOL_PERCENT, "solver": SOLVER,
                "image_shape": scene.IMAGE_SHAPE, "pixel_size_um": scene.PIXEL_SIZE_UM,
                "spoke_counts": scene.SPOKE_COUNTS, "background_counts": scene.BACKGROUND_COUNTS,
                "spoke_seed": scene.SPOKE_SEED, "n_spokes": scene.N_SPOKES,
                "spoke_length_um": scene.SPOKE_LENGTH_UM, "spoke_width_um": scene.SPOKE_WIDTH_UM,
                "background_center_nm": scene.BACKGROUND_CENTER_NM,
                "background_sigma_nm": scene.BACKGROUND_SIGMA_NM, "oversample": 3,
                "components": scene.COMPONENTS, "channel_centers_nm": detector.channel_centers.tolist(),
                "true_unit_sum_spectra": spectra.tolist(), "noise": "disabled",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    for blur in BLUR_FWHM_UM:
        psf = None if blur == 0 else GaussianPSF({name: blur for name in scene.COMPONENTS})
        spatial = render_scene(scene.build_scene(), grid, psf, oversample=3)
        truth = spatial.blurred_counts.reshape(-1, len(scene.COMPONENTS))
        pixels = truth @ spectra
        foreground = truth[:, :3].sum(axis=1) >= scene.BACKGROUND_COUNTS
        starts, hull, anchors = diagnostic.select_starts(pixels)
        initial, _ = diagnostic.evaluate_run(pixels, truth, spectra, starts, foreground)
        for k, name in enumerate(scene.COMPONENTS):
            initial_rows.append({"blur_fwhm_um": blur, "component": name,
                "spectral_error": initial.spectral_relative_l2_per_component[k],
                "abundance_rmse": initial.abundance_rmse_per_component[k],
                "reconstruction_relative_l2": initial.reconstruction_relative_l2})
        pools = [("hull", -1, 0, np.unique(hull)),
                 ("full_image", -1, 0, np.arange(len(pixels)))]
        for seed in SAMPLING_SEEDS:
            pools.extend(("hull_plus_uniform", seed, budget, indices)
                         for budget, indices in expanded_pools(
                             len(pixels), hull, ADDED_PIXEL_BUDGETS, seed).items())
        previews = {}
        for pool, seed, budget, indices in pools:
            base = {"blur_fwhm_um": blur, "pool": pool, "sampling_seed": seed,
                    "added_budget": budget, "actual_added_pixels": len(indices)-len(np.unique(hull)),
                    "fit_pixels": len(indices), "hull_pixels": len(np.unique(hull)),
                    "foreground_fit_pixels": int(foreground[indices].sum()),
                    "background_region_fit_pixels": int((~foreground[indices]).sum())}
            selections.append({**base, "fit_indices": indices.tolist(),
                               "initial_spectra": starts.tolist(),
                               "selected_start_indices": anchors.tolist()})
            label = f"blur {blur:g} / {pool} / +{budget} / seed {seed}"
            began = perf_counter()
            try:
                with tqdm(total=MAX_ITERATIONS, desc=label, unit="iter", leave=False) as progress:
                    _, fitted, history, diagnostics = mcr_als(
                        pixels[indices], starts.copy(), nIter=MAX_ITERATIONS, normL="L1",
                        tol_percent=TOL_PERCENT, solver=SOLVER, return_diagnostics=True,
                        iteration_callback=lambda _: progress.update(1))
                als_seconds = perf_counter()-began
                metrics, regional = diagnostic.evaluate_run(pixels, truth, spectra, fitted, foreground)
                common = {**base, "status": diagnostics.stop_reason, "failure": "",
                          "iterations": diagnostics.iterations_completed,
                          "best_iteration": diagnostics.best_iteration,
                          "als_seconds": als_seconds, "total_seconds": perf_counter()-began}
                if pool != "hull_plus_uniform" or seed == SAMPLING_SEEDS[0]:
                    previews[f"plus_{budget}" if pool == "hull_plus_uniform" else pool] = metrics
                for k, name in enumerate(scene.COMPONENTS):
                    row = {**common, "component": name,
                           "spectral_error": metrics.spectral_relative_l2_per_component[k],
                           "max_channel_error": float(np.max(np.abs(metrics.estimated_spectra[k]-spectra[k]))),
                           "abundance_rmse": metrics.abundance_rmse_per_component[k],
                           "abundance_relative_l2": metrics.abundance_relative_l2_per_component[k],
                           "reconstruction_relative_l2": metrics.reconstruction_relative_l2}
                    rows.append(row)
                    for region in ("foreground", "background_region"):
                        regions.append({**row, "region": region, **regional[(region, name)]})
                factor_key = f"blur_{blur:g}_{pool}_seed_{seed}_budget_{budget}"
                saved_factors[factor_key + "_spectra"] = metrics.estimated_spectra
                saved_factors[factor_key + "_abundances"] = metrics.estimated_abundances
                selections[-1]["factor_key"] = factor_key
                selections[-1]["fit_history"] = history.tolist()
            except (RuntimeError, ValueError, np.linalg.LinAlgError, QhullError) as error:
                print(f"{label}: failed: {error}")
                for name in scene.COMPONENTS:
                    rows.append({**base, "status": "failed", "failure": str(error),
                        "iterations": "", "best_iteration": "", "als_seconds": np.nan,
                        "total_seconds": perf_counter()-began, "component": name,
                        **dict.fromkeys(("spectral_error", "max_channel_error", "abundance_rmse",
                                        "abundance_relative_l2", "reconstruction_relative_l2"), np.nan)})
            # Save numerical evidence incrementally in case a later run is interrupted.
            _write_csv(output_dir / "summary.csv", rows)
        comparison_figure(output_dir / f"comparison_{blur:.2f}um.png", blur,
                          pixels, truth, spectra, initial, previews)
    _write_csv(output_dir / "initial_metrics.csv", initial_rows)
    if regions:
        _write_csv(output_dir / "regional_errors.csv", regions)
    (output_dir / "settings.json").write_text(json.dumps({**settings, "fit_selection": selections}, indent=2)+"\n")
    np.savez_compressed(output_dir / "fitted_factors.npz", **saved_factors)
    overview(output_dir / "overview.png", rows)
    report = ["# Convex-hull fitting-pool expansion", "",
        "Noiseless scene; EBFP2, EGFP, mCherry and uniform broad background. "
        "Identical blind starts and solver settings within each blur condition. "
        "All pools preserve every hull pixel. Added uniform samples are nested within each seed.", "",
        "Ground truth is used only for evaluation and regional labels. Background-region pixels "
        "have total fluorophore counts below 24 and may include faint blur tails.", "",
        f"{MAX_ITERATIONS} updates maximum, L1 normalization, {TOL_PERCENT}% tolerance, solver {SOLVER}. "
        "Full-image fitting is a reference with the same cap, not a guaranteed optimum. "
        "Max-iteration results remain unconverged; returned factors come from the best iteration.", "",
        "[Overview](overview.png) · [Unblurred maps](comparison_0.00um.png) · "
        "[Blurred maps](comparison_0.65um.png)", "",
        "Curves show median and range across three sampling seeds, not confidence intervals. "
        "Maps show the preselected first seed, not the best outcome. Runtime is a single "
        "observed timing per run, not a controlled performance benchmark.", "",
        "| Blur µm | Pool | Added budget | Seed | Fit pixels | BG region pixels | BG shape error | BG count RMSE | Reconstruction L2 | Iterations/status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["component"] == scene.BACKGROUND_NAME:
            report.append(f"| {r['blur_fwhm_um']:g} | {r['pool']} | {r['added_budget']} | "
                f"{r['sampling_seed']} | {r['fit_pixels']} | {r['background_region_fit_pixels']} | "
                f"{r['spectral_error']:.4g} | {r['abundance_rmse']:.4g} | "
                f"{r['reconstruction_relative_l2']:.4g} | {r['iterations']} / {r['status']} |")
    report.extend(["", "## Interpretation", "",
        "More pixels change the influence of common image profiles on ALS. Assess each component "
        "and regional errors alongside reconstruction; low residuals can coexist with incorrect factors. "
        "This experiment does not establish that random coverage resolves ambiguity, or that a "
        "larger fitting pool is universally better. Weighting, adaptive residual selection, "
        "superpixels, constraints, and noise scaling are not implemented here.", "",
        "Exact pixel indices, starting spectra, fit histories, and scene settings are in settings.json. "
        "Initial metrics are separate from fitted metrics. See summary.csv for all components and "
        "regional_errors.csv for foreground/background errors; failed runs remain in summary.csv."])
    report.extend(recovery_summary(rows))
    (output_dir / "summary.md").write_text("\n".join(report)+"\n")
    print(f"Saved fitting-pool evaluation to {output_dir}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    main(parser.parse_args().output_dir)
