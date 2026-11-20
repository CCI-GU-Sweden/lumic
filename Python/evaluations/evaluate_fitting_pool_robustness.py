"""Repeated subsampling of non-hull pixels, with fixed hull and blind starts.

Build the fitting-pool expansion first, reuse its first three +512 runs,
then fit seven additional seeds. This measures sampling sensitivity, not
bootstrap uncertainty, measurement noise, or accuracy on real images.
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

from lumic import mcr_als
from lumic.evaluation import evaluate_factors
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene
from . import evaluate_fitting_pool_expansion as expansion
from . import evaluate_background_recovery as diagnostic
from . import evaluate_radial_spoke_blur as scene
from ._execution import record_evaluation, prepare_prerequisites


SAMPLING_SEEDS = tuple(range(2026, 2036))
ADDED_BUDGET = 512
OUTPUT_DIR = expansion.OUTPUT_DIR.parent / "fitting_pool_robustness"


def compare_iteration_caps(base_dir):
    """Compare identical starts and pixel pools, differing only in iteration cap."""
    base_dir = Path(base_dir)
    old_dir = base_dir / "iterations_500"
    old = json.loads((old_dir / "settings.json").read_text())
    new = json.loads((base_dir / "settings.json").read_text())
    # Scene settings, solver and selection configuration must all agree.
    for key in old:
        if key not in ("max_iterations", "fit_selection"):
            if old[key] != new[key]:
                raise ValueError(f"iteration comparison changed {key}")
    for a, b in zip(old["fit_selection"], new["fit_selection"]):
        for key in ("blur_fwhm_um", "pool", "sampling_seed", "added_budget",
                    "fit_indices", "initial_spectra", "selected_start_indices"):
            if a[key] != b[key]:
                raise ValueError(f"iteration comparison changed selection {key}")
    if len(old["fit_selection"]) != len(new["fit_selection"]):
        raise ValueError("iteration comparison changed the number of pools")
    def identity(row):
        return tuple(row[k] for k in ("blur_fwhm_um", "pool", "sampling_seed",
                                     "added_budget", "component"))
    with (old_dir / "summary.csv").open() as file:
        previous = {identity(r): r for r in csv.DictReader(file)}
    with (base_dir / "summary.csv").open() as file:
        current = list(csv.DictReader(file))
    if set(previous) != {identity(r) for r in current}:
        raise ValueError("iteration comparison requires identical metric rows")
    rows = []
    for r in current:
        a = previous[identity(r)]
        row = {k: r[k] for k in ("blur_fwhm_um", "pool", "sampling_seed", "added_budget", "component")}
        for metric in ("spectral_error", "abundance_rmse", "reconstruction_relative_l2",
                       "iterations", "best_iteration", "status"):
            row[f"cap_{old['max_iterations']}_{metric}"] = a[metric]
            row[f"cap_{new['max_iterations']}_{metric}"] = r[metric]
        rows.append(row)
    expansion._write_csv(base_dir / "iteration_comparison.csv", rows)
    fig = Figure(figsize=(14, 8), dpi=130)
    axes = fig.subplots(2, 3)
    labels = ["Hull", *(f"+{n}" for n in expansion.ADDED_PIXEL_BUDGETS), "Full"]
    report = ["# Iteration-cap comparison", "",
        "Scene settings, exact fitting indices and starting spectra were checked for agreement. "
        "Both caps restart the same starts; these are best-returned factors, not continuation runs. "
        "Early stopping remains enabled. The figure shows the preselected first sampling seed.", "",
        "[Comparison figure](iteration_comparison.png)", "",
        "| Blur µm | Pool | Added | Seed | BG RMSE, 500 | BG RMSE, 1000 | Residual, 500 | Residual, 1000 | Stop, 1000 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for blur_index, blur in enumerate(expansion.BLUR_FWHM_UM):
        entries = [r for r in rows if float(r["blur_fwhm_um"]) == blur
                   and r["component"] == scene.BACKGROUND_NAME]
        selected = [next(r for r in entries if r["pool"] == "hull")]
        selected.extend(next(r for r in entries if r["pool"] == "hull_plus_uniform"
                        and int(r["sampling_seed"]) == expansion.SAMPLING_SEEDS[0]
                        and int(r["added_budget"]) == n) for n in expansion.ADDED_PIXEL_BUDGETS)
        selected.append(next(r for r in entries if r["pool"] == "full_image"))
        for column, (metric, title) in enumerate((("spectral_error", "Background spectral shape error"),
                ("abundance_rmse", "Background concentration RMSE (counts)"),
                ("reconstruction_relative_l2", "Full-image relative residual"))):
            ax = axes[blur_index, column]
            for cap in (old["max_iterations"], new["max_iterations"]):
                values = [float(r[f"cap_{cap}_{metric}"]) for r in selected]
                if column == 2:
                    values = np.maximum(values, 1e-16)
                    ax.set_yscale("log")
                ax.plot(np.arange(len(selected)), values, "o-", label=f"Cap {cap}")
            ax.set_xticks(np.arange(len(selected)))
            ax.set_xticklabels(labels)
            ax.set_title(f"{title}\nBlur {blur:g} µm")
            ax.grid(alpha=.2)
            ax.legend()
        for r in entries:
            report.append(f"| {blur:g} | {r['pool']} | {r['added_budget']} | {r['sampling_seed']} | "
                f"{float(r['cap_500_abundance_rmse']):.4g} | {float(r['cap_1000_abundance_rmse']):.4g} | "
                f"{float(r['cap_500_reconstruction_relative_l2']):.4g} | "
                f"{float(r['cap_1000_reconstruction_relative_l2']):.4g} | {r['cap_1000_status']} |")
    fig.suptitle("Same pools and starts · 500 versus 1000 iterations · first seed")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(base_dir / "iteration_comparison.png")
    report.extend(["", "All component metrics and stopping diagnostics are in iteration_comparison.csv. "
        "The 500-cap outputs are preserved in iterations_500/. Differences at the cap can reflect "
        "incomplete optimization; a smaller residual still does not establish correct factors."])
    (base_dir / "iteration_comparison.md").write_text("\n".join(report)+"\n")
    return rows


def save_spread(output_dir, blur, successful, true_spectra, detector):
    """Channel bands and concentration spread after truth-based alignment.

    All successful fits are matched to known simulation spectra and put on the
    unit-sum scale by evaluate_factors. Spread measures sensitivity to which
    pixels were added; it is not a confidence interval or proof of accuracy.
    Failed seeds are excluded from bands and retained in the run table.
    """
    if not successful:
        return [], [], {}
    spectra = np.stack([m.estimated_spectra for _, m in successful])
    counts = np.stack([m.estimated_abundances for _, m in successful])
    spectral_std = np.std(spectra, axis=0, ddof=1) if len(spectra) > 1 else np.full_like(spectra[0], np.nan)
    count_std = np.std(counts, axis=0, ddof=1) if len(counts) > 1 else np.full_like(counts[0], np.nan)
    median_spectra, median_counts = np.median(spectra, axis=0), np.median(counts, axis=0)
    low_spectra, high_spectra = spectra.min(axis=0), spectra.max(axis=0)
    spectral_rows, summary = [], []
    fig = Figure(figsize=(16, 8), dpi=130)
    axes = fig.subplots(2, 4)
    for k, name in enumerate(scene.COMPONENTS):
        for channel, wavelength in enumerate(detector.channel_centers):
            spectral_rows.append({"blur_fwhm_um": blur, "component": name,
                "wavelength_nm": wavelength, "successful_seeds": len(successful),
                "true_fraction": true_spectra[k, channel],
                "median_fraction": median_spectra[k, channel],
                "std_fraction": spectral_std[k, channel],
                "min_fraction": low_spectra[k, channel], "max_fraction": high_spectra[k, channel]})
        rmse = [m.abundance_rmse_per_component[k] for _, m in successful]
        shape_error = [m.spectral_relative_l2_per_component[k] for _, m in successful]
        summary.append({"blur_fwhm_um": blur, "component": name,
            "successful_seeds": len(successful), "requested_seeds": len(SAMPLING_SEEDS),
            "median_abundance_rmse": float(np.median(rmse)),
            "min_abundance_rmse": min(rmse), "max_abundance_rmse": max(rmse),
            "median_spectral_error": float(np.median(shape_error)),
            "max_channel_std": float(np.max(spectral_std[k])),
            "mean_pixel_count_std": float(np.mean(count_std[:, k])),
            "max_pixel_count_std": float(np.max(count_std[:, k]))})
        ax = axes[0, k]
        ax.plot(detector.channel_centers, true_spectra[k], "k", label="True")
        ax.plot(detector.channel_centers, median_spectra[k], label="Median recovered")
        ax.fill_between(detector.channel_centers, low_spectra[k], high_spectra[k], alpha=.25, label="Seed range")
        ax.set_title("Broad background" if name == scene.BACKGROUND_NAME else name)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction per channel")
        ax.legend(fontsize=7)
        ax = axes[1, k]
        im = ax.imshow(count_std[:, k].reshape(scene.IMAGE_SHAPE), cmap="magma", vmin=0,
                       vmax=max(float(np.nanmax(count_std)), 1e-10) if len(successful)>1 else 1)
        ax.set_title("Across-seed concentration SD (counts)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"Repeated subsampling · blur {blur:g} µm · {len(successful)} successful seeds · hull + {ADDED_BUDGET}")
    fig.text(.5,.02,"Fixed hull, starts and data. Spread is sampling sensitivity, not uncertainty or accuracy. Ground truth used for alignment/evaluation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(output_dir / f"spread_{blur:.2f}um.png")
    arrays = {f"blur_{blur:g}_spectra": spectra, f"blur_{blur:g}_counts": counts,
              f"blur_{blur:g}_median_counts": median_counts.reshape(*scene.IMAGE_SHAPE, 4),
              f"blur_{blur:g}_count_std": count_std.reshape(*scene.IMAGE_SHAPE, 4)}
    return spectral_rows, summary, arrays


@record_evaluation("base_dir")
def main(base_dir=OUTPUT_DIR, *, reuse_prerequisites=False):
    base_dir = Path(base_dir)
    prepare_prerequisites(base_dir, stage="evaluate_fitting_pool_expansion",
        required=("settings.json", "summary.csv", "fitted_factors.npz"),
        reuse=reuse_prerequisites, build=lambda: expansion.main(base_dir))
    settings = json.loads((base_dir / "settings.json").read_text())
    if settings["max_iterations"] != expansion.MAX_ITERATIONS or settings["tol_percent"] != expansion.TOL_PERCENT or settings["solver"] != expansion.SOLVER:
        raise ValueError("cached expansion settings must match the current solver settings")
    if tuple(settings["sampling_seeds"]) != expansion.SAMPLING_SEEDS:
        raise ValueError("cached baseline seeds differ from the current configuration")
    output_dir = base_dir / "repeated_sampling"
    output_dir.mkdir(exist_ok=True)
    with (base_dir / "summary.csv").open() as file:
        baseline = list(csv.DictReader(file))
    cached = np.load(base_dir / "fitted_factors.npz")
    rows, channel_rows, spread_rows, selections, arrays = [], [], [], [], {}
    seed_orders = {}
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = np.asarray(settings["true_unit_sum_spectra"])
    grid = SamplingGrid(scene.IMAGE_SHAPE, pixel_size_um=scene.PIXEL_SIZE_UM)
    for blur in settings["blur_fwhm_um"]:
        psf = None if blur==0 else GaussianPSF({name:blur for name in scene.COMPONENTS})
        spatial = render_scene(scene.build_scene(), grid, psf, oversample=settings["oversample"])
        truth = spatial.blurred_counts.reshape(-1,4)
        pixels = truth @ spectra
        foreground = truth[:,:3].sum(axis=1) >= scene.BACKGROUND_COUNTS
        first = next(e for e in settings["fit_selection"] if e["blur_fwhm_um"]==blur and e["pool"]=="hull")
        hull, starts = first["fit_indices"], np.asarray(first["initial_spectra"])
        successful = []
        for seed in SAMPLING_SEEDS:
            indices = expansion.expanded_pools(len(pixels), hull, (ADDED_BUDGET,), seed)[ADDED_BUDGET]
            reused = next((e for e in settings["fit_selection"] if e["blur_fwhm_um"]==blur
                and e["sampling_seed"]==seed and e["added_budget"]==ADDED_BUDGET), None)
            began = perf_counter()
            failure = ""
            try:
                if reused is not None:
                    np.testing.assert_array_equal(indices, reused["fit_indices"])
                    key = reused["factor_key"]
                    metrics = evaluate_factors(pixels, truth, spectra,
                              cached[key+"_abundances"], cached[key+"_spectra"])
                    old = next(r for r in baseline if float(r["blur_fwhm_um"])==blur
                        and int(r["sampling_seed"])==seed and int(r["added_budget"])==ADDED_BUDGET)
                    status, iterations, best = old["status"], int(old["iterations"]), int(old["best_iteration"])
                    history = reused["fit_history"]
                else:
                    with tqdm(total=expansion.MAX_ITERATIONS, desc=f"Robustness blur {blur:g} / seed {seed}", unit="iter", leave=False) as progress:
                        _, fitted, history, diagnostics = mcr_als(pixels[indices], starts.copy(),
                            nIter=expansion.MAX_ITERATIONS, normL="L1", tol_percent=expansion.TOL_PERCENT,
                            solver=expansion.SOLVER, return_diagnostics=True,
                            iteration_callback=lambda _:progress.update(1))
                    metrics, _ = diagnostic.evaluate_run(pixels, truth, spectra, fitted, foreground)
                    status, iterations, best = diagnostics.stop_reason, diagnostics.iterations_completed, diagnostics.best_iteration
                    history = history.tolist()
                successful.append((seed, metrics))
                selections.append({"blur_fwhm_um":blur,"seed":seed,"fit_indices":indices.tolist(),
                    "initial_spectra":starts.tolist(),"history":history,"reused_expansion":reused is not None})
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                status, iterations, best, failure = "failed", "", "", str(error)
                metrics = None
                selections.append({"blur_fwhm_um":blur,"seed":seed,"fit_indices":indices.tolist(),"failure":failure})
            for k,name in enumerate(scene.COMPONENTS):
                rows.append({"blur_fwhm_um":blur,"sampling_seed":seed,"component":name,
                    "status":status,"iterations":iterations,"best_iteration":best,"failure":failure,
                    "reused_expansion":reused is not None,"analysis_seconds":perf_counter()-began,
                    "spectral_error":metrics.spectral_relative_l2_per_component[k] if metrics else np.nan,
                    "abundance_rmse":metrics.abundance_rmse_per_component[k] if metrics else np.nan,
                    "reconstruction_relative_l2":metrics.reconstruction_relative_l2 if metrics else np.nan})
            expansion._write_csv(output_dir/"runs.csv", rows)
        seed_orders[str(blur)] = [seed for seed, _ in successful]
        channels, spread, saved = save_spread(output_dir, blur, successful, spectra, detector)
        channel_rows.extend(channels); spread_rows.extend(spread); arrays.update(saved)
    cached.close()
    if channel_rows:
        expansion._write_csv(output_dir/"spectral_spread.csv",channel_rows)
        expansion._write_csv(output_dir/"spread_summary.csv",spread_rows)
    np.savez_compressed(output_dir/"aligned_factors.npz",**arrays)
    (output_dir/"settings.json").write_text(json.dumps({"added_budget":ADDED_BUDGET,
        "sampling_seeds":SAMPLING_SEEDS,"base_settings":settings,
        "successful_seed_order":seed_orders,
        "selections":selections},indent=2)+"\n")
    report=["# Repeated subsampling robustness", "",
        f"Ten sampling seeds at hull + {ADDED_BUDGET}, fixed hull, starts, scene, "
        f"and {expansion.MAX_ITERATIONS}-update cap. The first three fits are reused from the budget sweep. "
        "This samples non-hull pixels without replacement; it is not bootstrap resampling.", "",
        "[Unblurred spread](spread_0.00um.png) · [Blurred spread](spread_0.65um.png)", "",
        "Bands show seed min/max; concentration maps show sample SD (ddof=1). "
        "Truth-based matching and unit-sum scaling align factors for this simulation. "
        "Spread measures conditional sampling sensitivity, not noise uncertainty or a confidence interval. "
        "Stable but incorrect factors are possible. A capped fit may remain unconverged. "
        "This does not assess uncertainty in hull selection, initial spectra, or acquisition noise.", "",
        "| Blur µm | Component | Successful seeds | Median count RMSE | RMSE range | Mean pixel SD | Max pixel SD |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in spread_rows:
        report.append(f"| {r['blur_fwhm_um']:g} | {r['component']} | {r['successful_seeds']} | "
            f"{r['median_abundance_rmse']:.4g} | {r['min_abundance_rmse']:.4g}–{r['max_abundance_rmse']:.4g} | "
            f"{r['mean_pixel_count_std']:.4g} | {r['max_pixel_count_std']:.4g} |")
    report.extend(["", "Failed seeds, if any, remain in runs.csv and are excluded from bands. "
        "Exact indices and histories are saved; aligned_factors.npz contains per-seed spectra/counts, "
        "median counts, and SD maps. Within each blur, successful seeds appear in ascending declared order; "
        "consult runs.csv for exclusions. Timing includes analysis/cache loading and is not comparable "
        "to the budget sweep's ALS-only timing."])
    (output_dir/"summary.md").write_text("\n".join(report)+"\n")
    if (base_dir / "iterations_500" / "settings.json").exists():
        compare_iteration_caps(base_dir)
    print(f"Saved repeated-sampling evaluation to {output_dir}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--base-dir", dest="output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reuse-prerequisites", action="store_true", help="Reuse a completed fitting-pool expansion in the output directory.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.output_dir, reuse_prerequisites=args.reuse_prerequisites)
