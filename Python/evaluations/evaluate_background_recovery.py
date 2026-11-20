"""Diagnose weak-background recovery on the existing noiseless spoke scene.

Run from Python: python -m evaluations.evaluate_background_recovery
No optimizer changes: compare fixed truth, ALS from truth, and blind starts.
"""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

from lumic import (essential_spectral_pixles, mcr_als, normalize_components,
                   positive_least_squares, unique_profiles)
from lumic.evaluation import (concentration_error_maps, evaluate_factors,
                              reconstruction_error_maps)
from lumic.simulation import (GaussianPSF, SamplingGrid, SpectralDetector,
                              render_scene, simulate_linear_mixture)
from . import evaluate_radial_spoke_blur as scene


# User settings. Keep the scene/counts/detector from the original experiment.
BLUR_FWHM_UM = (0.0, 0.65)
ITERATION_CHECKPOINTS = (0, 50, 200, 500)
TOL_PERCENT = 0.1
FIT_SEED = scene.SPOKE_SEED
MAX_FIT_PIXELS = 512
SOLVER = "auto"
RESULTS = Path(__file__).resolve().parent / "results" / "background_recovery"
PATHS = ("fixed_truth", "als_truth", "als_blind")
LABELS = {"fixed_truth": "Fixed true spectra", "als_truth": "ALS from truth",
          "als_blind": "ALS from blind selection"}


def select_starts(pixels):
    """Blind selection uses data alone, preserving the previous hull RNG seed."""
    state = np.random.get_state()
    np.random.seed(scene.SPOKE_SEED)
    try:
        candidates, hull_indices = essential_spectral_pixles(pixels, npc=4)
        starts, indices = unique_profiles(candidates.T, nr=4,
                                         f=scene.UNIQUE_PROFILES_F_PERCENT)
    finally:
        np.random.set_state(state)
    return normalize_components(starts, do_norm="L1"), hull_indices, hull_indices[indices]


def fit_spectra(pixels, starts, fitting_indices, limit, label):
    """Restart identical starts at each cap to inspect returned best factors.

    The public callback reports iteration numbers, not intermediate factors.
    Independent capped runs therefore provide checkpoints without changing
    the solver API. Their factors are best-returned factors, not necessarily
    the factors at the last executed update.
    """
    if limit == 0:
        return starts.copy(), np.array([]), None
    with tqdm(total=limit, desc=label, unit="iter", leave=False) as progress:
        _, spectra, history, diagnostics = mcr_als(
            pixels[fitting_indices], starts, nIter=limit, normL="L1",
            tol_percent=TOL_PERCENT, solver=SOLVER, return_diagnostics=True,
            iteration_callback=lambda _: progress.update(1),
        )
        progress.set_postfix_str(diagnostics.stop_reason)
    return spectra, history, diagnostics


def evaluate_run(pixels, truth, true_spectra, spectra, foreground):
    """Project factors across the full image, then compare with simulation truth."""
    counts = positive_least_squares(pixels, spectra, solver=SOLVER)
    metrics = evaluate_factors(pixels, truth, true_spectra, counts, spectra)
    difference = concentration_error_maps(truth, true_spectra, counts, spectra).signed
    regional = {}
    for name, mask in (("foreground", foreground), ("background_region", ~foreground)):
        # The background region includes faint blur tails below the declared
        # threshold; it is not necessarily an exactly fluorophore-free region.
        for component, component_name in enumerate(scene.COMPONENTS):
            delta = difference[mask, component]
            reference = truth[mask, component]
            denominator = np.linalg.norm(reference)
            regional[(name, component_name)] = {
                "pixels": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else float("nan"),
                "bias": float(np.mean(delta)) if delta.size else float("nan"),
                # Avoid unstable relative errors for absent/negligible truth.
                "relative_l2": (float(np.linalg.norm(delta) / denominator)
                                if denominator > 1e-8 * np.linalg.norm(truth[:, component])
                                and denominator > 0 else float("nan")),
            }
    return metrics, regional


def comparison_figure(path, fwhm, truth, spectra, starts, final, histories, wavelengths):
    """Show background distortions and all component concentration errors."""
    fig = Figure(figsize=(15, 13), dpi=120)
    axes = fig.subplots(5, 3)
    bg = len(scene.COMPONENTS) - 1
    shape = scene.IMAGE_SHAPE
    bg_max = max(float(truth[:, bg].max()),
                 *(float(final[k].estimated_abundances[:, bg].max()) for k in PATHS))
    errors = {k: concentration_error_maps(truth, spectra,
              final[k].estimated_abundances, final[k].estimated_spectra).signed
              for k in PATHS}
    error_max = max(float(np.abs(errors[k][:, bg]).max())
                    for k in PATHS)
    error_max = max(error_max, 1e-10)
    count_error_max = max(float(final[k].abundance_rmse_per_component.max()) for k in PATHS)
    spectrum_max = max(float(spectra[bg].max()),
                       *(float(final[k].estimated_spectra[bg].max()) for k in PATHS))
    for col, key in enumerate(PATHS):
        metrics = final[key]
        ax = axes[0, col]
        ax.plot(wavelengths, spectra[bg], "k", label="True")
        # Match starting factors separately; use truth only for evaluation.
        ax.plot(wavelengths, starts[key].estimated_spectra[bg], ":", label="Start")
        ax.plot(wavelengths, metrics.estimated_spectra[bg], "--", label="Returned")
        ax.set(title=LABELS[key] + "\nBackground spectrum", xlabel="Wavelength (nm)",
               ylabel="Fraction per channel", ylim=(0, spectrum_max * 1.1))
        ax.legend(fontsize=8)
        ax = axes[1, col]
        image = ax.imshow(metrics.estimated_abundances[:, bg].reshape(shape),
                          vmin=0, vmax=bg_max, cmap="viridis")
        ax.set_title(f"Recovered background (truth = {scene.BACKGROUND_COUNTS:g} counts)")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax = axes[2, col]
        image = ax.imshow(errors[key][:, bg].reshape(shape),
                          cmap="coolwarm", vmin=-error_max, vmax=error_max)
        ax.set_title("Background count error: recovered − true")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax = axes[3, col]
        ax.bar(np.arange(4), metrics.abundance_rmse_per_component)
        ax.set(xticks=np.arange(4), xticklabels=["EBFP2", "EGFP", "mCherry", "BG"],
               ylabel="Count RMSE", ylim=(0, max(count_error_max * 1.1, 1e-10)),
               title="All-pixel concentration errors")
        ax = axes[4, col]
        history = histories[key]
        if history.size:
            ax.semilogy(np.arange(history.size), np.maximum(history, 1e-16))
        else:
            ax.text(0.5, 0.5, "Fixed spectra: no ALS updates", ha="center", transform=ax.transAxes)
        ax.set(xlabel="Completed ALS updates", ylabel="Fit-subset relative L2",
               title="Reconstruction history")
        if history.size:
            ax.set_ylim(1e-16, 0.1)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Weak-background diagnosis · blur {fwhm:g} µm · no noise", fontsize=16)
    fig.text(0.5, 0.025, "Both ALS paths use identical fitting pixels. "
             "Full-image metrics; history display floor 1e-16. Low residual does not establish correct factors.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.7, wspace=0.45)
    fig.savefig(path)


def residual_figure(path, fwhm, pixels, truth, true_spectra, final):
    """Separate observable reconstruction residuals from truth-only errors.

    The prediction uses estimated factors only. Evaluation has reordered the
    factors against truth, but that permutation leaves their product intact.
    This noiseless experiment uses raw count residuals, not noise significance.
    """
    residual_rms = {}
    background_errors = {}
    for key in PATHS:
        factors = final[key]
        prediction = factors.estimated_abundances @ factors.estimated_spectra
        residual_rms[key] = reconstruction_error_maps(pixels, prediction).rms
        background_errors[key] = concentration_error_maps(
            truth, true_spectra, factors.estimated_abundances,
            factors.estimated_spectra).signed[:, -1]
    floor = 1e-10  # Display only; keep numerical residuals unchanged.
    maximum = max(max(float(values.max()) for values in residual_rms.values()), 1e-3)
    error_limit = max(max(float(np.abs(values).max()) for values in background_errors.values()), 1)
    fig = Figure(figsize=(14, 8), dpi=120)
    axes = fig.subplots(2, 3)
    for col, key in enumerate(PATHS):
        ax = axes[0, col]
        image = ax.imshow(np.maximum(residual_rms[key], floor).reshape(scene.IMAGE_SHAPE),
                          norm=LogNorm(vmin=floor, vmax=maximum), cmap="magma")
        ax.set_title(LABELS[key] + "\nResidual RMS/channel (counts)")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax = axes[1, col]
        image = ax.imshow(background_errors[key].reshape(scene.IMAGE_SHAPE),
                          vmin=-error_limit, vmax=error_limit, cmap="coolwarm")
        ax.set_title("Background count error\nRecovered − true; truth required")
        fig.colorbar(image, ax=ax, fraction=0.046)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(f"Residuals can miss incorrect factors · blur {fwhm:g} µm · no noise")
    fig.text(0.5, 0.025, "Top row needs measured data and fitted factors only. Bottom row needs simulation truth. "
             "Shared scales within each row; Residual scale: floor 1e-10; upper limit at least 1e-3 counts.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.86, bottom=0.1, hspace=0.35, wspace=0.4)
    fig.savefig(path)


def checkpoint_figure(path, rows):
    fig = Figure(figsize=(14, 8), dpi=120)
    axes = fig.subplots(2, 3)
    for row, fwhm in enumerate(BLUR_FWHM_UM):
        for key in PATHS:
            chosen = [r for r in rows if r["blur_fwhm_um"] == fwhm and r["path"] == key
                      and r["component"] == scene.BACKGROUND_NAME]
            for col, (field, title) in enumerate((
                ("reconstruction_relative_l2", "Full-image reconstruction error"),
                ("spectral_error", "Background spectral-shape error"),
                ("abundance_rmse", "Background count RMSE"),
            )):
                ax = axes[row, col]
                ax.plot([r["iteration_cap"] for r in chosen],
                        [max(r[field], 1e-16) for r in chosen], "o-", label=LABELS[key])
                ax.set(yscale="log", xlabel="Requested iteration cap",
                       title=f"{title}\nBlur {fwhm:g} µm")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
    fig.suptitle("Restarted checkpoints · best-returned factors · display floor 1e-16")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)


def main(output_dir=RESULTS, fitting_pool="hull"):
    if fitting_pool not in ("hull", "random_subset"):
        raise ValueError("fitting_pool must be hull or random_subset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (not ITERATION_CHECKPOINTS or ITERATION_CHECKPOINTS[0] != 0
            or any(not isinstance(n, int) or n < 0 for n in ITERATION_CHECKPOINTS)
            or tuple(sorted(set(ITERATION_CHECKPOINTS))) != ITERATION_CHECKPOINTS):
        raise ValueError("iteration checkpoints must be increasing integers starting at zero")
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = scene.detector_spectra(detector)
    grid = SamplingGrid(scene.IMAGE_SHAPE, pixel_size_um=scene.PIXEL_SIZE_UM)
    rows, regional_rows, final_rows, selection_rows = [], [], [], []
    for fwhm in BLUR_FWHM_UM:
        psf = None if fwhm == 0 else GaussianPSF({name: fwhm for name in scene.COMPONENTS})
        spatial = render_scene(scene.build_scene(), grid, psf, oversample=3)
        simulation = simulate_linear_mixture(spatial.blurred_counts, spectra,
                                             detector.channel_centers,
                                             fluorophore_names=scene.COMPONENTS)
        pixels = simulation.image.reshape(-1, detector.n_channels)
        truth = spatial.blurred_counts.reshape(-1, len(scene.COMPONENTS))
        foreground = truth[:, :3].sum(axis=1) >= scene.BACKGROUND_COUNTS
        blind_starts, hull_indices, selected_pixels = select_starts(pixels)
        # Intended image workflow: retain every hull candidate for ALS, not
        # only the four candidates selected as initial component estimates.
        # Both truth-start and blind-start controls use this identical pool.
        if fitting_pool == "hull":
            chosen = hull_indices
        else:
            chosen = np.random.default_rng(FIT_SEED).choice(
                len(pixels), min(MAX_FIT_PIXELS, len(pixels)), replace=False)
            chosen = np.unique(np.concatenate((chosen, selected_pixels)))
        selection_rows.append({"blur_fwhm_um": fwhm, "fit_indices": chosen.tolist(),
                               "foreground_fit_pixels": int(foreground[chosen].sum()),
                               "background_fit_pixels": int((~foreground[chosen]).sum())})
        final, starts, histories = {}, {}, {}
        for key in PATHS:
            initial = blind_starts if key == "als_blind" else spectra
            for limit in ITERATION_CHECKPOINTS:
                if key == "fixed_truth":
                    fitted, history, diagnostics = spectra, np.array([]), None
                else:
                    fitted, history, diagnostics = fit_spectra(
                        pixels, initial, chosen, limit, f"Blur {fwhm:g} / {key} / {limit}")
                metrics, regions = evaluate_run(pixels, truth, spectra, fitted, foreground)
                if limit == 0:
                    starts[key] = metrics
                final[key], histories[key] = metrics, history
                for k, component in enumerate(scene.COMPONENTS):
                    record = {"blur_fwhm_um": fwhm, "path": key, "iteration_cap": limit, "fitting_pool": fitting_pool,
                              "iterations": diagnostics.iterations_completed if diagnostics else 0,
                              "best_iteration": diagnostics.best_iteration if diagnostics else 0,
                              "status": diagnostics.stop_reason if diagnostics else "fixed_spectra",
                              "fit_pixels": len(chosen), "component": component,
                              "reconstruction_relative_l2": metrics.reconstruction_relative_l2,
                              "spectral_error": metrics.spectral_relative_l2_per_component[k],
                              "max_channel_error": float(np.max(np.abs(metrics.estimated_spectra[k] - spectra[k]))),
                              "abundance_rmse": metrics.abundance_rmse_per_component[k],
                              "abundance_relative_l2": metrics.abundance_relative_l2_per_component[k]}
                    rows.append(record)
                    for region in ("foreground", "background_region"):
                        regional_rows.append({**record, "region": region, **regions[(region, component)]})
                    if limit == ITERATION_CHECKPOINTS[-1]:
                        final_rows.append(record)
        comparison_figure(output_dir / f"background_{fwhm:.2f}um.png", fwhm,
                          truth, spectra, starts, final, histories, detector.channel_centers)
        residual_figure(output_dir / f"residuals_{fwhm:.2f}um.png", fwhm,
                        pixels, truth, spectra, final)
        scene._case_figure(output_dir / f"blind_factors_{fwhm:.2f}um.png", fwhm,
                           spatial, spectra, starts["als_blind"], final["als_blind"],
                           f"Blind path: {ITERATION_CHECKPOINTS[-1]}-update cap; "
                           f"{diagnostics.stop_reason}")
    for name, records in (("summary.csv", rows), ("regional_errors.csv", regional_rows)):
        with (output_dir / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    metadata = {"blur_fwhm_um": BLUR_FWHM_UM, "iteration_checkpoints": ITERATION_CHECKPOINTS,
                "tol_percent": TOL_PERCENT, "solver": SOLVER, "fit_seed": FIT_SEED,
                "fitting_pool": fitting_pool,
                "image_shape": scene.IMAGE_SHAPE, "pixel_size_um": scene.PIXEL_SIZE_UM,
                "spoke_counts": scene.SPOKE_COUNTS, "background_counts": scene.BACKGROUND_COUNTS,
                "background_center_nm": scene.BACKGROUND_CENTER_NM,
                "background_sigma_nm": scene.BACKGROUND_SIGMA_NM,
                "components": scene.COMPONENTS, "oversample": 3,
                "max_fit_pixels": MAX_FIT_PIXELS,
                "unique_profiles_f_percent": scene.UNIQUE_PROFILES_F_PERCENT,
                "n_spokes": scene.N_SPOKES, "spoke_seed": scene.SPOKE_SEED,
                "spoke_length_um": scene.SPOKE_LENGTH_UM, "spoke_width_um": scene.SPOKE_WIDTH_UM,
                "detector_profile": "980_32ch", "channel_centers_nm": detector.channel_centers.tolist(),
                "true_unit_sum_spectra": spectra.tolist(), "noise": "disabled",
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                "fit_selection": selection_rows}
    (output_dir / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoint_figure(output_dir / "checkpoints.png", rows)
    report = ["# Weak-background recovery diagnosis", "",
              "Noiseless existing radial-spoke scene: EBFP2, EGFP, mCherry, plus a uniform broad background.",
              f"Spokes: {scene.SPOKE_COUNTS:g} counts; background: {scene.BACKGROUND_COUNTS:g} counts/pixel. "
              f"Iteration caps: {ITERATION_CHECKPOINTS}; tolerance: {TOL_PERCENT:g}%.", "",
              f"Fitting pool: {fitting_pool}. Both ALS paths use identical fitting pixels. "
              "True spectra are supplied only to the diagnostic control paths. "
              "All paths are evaluated on the full image after fixed-reference abundance projection.", "",
              "Each checkpoint restarts from the same initial spectra and returns the solver's best factors. "
              "The x-axis is the requested cap, not necessarily the completed iteration count. "
              "The fixed-reference baseline is repeated as a horizontal comparison.", "",
              "[Checkpoint curves](checkpoints.png) · [Unblurred comparison](background_0.00um.png) · [Blurred comparison](background_0.65um.png)", "",
              "[Unblurred residuals versus concentration errors](residuals_0.00um.png) · "
              "[Blurred residuals versus concentration errors](residuals_0.65um.png). "
              "Residual maps use measured minus predicted channel intensities; concentration errors require truth. "
              "Raw residual RMS is not a calibrated noise significance score; no noise is present here.", "",
              "| Blur µm | Path | Component | Spectral error | Max channel error | Count RMSE | Reconstruction L2 | Iterations/status |",
              "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in final_rows:
        report.append(f"| {r['blur_fwhm_um']:g} | {r['path']} | {r['component']} | "
                      f"{r['spectral_error']:.4g} | {r['max_channel_error']:.4g} | "
                      f"{r['abundance_rmse']:.4g} | {r['reconstruction_relative_l2']:.4g} | "
                      f"{r['iterations']} / {r['status']} |")
    report += ["", "## Observations from this run", "",
               "Fixed true-spectrum fitting recovers all components to numerical precision in both conditions. "
               "ALS initialized with all true spectra preserves that solution and converges after one update. "
               "This supports the abundance solver and confirms a correct factorization is feasible on the shared fitting subset.", ""]
    for fwhm in BLUR_FWHM_UM:
        first = next(r for r in rows if r["blur_fwhm_um"] == fwhm and r["path"] == "als_blind"
                     and r["iteration_cap"] == 0 and r["component"] == scene.BACKGROUND_NAME)
        last = next(r for r in final_rows if r["blur_fwhm_um"] == fwhm and r["path"] == "als_blind"
                    and r["component"] == scene.BACKGROUND_NAME)
        report.append(f"- Blur {fwhm:g} µm: blind background shape error changes from "
                      f"{first['spectral_error']:.3g} to {last['spectral_error']:.3g}; "
                      f"background count RMSE changes from {first['abundance_rmse']:.3g} to "
                      f"{last['abundance_rmse']:.3g}. Full-image reconstruction improves from "
                      f"{first['reconstruction_relative_l2']:.3g} to {last['reconstruction_relative_l2']:.3g}. "
                      f"Stopping status: {last['status']}.")
    report += ["", "The blind background starts with the correct spectral shape, but the other starting spectra "
               "are mixed. A correct initial background alone therefore does not protect it during joint refinement. "
               "Inspect the maps and metrics for whether hull-only fitting changes the background distortion.",
               "", "## Interpretation boundaries", "",
               "Spectral error uses the existing unit-L2 shape comparison; max channel error uses unit-sum spectra. "
               "Count errors use unit-sum paired factors and matched component order. "
               "Foreground has at least 24 total fluorophore counts; the complementary region can contain faint blur tails. "
               "Absent/negligible regional truth gives undefined (NaN) relative error, while count RMSE stays meaningful.", "",
               "The correct truth-start control and distorted blind result implicate initialization, "
               "convergence and/or ambiguity, rather than establish a faulty abundance solver. "
               "A low residual with inaccurate factors is compatible with rotational ambiguity, "
               "but these runs alone do not prove nonuniqueness or diagnose weighting. "
               "A 500-update cap is a prototype limit; max_iterations must not be described as convergence.", "",
               "Settings and exact fitting indices are saved in settings.json. No weighting, constraints, "
               "solver implementation, or simulation features were added or changed."]
    report += ["", "## Literature context", "",
               "[Vitale and Ruckebusch (2023)](https://doi.org/10.1002/cem.3442) discuss loss "
               "of influence from essential observations in the presence of minor components. "
               "[Ahmad et al. (2023)](https://doi.org/10.1002/cem.3478) propose sample-relevance "
               "weights based on convex peeling. These are different from noise-variance weights; "
               "hull-only fitting here is not an implementation of their weighted method.", "",
               "[Hugelier, Vitale and Ruckebusch (2018)](https://doi.org/10.1177/0003702817735598) "
               "include a hyperspectral example with a fixed background spectrum estimated from "
               "background-only pixels. Such a constraint introduces prior information not used here.", "",
               "[Carabajal et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/37244664/) attribute abnormal "
               "MCR background profiles in chromatographic data to residual rotational ambiguity. "
               "This is a related factorization problem, not proof of the mechanism in this microscopy scene.", "",
               "[Skvortsov (2017)](https://doi.org/10.1002/cem.2831) discusses weighted MCR for "
               "unequal measurement uncertainty and notes that ambiguity can remain under different residual criteria. "
               "Our present scene is noiseless; those noise-based weights are not an established remedy here."]
    (output_dir / "summary.md").write_text("\n".join(report) + "\n")
    print(f"Saved background recovery evaluation to {output_dir}")
    return rows


def compare_pools(output_dir):
    """Compare identical simulations/starts under the two fitting pools."""
    output_dir = Path(output_dir)
    collections = {}
    settings = {}
    for pool, directory in (("hull", output_dir),
                            ("random_subset", output_dir / "random_subset_control")):
        with (directory / "summary.csv").open() as file:
            collections[pool] = list(csv.DictReader(file))
        settings[pool] = json.loads((directory / "settings.json").read_text())
    # Compare saved truth and stage-zero metrics to guard against accidentally
    # presenting a different scene or initialization as a pool-only change.
    for key in ("true_unit_sum_spectra", "channel_centers_nm", "image_shape",
                "spoke_counts", "background_counts", "iteration_checkpoints", "tol_percent"):
        if settings["hull"][key] != settings["random_subset"][key]:
            raise ValueError(f"pool comparison requires identical {key}")
    for left, right in zip(collections["hull"], collections["random_subset"]):
        if (left["blur_fwhm_um"], left["path"], left["iteration_cap"], left["component"]) != (
                right["blur_fwhm_um"], right["path"], right["iteration_cap"], right["component"]):
            raise ValueError("pool comparison requires identical checkpoint rows")
        if left["iteration_cap"] == "0":
            for field in ("spectral_error", "abundance_rmse", "reconstruction_relative_l2"):
                np.testing.assert_allclose(float(left[field]), float(right[field]),
                                           rtol=1e-12, atol=1e-14)
    records = []
    for pool, entries in collections.items():
        records.extend({**entry, "fitting_pool": pool} for entry in entries
                       if entry["path"] == "als_blind"
                       and int(entry["iteration_cap"]) == ITERATION_CHECKPOINTS[-1])
    for record in records:
        selection = next(entry for entry in settings[record["fitting_pool"]]["fit_selection"]
                         if entry["blur_fwhm_um"] == float(record["blur_fwhm_um"]))
        record["background_region_fit_pixels"] = selection["background_fit_pixels"]
        record["foreground_fit_pixels"] = selection["foreground_fit_pixels"]
    with (output_dir / "pool_comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    fig = Figure(figsize=(13, 7), dpi=120)
    axes = fig.subplots(2, 3)
    report = ["# Hull-only versus random-subset ALS fitting", "",
              "Identical noiseless scenes, blind starts, normalization, solver, tolerance, "
              "and 500-update caps. Only the fitting pool changes. Evaluation uses the full image.", "",
              "Hull-only uses every hull vertex; random-subset uses 512 random pixels plus the four selected starts.", "",
              "| Blur µm | Pool | Fit pixels | BG shape error | BG count RMSE | Reconstruction L2 | Iterations/status |",
              "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for row, fwhm in enumerate(BLUR_FWHM_UM):
        points = [next(r for r in records if float(r["blur_fwhm_um"]) == fwhm
                       and r["component"] == scene.BACKGROUND_NAME and r["fitting_pool"] == pool)
                  for pool in ("random_subset", "hull")]
        for r in points:
            report.append(f"| {fwhm:g} | {r['fitting_pool']} | {r['fit_pixels']} | "
                          f"{float(r['spectral_error']):.4g} | {float(r['abundance_rmse']):.4g} | "
                          f"{float(r['reconstruction_relative_l2']):.4g} | {r['iterations']} / {r['status']} |")
        for col, (field, title) in enumerate((("spectral_error", "Background shape error"),
                                             ("abundance_rmse", "Background count RMSE"),
                                             ("reconstruction_relative_l2", "Full-image reconstruction L2"))):
            ax = axes[row, col]
            ax.bar(["Random subset", "Hull-only"], [float(r[field]) for r in points],
                   color=["tab:blue", "tab:orange"])
            ax.set_title(f"{title} · blur {fwhm:g} µm")
            if field == "reconstruction_relative_l2":
                ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Same blind starts: better reconstruction can accompany worse background recovery")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "pool_comparison.png")
    report += ["", "## Representation of background pixels", "",
               "Background region: fewer than 24 total fluorophore counts; this includes faint blur tails.", "",
               "| Blur µm | Pool | Foreground fitting pixels | Background-region fitting pixels |",
               "| ---: | --- | ---: | ---: |"]
    for record in records:
        if record["component"] == scene.BACKGROUND_NAME:
            report.append(f"| {float(record['blur_fwhm_um']):g} | {record['fitting_pool']} | "
                          f"{record['foreground_fit_pixels']} | {record['background_region_fit_pixels']} |")
    report += ["", "Hull selection changes the representation of the background sharply. "
               "This supports testing sampling/weighting as a contributing mechanism, without establishing causality. "
               "The unblurred hull solution already reconstructs nearly exactly with incorrect factors; "
               "upweighting measurements cannot in general distinguish decompositions with identical predictions.", "",
               "[Comparison figure](pool_comparison.png) · [Hull-only report](summary.md) · "
               "[Random-subset control](random_subset_control/summary.md)", "",
               "In this experiment hull-only fitting worsens background spectral and count recovery "
               "while improving reconstruction. In the unblurred hull case, the solver converges "
               "to an almost exact reconstruction with substantially incorrect background factors. "
               "Truth-start controls preserve the correct solution for both pools. This supports "
               "multiple numerically valid decompositions as a concern; it does not establish "
               "that hull fitting is generally inferior or isolate all effects of initialization.", "",
               "Do not transfer the earlier random-subset conclusion to the intended hull-only "
               "workflow without these controls. Increasing iterations alone is not an established remedy."]
    (output_dir / "pool_comparison.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--fitting-pool", choices=("hull", "random_subset"), default="hull")
    parser.add_argument("--compare-random-subset", action="store_true",
                        help="also rerun the labelled random-subset control and compare pools")
    args = parser.parse_args()
    if args.compare_random_subset and args.fitting_pool != "hull":
        parser.error("--compare-random-subset requires the primary hull fitting pool")
    main(args.output_dir, args.fitting_pool)
    if args.compare_random_subset:
        main(args.output_dir / "random_subset_control", "random_subset")
        compare_pools(args.output_dir)
