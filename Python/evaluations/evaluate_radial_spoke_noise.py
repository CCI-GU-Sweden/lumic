"""Shot noise in the classical three-fluorophore-plus-background scene.

Run from Python: python -m evaluations.evaluate_radial_spoke_noise
Selection and fitting see only observed data. Simulation truth enters the
subsequent audit. This exploratory report is not a CI recovery assertion.
"""

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from matplotlib.figure import Figure
from scipy.spatial import QhullError
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from lumic import positive_least_squares
from lumic.evaluation import (compare_spectra, evaluate_factors,
    reconstruction_error_maps, residual_neighbor_correlation,
    residual_spectral_correlation)
from lumic.simulation import (GaussianPSF, SamplingGrid, SpectralDetector,
    add_confocal_noise, render_scene)
from lumic.utils import RGB_from_spec, select_analysis_pixels
from .evaluate_radial_spoke_blur import (BACKGROUND_COUNTS, COMPONENTS,
    IMAGE_SHAPE, PIXEL_SIZE_UM, SPOKE_COUNTS, SPOKE_SEED,
    build_scene, detector_spectra)
from ._residual_workflow import convex_hull_pool, fit_pool, inspect_fit_residuals
from ._execution import record_evaluation, print_case_summary


# Human inputs. Global scaling preserves all fluorophore/background ratios.
# These are ideal detected counts; no real-instrument gain calibration implied.
BLUR_FWHM_UM = (0., .65)
COUNT_SCALES = (1., .25, .0625)
NOISE_SEEDS = (2026, 2027, 2028)
SELECTION_SEED = 2026  # Fixed independently of the measurement noise seed.
MAX_ITERATIONS = 500
LONG_ITERATIONS = 2000
TOL_PERCENT = .01
RELATIVE_FLOOR = 1e-10  # Arithmetic floor, not a noise-based selection rule.
OUTPUT = Path(__file__).resolve().parent / "results/radial_spoke_noise"


def simulate_measurement(counts, spectra, scale, noise_seed):
    """Scale component counts, mix, then optionally draw per-channel Poisson noise.

    None means a noiseless control, not seed zero. Return expected component
    contributions, the expected detector image and the measured detector image.
    Keep all three separate: a random realization is not the underlying truth.
    """
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    truth = np.asarray(counts, dtype=float) * scale
    expected = truth @ spectra
    observed = (expected.copy() if noise_seed is None else
                add_confocal_noise(expected, rng=np.random.default_rng(noise_seed)))
    return truth, expected, observed


def analyze_observed(observed, wave, folder, max_iterations=MAX_ITERATIONS):
    """Blind four-component hull-only fit; no true spectra/maps are accepted.

    Keep all valid nonzero measured pixels (no median-intensity cutoff), including
    low-count background. A pure-zero noise realization cannot define a spectrum.
    Each realization gets its own hull and starts; the selection seed is fixed.
    Rank is specified by the researcher and never selected automatically.
    """
    image = np.moveaxis(observed, -1, 0)
    retained = select_analysis_pixels(image, median_quantile=0.)
    pool = convex_hull_pool(image, retained, npc=4, seed=SELECTION_SEED)
    fit = fit_pool(image, pool, n_components=4, output_dir=folder,
                   seed=SELECTION_SEED, max_iterations=max_iterations,
                   tol_percent=TOL_PERCENT, norm="L2")
    inspect_fit_residuals(image, fit, wave)
    fit["row"].update(eligible_pixels=len(retained.indices),
                      excluded_zero_pixels=retained.exclusion_counts["zero_signal"])
    return fit


def audit_fit(observed, expected, truth, true_spectra, fit):
    """Post-fit truth audit and observed residual diagnostics, kept distinct.

    Oracle noise residuals are observed minus the KNOWN expected image. Fitted
    residuals use the same data that selected/refined the factors; they can be
    smaller than oracle noise through fitting. Their ratio is not a goodness-of-
    fit test. Pearson correlations are descriptive, not calibrated significance.
    """
    with np.load(fit["folder"] / "fit.npz") as saved:
        counts = saved["abundances"].reshape(-1, 4)
    metrics = evaluate_factors(observed.reshape(-1, observed.shape[-1]),
        truth.reshape(-1, 4), true_spectra, counts, fit["spectra"])
    comparison = compare_spectra(true_spectra, metrics.estimated_spectra, match=False)
    reconstructed = counts.reshape(truth.shape) @ fit["spectra"]
    errors = reconstruction_error_maps(observed, reconstructed)
    informative = fit["row"]["relative_l2"] > RELATIVE_FLOOR
    spectral = (residual_spectral_correlation(errors.residuals) if informative
                else np.full(observed.shape[:-1], np.nan))
    valid = spectral[np.isfinite(spectral)]
    oracle_rms = float(np.sqrt(np.mean((observed - expected)**2)))
    fitted_rms = float(np.sqrt(np.mean(errors.residuals**2)))
    clean_relative = float(np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected))
    # A separate initial-spectrum audit quantifies what ALS changed, without
    # ever using true signatures to initialize or select the fitting pool.
    settings = json.loads((fit["folder"] / "settings.json").read_text())
    starts = np.asarray(settings["initial_spectra"])
    starting_counts = positive_least_squares(observed.reshape(-1, observed.shape[-1]),
        starts, do_norm=False, solver="auto")
    initial = evaluate_factors(observed.reshape(-1, observed.shape[-1]),
        truth.reshape(-1, 4), true_spectra, starting_counts, starts)
    row = dict(fit["row"], status="ok", clean_image_relative_l2=clean_relative,
        spectral_shape_mean=float(comparison.shape_distance.mean()),
        spectral_max_channel_change=float(comparison.max_channel_change.max()),
        abundance_relative_l2=metrics.abundance_relative_l2_global,
        abundance_rmse=metrics.abundance_rmse_global,
        initial_spectral_relative_l2=initial.spectral_relative_l2_global,
        final_spectral_relative_l2=metrics.spectral_relative_l2_global,
        fitted_residual_rms=fitted_rms, oracle_noise_rms=oracle_rms,
        expected_shot_noise_rms=float(np.sqrt(expected.mean())),
        residual_to_oracle_rms=fitted_rms / oracle_rms if oracle_rms else None,
        neighbor_correlation=(residual_neighbor_correlation(errors.residuals)
                              if informative else None),
        mean_spectral_correlation=float(valid.mean()) if valid.size else None,
        residual_above_floor=informative)
    component_rows = []
    for k, name in enumerate(COMPONENTS):
        component_rows.append(dict(component=name,
            shape_distance=float(comparison.shape_distance[k]),
            max_channel_change=float(comparison.max_channel_change[k]),
            count_rmse=float(metrics.abundance_rmse_per_component[k]),
            count_relative_l2=float(metrics.abundance_relative_l2_per_component[k])))
    np.savez_compressed(fit["folder"] / "truth_audit.npz",
        matched_counts=metrics.estimated_abundances.reshape(truth.shape),
        matched_spectra=metrics.estimated_spectra,
        matched_initial_spectra=initial.estimated_spectra,
        count_error=metrics.estimated_abundances.reshape(truth.shape) - truth,
        residual_mean=errors.residuals.mean(axis=(0, 1)),
        spectral_correlation=spectral)
    (fit["folder"] / "truth_audit.json").write_text(json.dumps(component_rows, indent=2) + "\n")
    return row, metrics, initial


def plot_case(folder, observed, truth, true_spectra, wave, fit, metrics, initial):
    """Human-readable spectra and contributions, with explicit truth-only panels."""
    recovered = metrics.estimated_abundances.reshape(truth.shape)
    fig = Figure(figsize=(15, 11), dpi=120)
    axes = fig.subplots(4, 4)
    for k, name in enumerate(COMPONENTS):
        axes[0, k].plot(wave, true_spectra[k], "k", label="True")
        axes[0, k].plot(wave, initial.estimated_spectra[k], ":", label="Blind start")
        axes[0, k].plot(wave, metrics.estimated_spectra[k], label="After ALS")
        axes[0, k].set(title=name, xlabel="Wavelength (nm)", ylabel="Channel fraction")
        axes[0, k].legend(fontsize=7)
        maximum = max(float(truth[..., k].max()), float(recovered[..., k].max()), 1e-12)
        difference = recovered[..., k] - truth[..., k]
        limit = max(float(np.max(np.abs(difference))), 1e-12)
        for row, data, title, cmap, low, high in (
            (1, truth[..., k], "True expected counts", "viridis", 0, maximum),
            (2, recovered[..., k], "Recovered counts", "viridis", 0, maximum),
            (3, difference, "Recovery error / truth required", "RdBu_r", -limit, limit)):
            h = axes[row, k].imshow(data, cmap=cmap, vmin=low, vmax=high)
            axes[row, k].set_title(title, fontsize=9)
            axes[row, k].axis("off")
            fig.colorbar(h, ax=axes[row, k], shrink=.7)
    fig.suptitle(f"{folder.name} · {fit['row']['iterations']} updates · {fit['row']['stop']}")
    fig.tight_layout(); fig.savefig(folder / "recovery.png")
    fig = Figure(figsize=(9, 4), dpi=120); axes = fig.subplots(1, 2)
    axes[0].imshow(RGB_from_spec(np.moveaxis(observed, -1, 0), do_norm=True))
    axes[0].set_title("Measured spectral RGB / display normalized"); axes[0].axis("off")
    h = axes[1].imshow(fit["rms"], cmap="magma")
    axes[1].set_title("Fitted residual RMS / counts"); axes[1].axis("off")
    fig.colorbar(h, ax=axes[1]); fig.tight_layout(); fig.savefig(folder / "measured_and_residual.png")


def save_report(output, rows):
    """Keep every seed and failure; plotted min/max ranges are not confidence intervals."""
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    fig = Figure(figsize=(12, 8), dpi=130); axes = fig.subplots(2, 3)
    labels = (("relative_l2", "Fit error vs measured image"),
              ("clean_image_relative_l2", "Prediction error vs clean image / truth"),
              ("spectral_shape_mean", "Mean spectral shape distance / truth"),
              ("abundance_relative_l2", "Contribution relative L2 / truth"),
              ("neighbor_correlation", "Spatial residual correlation / descriptive"),
              ("residual_to_oracle_rms", "Fitted RMS / oracle noise RMS / truth"))
    main = [r for r in rows if r["status"] == "ok" and not r["long_run"]]
    for ax, (metric, title) in zip(axes.flat, labels):
        for blur in BLUR_FWHM_UM:
            x, y, low, high = [], [], [], []
            for scale in sorted(COUNT_SCALES):
                values = [r[metric] for r in main if r["blur_fwhm_um"] == blur
                          and r["count_scale"] == scale and r["noise_seed"] is not None
                          and r.get(metric) is not None and np.isfinite(r[metric])]
                # NaN gaps prevent connecting curves across failed conditions.
                x.append(SPOKE_COUNTS * scale)
                y.append(np.mean(values) if values else np.nan)
                low.append(min(values) if values else np.nan)
                high.append(max(values) if values else np.nan)
                if ax is axes[0, 0] and len(values) < len(NOISE_SEEDS):
                    ax.text(SPOKE_COUNTS * scale, .06 + .07 * BLUR_FWHM_UM.index(blur),
                            f"{blur:g} µm: {len(values)}/{len(NOISE_SEEDS)}",
                            transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
            if x:
                line, = ax.plot(x, y, "o-", label=f"{blur:g} µm / noisy mean")
                ax.fill_between(x, low, high, color=line.get_color(), alpha=.15)
            controls = sorted([r for r in main if r["blur_fwhm_um"] == blur
                and r["noise_seed"] is None and r.get(metric) is not None], key=lambda r: r["count_scale"])
            if controls:
                ax.plot([SPOKE_COUNTS * r["count_scale"] for r in controls],
                        [r[metric] for r in controls], ":", label=f"{blur:g} µm / noiseless")
        ax.set(xscale="log", xlim=(SPOKE_COUNTS * min(COUNT_SCALES) * .8,
                                  SPOKE_COUNTS * max(COUNT_SCALES) * 1.2),
               xlabel="Expected counts per spoke (overlaps add)", ylabel=title)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle("Classical 3-fluorophore + background spokes · mean and seed min–max")
    fig.tight_layout(); fig.savefig(output / "overview.png")
    lines = ["# Classical radial spokes with shot noise", "",
        "EBFP2, EGFP, mCherry and broad background; four components specified by the researcher.",
        "Hull-only fitting; no residual pool expansion or automatic component-count choice.", "",
        "![Noise sweep](overview.png)", "",
        "Curves show means and min–max ranges of successful fits only, not confidence intervals. "
        "Gaps mark conditions with no successful fits; annotations show partial/zero success counts. "
        "Missing/failed fits remain in the table; no successful recovery is required.", "",
        "| Blur µm | Scale | Noise seed | Cap | Status / stop | Updates | Measured relative fit | Clean relative error | Shape distance | Contribution relative error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['blur_fwhm_um']} | {r['count_scale']} | {r['noise_seed']} | {r['max_iterations']} | "
            f"{r.get('stop', r['status'])} | {r.get('iterations', '')} | {r.get('relative_l2', np.nan):.4g} | "
            f"{r.get('clean_image_relative_l2', np.nan):.4g} | {r.get('spectral_shape_mean', np.nan):.4g} | "
            f"{r.get('abundance_relative_l2', np.nan):.4g} |")
    lines += ["", "## Run availability", "",
        "| Blur µm | Count scale | Successful noisy fits / requested |",
        "| --- | --- | --- |"]
    for blur in BLUR_FWHM_UM:
        for scale in COUNT_SCALES:
            attempts = [r for r in rows if r["blur_fwhm_um"] == blur
                and r["count_scale"] == scale and r["noise_seed"] is not None and not r["long_run"]]
            lines.append(f"| {blur} | {scale} | {sum(r['status']=='ok' for r in attempts)} / {len(attempts)} |")
    lines += ["", "## Longer-run check", ""]
    longer = [r for r in rows if r["long_run"] and r["status"] == "ok"]
    if not longer:
        lines.append("No capped condition has yet been rerun with the longer cap.")
    for r in longer:
        previous = next(p for p in main if p["blur_fwhm_um"] == r["blur_fwhm_um"]
            and p["count_scale"] == r["count_scale"] and p["noise_seed"] == r["noise_seed"])
        lines.append(f"Same measurement/starts at {r['blur_fwhm_um']} µm, scale {r['count_scale']}, "
            f"seed {r['noise_seed']}: {previous['max_iterations']} → {r['max_iterations']} updates. "
            f"Measured relative residual {previous['relative_l2']:.4g} → {r['relative_l2']:.4g}; "
            f"mean spectral shape distance {previous['spectral_shape_mean']:.4g} → {r['spectral_shape_mean']:.4g}; "
            f"contribution relative error {previous['abundance_relative_l2']:.4g} → {r['abundance_relative_l2']:.4g}. "
            f"Long-run stop: {r['stop']}.")
    lines += ["", "## Interpretation", "",
        "The same scene is scaled globally; component ratios, geometry and blur stay fixed. "
        "Noise is independent Poisson sampling of the complete expected detector spectrum. "
        "The global theoretical noise RMS is sqrt(mean expected counts); oracle RMS uses "
        "the realized noise against known truth. Fitting residuals are correlated with fitted "
        "data and may be smaller than oracle noise, without establishing correct factors.", "",
        "Each realization gets its own normalized hull/starts with a fixed selection seed; "
        "noise affects both initialization and refinement. All usable nonzero pixels are "
        "eligible without a median-intensity cutoff. Low-count noisy extremes can be selected. "
        "Noiseless amplitude-scaled controls check scaling separately from stochastic noise. "
        "This evaluation uses all hull pixels for ALS, whereas the older radial-spoke blur "
        "script uses a random fitting subset. Compare noise against the controls in THIS report.", "",
        "A recorded hull-projection failure is an implementation/conditioning boundary, "
        "not proof that information is physically unrecoverable. Successful early stopping "
        "only means the fit-improvement criterion was met; it does not certify accurate factors.", "",
        "Inspect representative recovery.png, measured_and_residual.png, residual_pca.png, "
        "and truth_audit.json/npz in each case. Spectral signs, PC ordering and plot scales "
        "can differ. Correlations describe structure, not significance; constant or "
        "near-arithmetic-zero residuals are uninformative. Expected counts are idealized, "
        "not calibrated confocal detector units. No read noise, thresholds or acquisition "
        "automation are introduced. Cap hits and longer-run controls remain explicit."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


@record_evaluation("output")
def run(output=OUTPUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = detector_spectra(detector); wave = detector.channel_centers
    settings = dict(components=COMPONENTS, image_shape=IMAGE_SHAPE, pixel_size_um=PIXEL_SIZE_UM,
        spoke_counts=SPOKE_COUNTS, background_counts=BACKGROUND_COUNTS, spoke_seed=SPOKE_SEED,
        blur_fwhm_um=BLUR_FWHM_UM, count_scales=COUNT_SCALES, noise_seeds=NOISE_SEEDS,
        selection_seed=SELECTION_SEED, max_iterations=MAX_ITERATIONS,
        long_iterations=LONG_ITERATIONS, tol_percent=TOL_PERCENT, noise="Poisson only",
        pool="all normalized hull vertices", median_quantile=0, spectral_norm="L2",
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []; long_checked = False
    for blur in BLUR_FWHM_UM:
        psf = None if not blur else GaussianPSF({name: blur for name in COMPONENTS})
        spatial = render_scene(build_scene(), SamplingGrid(IMAGE_SHAPE, PIXEL_SIZE_UM), psf, oversample=3)
        for scale in COUNT_SCALES:
            for seed in (None, *NOISE_SEEDS):
                truth, expected, observed = simulate_measurement(spatial.blurred_counts, spectra, scale, seed)
                label = f"blur_{blur:.2f}_scale_{scale:g}_seed_{seed}"
                caps = [MAX_ITERATIONS]
                for cap in caps:
                    folder = output / (label if cap == MAX_ITERATIONS else label + "_long")
                    folder.mkdir(exist_ok=True)
                    identity = dict(blur_fwhm_um=blur, count_scale=scale, noise_seed=seed,
                                    max_iterations=cap, long_run=cap != MAX_ITERATIONS)
                    tqdm.write(f"\n{folder.name} / up to {cap} ALS updates")
                    try:
                        with threadpool_limits(limits=1):
                            fit = analyze_observed(observed, wave, folder, max_iterations=cap)
                            row, metrics, initial = audit_fit(observed, expected, truth, spectra, fit)
                        if seed in (None, NOISE_SEEDS[0]) or cap != MAX_ITERATIONS:
                            plot_case(folder, observed, truth, spectra, wave, fit, metrics, initial)
                        # Repeat the first capped condition at the longer
                        # cap using identical data, hull and deterministic starts.
                        if row["iterations"] == cap and not long_checked:
                            caps.append(LONG_ITERATIONS); long_checked = True
                        rows.append(dict(identity, **row))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, QhullError) as error:
                        rows.append(dict(identity, status=f"failed: {type(error).__name__}: {error}"))
                    (folder / "evaluation.json").write_text(json.dumps(rows[-1], indent=2) + "\n")
                    save_report(output, rows)  # Preserve partial reports if interrupted.
    print(f"Saved shot-noise evaluation to {output}", flush=True)
    print_case_summary(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
