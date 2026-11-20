"""Sum-bin the lowest-count spoke measurements before blind spectral discovery.

Run from Python: python -m evaluations.evaluate_radial_spoke_binning
The same Poisson realization is used for 1x1, 2x2 and 4x4 blocks. Spectra are
learned on the summed image and projected onto ORIGINAL noisy pixels. No new
noise is drawn after binning. This is an evaluation helper, not a new public API.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.spatial import QhullError
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from lumic import positive_least_squares
from lumic.evaluation import reconstruction_error_maps
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene
from .evaluate_radial_spoke_noise import (BLUR_FWHM_UM, COUNT_SCALES, NOISE_SEEDS,
    MAX_ITERATIONS, TOL_PERCENT, SELECTION_SEED, COMPONENTS, IMAGE_SHAPE, PIXEL_SIZE_UM,
    build_scene, detector_spectra, simulate_measurement, analyze_observed, audit_fit, plot_case)
from ._execution import record_evaluation, print_case_summary


BIN_FACTORS = (1, 2, 4)
COUNT_SCALE = min(COUNT_SCALES)
OUTPUT = Path(__file__).resolve().parent / "results/radial_spoke_noise/binning"


def sum_bin(image, factor):
    """Sum nonoverlapping square spatial blocks, keeping the channel axis intact.

    Channel-last finite nonnegative images only. Dimensions must be divisible
    by factor: never crop the scene or pad photons silently. Counts, field of
    view and spectral channels are preserved, while pixel spacing grows by
    factor. Unlike averaging, sums remain interpretable as detected counts.
    """
    image = np.asarray(image)
    if (image.ndim != 3 or not image.size or np.iscomplexobj(image)
            or not np.issubdtype(image.dtype, np.number)
            or not np.isfinite(image).all() or np.any(image < 0)):
        raise ValueError("image must be finite nonnegative channel-last spatial data")
    if (isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer))
            or factor < 1 or image.shape[0] % factor or image.shape[1] % factor):
        raise ValueError("factor must be a positive integer dividing both spatial dimensions")
    height, width, channels = image.shape
    return image.astype(float).reshape(height // factor, factor,
        width // factor, factor, channels).sum(axis=(1, 3))


def project_original(observed, discovery, folder):
    """Use learned spectra at original resolution; do not refine them again.

    The counts at original pixels retain their original noise. Discovery pool
    indices refer to the coarse grid and are retained only in the discovery
    folder, never relabeled as original-image fitting coordinates.
    """
    folder.mkdir(parents=True, exist_ok=True)
    counts = positive_least_squares(observed.reshape(-1, observed.shape[-1]),
        discovery["spectra"], do_norm=False, solver="auto").reshape(*observed.shape[:2], 4)
    if not np.isfinite(counts).all():
        raise RuntimeError("Original-resolution projection contains unresolved pixels")
    reconstructed = counts @ discovery["spectra"]
    errors = reconstruction_error_maps(observed, reconstructed)
    row = {"iterations": discovery["row"]["iterations"], "stop": discovery["row"]["stop"],
        "fit_pixels": discovery["row"]["fit_pixels"],
        "discovery_eligible_pixels": discovery["row"]["eligible_pixels"],
        "relative_l2": float(np.linalg.norm(errors.residuals) / np.linalg.norm(observed))}
    np.savez_compressed(folder / "fit.npz", abundances=counts,
                        spectra=discovery["spectra"], history=discovery["history"])
    settings = json.loads((discovery["folder"] / "settings.json").read_text())
    settings.update(projection="fixed learned spectra on original measured pixels",
                    discovery_folder=str(discovery["folder"].relative_to(folder.parent)),
                    original_image_shape=list(observed.shape))
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return dict(folder=folder, spectra=discovery["spectra"], row=row, rms=errors.rms)


def plot_comparison(output, blur, wave, true_spectra, rows):
    """One seed, shared original-pixel error scales; missing fits remain labeled."""
    fig = Figure(figsize=(13, 12), dpi=120); axes = fig.subplots(4, len(BIN_FACTORS))
    errors = []
    spectral_max = float(true_spectra.max())
    for factor in BIN_FACTORS:
        folder = output / f"blur_{blur:.2f}_seed_{NOISE_SEEDS[0]}_bin_{factor}" / "original_projection"
        if (folder / "truth_audit.npz").exists():
            with np.load(folder / "truth_audit.npz") as saved:
                errors.append(float(np.max(np.abs(saved["count_error"][..., :3]))))
                spectral_max = max(spectral_max, float(saved["matched_spectra"].max()))
    error_limit = max([1e-12, *errors])
    for col, factor in enumerate(BIN_FACTORS):
        row = next(r for r in rows if r["blur_fwhm_um"] == blur
                   and r["noise_seed"] == NOISE_SEEDS[0] and r["bin_factor"] == factor)
        if row["status"] != "ok":
            for ax in axes[:, col]:
                ax.text(.5, .5, "Discovery failed\nSee run table", ha="center", transform=ax.transAxes)
                ax.axis("off")
            axes[0, col].set_title(f"{factor}×{factor} sum bins")
            continue
        with np.load(output / row["folder"] / "original_projection/truth_audit.npz") as saved:
            fitted_spectra = saved["matched_spectra"]
            count_error = saved["count_error"]
        for k, name in enumerate(COMPONENTS):
            axes[0, col].plot(wave, true_spectra[k], ":", color=f"C{k}")
            axes[0, col].plot(wave, fitted_spectra[k], color=f"C{k}", label=name)
        axes[0, col].set(title=f"{factor}×{factor} bins · {PIXEL_SIZE_UM*factor:g} µm spacing",
                         xlabel="Wavelength (nm)", ylabel="Fraction per channel",
                         ylim=(0, 1.05 * spectral_max))
        axes[0, col].legend(fontsize=6)
        for k in range(3):
            h = axes[k+1, col].imshow(count_error[..., k], cmap="RdBu_r",
                                      vmin=-error_limit, vmax=error_limit)
            axes[k+1, col].set_title(f"{COMPONENTS[k]} error / ORIGINAL pixels", fontsize=9)
            axes[k+1, col].axis("off"); fig.colorbar(h, ax=axes[k+1, col], shrink=.7)
    fig.suptitle(f"Same low-count realization · blur {blur:g} µm · seed {NOISE_SEEDS[0]}\n"
                 "Spectra: dotted truth / solid recovered; error maps require simulation truth")
    fig.tight_layout(); fig.savefig(output / f"comparison_{blur:.2f}um.png")


def save_report(output, rows):
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Spatial binning before blind spectral discovery", "",
        "Lowest count scale only; same measured pixels for all bin factors. "
        "2×2 and 4×4 bins sum 4 and 16 original pixels, respectively. "
        "No extra noise is drawn and no truth is supplied to discovery.", "",
        "![No blur comparison](comparison_0.00um.png)", "",
        "![Blurred comparison](comparison_0.65um.png)", "",
        "## Per-seed results", "",
        "Spectral metrics describe learned signatures. Contribution and clean-image errors "
        "below are evaluated on ORIGINAL pixels, not coarse sums, to keep a common target.", "",
        "| Blur µm | Seed | Bin | ALS pixels | Updates / stop | Shape distance | Max channel difference | Original contribution relative error | Original clean-image error | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['blur_fwhm_um']} | {row['noise_seed']} | {row['bin_factor']} | "
            f"{row.get('fit_pixels', '')} | {row.get('iterations', '')} / {row.get('stop', '')} | "
            f"{row.get('spectral_shape_mean', np.nan):.4g} | {row.get('spectral_max_channel_change', np.nan):.4g} | "
            f"{row.get('abundance_relative_l2', np.nan):.4g} | {row.get('clean_image_relative_l2', np.nan):.4g} | {row['status']} |")
    lines += ["", "## Availability and seed spread", "",
        "Means and ranges use completed fits only; failed seeds are explicitly counted.", "",
        "| Blur µm | Bin | Completed / attempted | Shape mean [min, max] | Original contribution relative error mean [min, max] |",
        "| --- | --- | --- | --- | --- |"]
    for blur in BLUR_FWHM_UM:
        for factor in BIN_FACTORS:
            attempted = [r for r in rows if r["blur_fwhm_um"] == blur and r["bin_factor"] == factor]
            good = [r for r in attempted if r["status"] == "ok"]
            summaries = []
            for field in ("spectral_shape_mean", "abundance_relative_l2"):
                values = [r[field] for r in good]
                summaries.append(f"{np.mean(values):.4g} [{min(values):.4g}, {max(values):.4g}]" if values else "unavailable")
            lines.append(f"| {blur} | {factor} | {len(good)} / {len(attempted)} | {' | '.join(summaries)} |")
    lines += ["", "## Interpretation", "",
        "The coarse expected image and component maps are sums over the same blocks. "
        "Discovery audits in discovery/truth_audit.* compare against these coarse expected counts. "
        "original_projection/truth_audit.* instead compare original noisy-pixel fits against "
        "original expected contributions. Do not compare their absolute RMSE as equal-size pixels.", "",
        "Binning preserves total counts and field of view, increases sampling pitch and "
        "combines neighboring spectral mixtures. It can improve photon statistics for discovery "
        "while reducing purity/spatial detail. Original-resolution projection adds no new photons "
        "to those pixels; it does not recover lost spatial information or guarantee accurate factors. "
        "All cases keep the four-component hypothesis and 500-update cap. Cap hits, projection "
        "failures and alternative valid factorizations remain limitations. No thresholds or "
        "automatic rank selection are introduced."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


@record_evaluation("output")
def run(output=OUTPUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = detector_spectra(detector); wave = detector.channel_centers
    (output / "settings.json").write_text(json.dumps(dict(
        count_scale=COUNT_SCALE, blur_fwhm_um=BLUR_FWHM_UM, noise_seeds=NOISE_SEEDS,
        bin_factors=BIN_FACTORS, operation="sum measured pixels; no fresh noise",
        selection_seed=SELECTION_SEED, max_iterations=MAX_ITERATIONS, tol_percent=TOL_PERCENT,
        original_pixel_size_um=PIXEL_SIZE_UM, image_shape=IMAGE_SHAPE,
        projection="learn binned spectra, project onto original pixels"), indent=2) + "\n")
    rows = []
    for blur in BLUR_FWHM_UM:
        psf = None if not blur else GaussianPSF({name: blur for name in COMPONENTS})
        spatial = render_scene(build_scene(), SamplingGrid(IMAGE_SHAPE, PIXEL_SIZE_UM), psf, oversample=3)
        for seed in NOISE_SEEDS:
            truth, expected, observed = simulate_measurement(spatial.blurred_counts, spectra, COUNT_SCALE, seed)
            for factor in BIN_FACTORS:
                folder = output / f"blur_{blur:.2f}_seed_{seed}_bin_{factor}"
                folder.mkdir(exist_ok=True)
                row = dict(blur_fwhm_um=blur, noise_seed=seed, bin_factor=factor, folder=folder.name)
                tqdm.write(f"\nBinning comparison: {folder.name}")
                try:
                    binned = sum_bin(observed, factor)
                    with threadpool_limits(limits=1):
                        discovery = analyze_observed(binned, wave, folder / "discovery")
                        coarse_row, _, _ = audit_fit(binned, sum_bin(expected, factor),
                            sum_bin(truth, factor), spectra, discovery)
                        (folder / "discovery/evaluation.json").write_text(json.dumps(coarse_row, indent=2) + "\n")
                        projection = project_original(observed, discovery, folder / "original_projection")
                        original_row, metrics, initial = audit_fit(observed, expected, truth, spectra, projection)
                    row.update(original_row)
                    if seed == NOISE_SEEDS[0]:
                        plot_case(projection["folder"], observed, truth, spectra, wave, projection, metrics, initial)
                except (ValueError, RuntimeError, np.linalg.LinAlgError, QhullError) as error:
                    row.update(status=f"failed: {type(error).__name__}: {error}")
                rows.append(row)
                (folder / "evaluation.json").write_text(json.dumps(row, indent=2) + "\n")
                save_report(output, rows)
        plot_comparison(output, blur, wave, spectra, rows)
    print(f"Saved binning comparison to {output}", flush=True)
    print_case_summary(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
