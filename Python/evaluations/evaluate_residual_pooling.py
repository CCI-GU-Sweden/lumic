"""One observed-data-only round of random and spatial-residual pool additions.

Run from Python: python -m evaluations.evaluate_residual_pooling
analyze_image receives only a spectral image and wavelengths. The entry point
creates the existing noiseless scene and performs a separate truth audit AFTER
analysis; that audit never guides selection, fitting or spatial diagnostics.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import skimage
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

from lumic import mcr_als, positive_least_squares
from lumic.evaluation import align_factors, compare_spectra, evaluate_factors, reconstruction_error_maps
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene
from . import evaluate_background_recovery as diagnostic
from . import evaluate_fitting_pool_expansion as expansion
from . import evaluate_radial_spoke_blur as scene
from ._residual_pooling import addition_pools, candidate_masks, spatial_diagnostics
from ._execution import record_evaluation


# Bounded pilot, not an adaptive optimization loop or noise sweep.
BLUR_FWHM_UM = (0., .65)
ADDED_BUDGET = 256
SAMPLING_SEEDS = (2026, 2027, 2028)
HIGH_RESIDUAL_QUANTILE = .8
OPENING_RADIUS_PIXELS = 1
MAX_ITERATIONS = 500
TOL_PERCENT = .1
RELATIVE_RESIDUAL_FLOOR = 1e-10
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "residual_pooling"


def analyze_image(image, wavelengths, output_dir):
    """Compare four-component pool strategies using observable data only.

    image is (y, x, channels), not the CZI (channel, y, x) layout. All strategies
    keep the original hull and restart the same blind spectra with identical
    solver settings. Residual shortlists come from the hull-only fit and remain
    fixed for all seeds. There is one selection-and-refit round. Empty or
    near-zero residual shortlists are reported as unavailable, not filled with
    unrelated pixels. Common addition budgets are reduced if a nonempty spatial
    shortlist cannot supply the requested number.

    Full-image residuals, correlations and spectral drift describe sensitivity,
    not recovery accuracy. This same image informs selection AND reporting:
    these are not independent validation scores. Raw RMS/spatial correlation
    are not noise-whitened and must not be used to rank noisy real images as if
    all detector units were photon counts. Opening is a spatial-support heuristic,
    not a statistical test for nonrandom noise or an ALS spatial regularizer.
    """
    image = np.asarray(image, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if (image.ndim != 3 or not image.size or not np.isfinite(image).all()
            or np.any(image < 0) or not np.any(image)
            or wavelengths.shape != (image.shape[-1],) or not np.isfinite(wavelengths).all()):
        raise ValueError("provide a finite nonnegative nonzero channel-last image and matching wavelengths")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixels = image.reshape(-1, image.shape[-1])
    starts, hull, anchors = diagnostic.select_starts(pixels)  # Fixed four-component pilot.
    hull = np.unique(hull)
    rows, spectral_rows, fits = [], [], {}

    def fit_pool(strategy, seed, indices, actual_budget):
        began = perf_counter()
        label = f"{output_dir.name} / {strategy} / seed {seed}"
        with tqdm(total=MAX_ITERATIONS, desc=label, unit="iter", leave=False) as progress:
            _, spectra, history, info = mcr_als(pixels[indices], starts.copy(), nIter=MAX_ITERATIONS,
                normL="L1", tol_percent=TOL_PERCENT, solver="auto", return_diagnostics=True,
                iteration_callback=lambda _: progress.update(1))
        counts = positive_least_squares(pixels, spectra, solver="auto")
        aligned = align_factors(starts, counts, spectra)
        prediction = (aligned.abundances @ aligned.spectra).reshape(image.shape)
        maps = reconstruction_error_maps(image, prediction)
        stats = spatial_diagnostics(image, maps.residuals, relative_floor=RELATIVE_RESIDUAL_FLOOR)
        row = {"strategy": strategy, "seed": seed, "added_pixels": actual_budget,
            "fit_pixels": len(indices), "iterations": info.iterations_completed,
            "best_iteration": info.best_iteration, "stop": info.stop_reason,
            "seconds_including_projection": perf_counter()-began, **stats}
        key = f"{strategy}_{seed}"
        fits[key] = {"row": row, "indices": indices, "counts": aligned.abundances,
                     "spectra": aligned.spectra, "history": history, "maps": maps}
        rows.append(row)
        return fits[key]

    baseline = fit_pool("hull", -1, hull, 0)
    eligible = np.linalg.norm(image, axis=-1) > 0
    eligible.flat[hull] = False
    masks = candidate_masks(baseline["maps"].rms, eligible,
        quantile=HIGH_RESIDUAL_QUANTILE, opening_radius=OPENING_RADIUS_PIXELS,
        informative=baseline["row"]["residual_above_floor"])
    selections = []
    unavailable = set()
    for seed in SAMPLING_SEEDS:
        additions = addition_pools(hull, eligible, masks, ADDED_BUDGET, seed)
        unavailable.update(additions["unavailable"])
        for strategy, indices in additions["pools"].items():
            fit_pool(strategy, seed, indices, additions["actual_budget"])
        selections.append({"seed": seed, "actual_budget": additions["actual_budget"],
            "candidate_counts": additions["candidate_counts"], "unavailable": additions["unavailable"]})
    arrays = {"blind_starts": starts, "hull_indices": hull, "eligible_mask": eligible,
              "high_mask": masks["high"], "opened_mask": masks["spatial"], "region_labels": masks["labels"]}
    for key, fitted in fits.items():
        change = compare_spectra(baseline["spectra"], fitted["spectra"], match=False)
        for k in range(len(starts)):
            spectral_rows.append({"strategy": fitted["row"]["strategy"], "seed": fitted["row"]["seed"],
                "component_id": k+1, "shape_change_from_hull": change.shape_distance[k],
                "max_channel_change_from_hull": change.max_channel_change[k]})
        for field in ("counts", "spectra", "history", "indices"):
            arrays[f"{key}_{field}"] = fitted[field]
        arrays[f"{key}_rms"] = fitted["maps"].rms
        arrays[f"{key}_relative_l2"] = fitted["maps"].relative_l2
    expansion._write_csv(output_dir/"observed_metrics.csv", rows)
    expansion._write_csv(output_dir/"spectral_drift.csv", spectral_rows)
    np.savez_compressed(output_dir/"factors_and_masks.npz", **arrays)
    settings = {"image_shape": image.shape, "components_requested": len(starts),
        "wavelengths_nm": wavelengths.tolist(), "max_iterations": MAX_ITERATIONS,
        "tol_percent": TOL_PERCENT, "solver": "auto", "norm": "L1",
        "requested_budget": ADDED_BUDGET, "seeds": SAMPLING_SEEDS,
        "quantile": HIGH_RESIDUAL_QUANTILE, "threshold_rms": masks["threshold"],
        "opening_radius_pixels": OPENING_RADIUS_PIXELS, "connectivity": 1,
        "relative_residual_floor": RELATIVE_RESIDUAL_FLOOR, "selection_reason": masks["reason"],
        "initial_spectra": starts.tolist(), "selected_start_indices": anchors.tolist(),
        "selections": selections, "noise_scaling": "none; noiseless pilot",
        "fit_selection": [{"strategy": f["row"]["strategy"], "seed": f["row"]["seed"],
                           "indices": f["indices"].tolist()} for f in fits.values()],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "skimage": skimage.__version__}}
    (output_dir/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    plot_case(image, wavelengths, fits, masks, hull, output_dir)
    report = ["# Observable residual-pooling pilot", "",
        "Selection, fitting, spatial diagnostics and anonymous component matching use no "
        "ground-truth spectra or concentrations. The four-component count is a declared prior. "
        "This is one round on the same image, not independent validation or calibrated model selection.", "",
        f"Requested additions: {ADDED_BUDGET}; seeds {SAMPLING_SEEDS}; cap {MAX_ITERATIONS}; "
        f"tolerance {TOL_PERCENT}%. Mask: top residual-RMS quantile {HIGH_RESIDUAL_QUANTILE}, "
        f"then scikit-image opening with disk radius {OPENING_RADIUS_PIXELS} pixel(s). "
        "The common budget is reduced to the smallest nonempty shortlist. Larger surviving "
        "regions contribute more pixels; no spectral/region balancing is imposed.", "",
        f"Baseline selection status: **{masks['reason']}**. Unavailable strategies: "
        f"{', '.join(sorted(unavailable)) or 'none'}. Relative residuals at/below "
        f"{RELATIVE_RESIDUAL_FLOOR:g} are too small for this diagnostic; no residual-guided "
        "selection or spatial-correlation interpretation is made there.", "",
        "| Strategy | Seed | Added | Relative fit | Mean RMS | Neighbor correlation | Stop |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        report.append(f"| {row['strategy']} | {row['seed']} | {row['added_pixels']} | "
            f"{row['relative_l2']:.4g} | {row['mean_rms']:.4g} | {row['neighbor_correlation']:.3g} | {row['stop']} |")
    report.extend(["", "## Interpretation limits", "",
        "Raw neighbor correlation compares adjacent signed channel residuals after channel centering. "
        "It is descriptive, not a test for randomness; constant offsets are removed by centering. "
        "The RMS/signal correlation in CSV shows brightness association. Neither metric is "
        "shot-noise-normalized. In noisy counts, variance increases with signal; arbitrary confocal "
        "units need gain/offset/read-noise information before noise-scaled interpretation. "
        "PSF/resampling correlations and model mismatch can produce structure too.", "",
        "Opening can remove real small/thin features and preserve chance noise clusters. It imposes "
        "spatial support on candidate selection, not spatial constraints on ALS. Flat or smaller "
        "residuals, or stable spectra, cannot establish correct factor recovery. Capped fits may "
        "remain incompletely optimized. No strategy is automatically declared the winner.", "",
        "Figures use the first seed for maps and all available seeds for spectral bands (ranges, "
        "not confidence intervals). Residual-map scales are shared across strategies. The linear "
        "view preserves absolute size; the additional log view exposes weaker patterns and is "
        "not a noise-significance display. All residuals here are tiny expected-count mismatches, "
        "well below one count; whether the same patterns survive measurement noise is untested.", "",
        "[Selection masks](selection.png) · [Residual maps](residual_maps.png) · "
        "[Log residual maps](residual_maps_log.png) · "
        "[Anonymous spectra](spectra.png)"])
    (output_dir/"summary.md").write_text("\n".join(report)+"\n")
    return fits


def plot_case(image, wavelengths, fits, masks, hull, output_dir):
    """Human inspection of spatial support, sampled points and fitted signatures."""
    baseline = fits["hull_-1"]
    fig = Figure(figsize=(13, 8), dpi=130)
    axes = fig.subplots(2, 3)
    panels = ((image.sum(axis=-1), "Observed total signal"),
              (baseline["maps"].rms, "Hull-only residual RMS"),
              (masks["high"], "High-residual threshold mask"),
              (masks["spatial"], "Mask after binary opening"),
              (masks["labels"], "Surviving connected regions"),
              (image.sum(axis=-1), "Spatial additions, first seed"))
    for ax, (panel, title) in zip(axes.flat, panels):
        handle = ax.imshow(panel, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(handle, ax=ax, shrink=.75)
    first = fits.get(f"spatial_residual_{SAMPLING_SEEDS[0]}")
    if first:
        added = np.setdiff1d(first["indices"], hull)
        y,x = np.unravel_index(added, image.shape[:2])
        axes[1,2].scatter(x,y,s=7,c="red",label=f"{len(added)} additions")
        axes[1,2].legend(fontsize=8)
    else:
        axes[1,2].text(.5,.5,"Spatial strategy unavailable",transform=axes[1,2].transAxes,ha="center",color="white")
    fig.tight_layout()
    fig.savefig(output_dir/"selection.png")
    methods = ("hull", "uniform", "high_residual", "spatial_residual")
    shown = [baseline if m=="hull" else fits.get(f"{m}_{SAMPLING_SEEDS[0]}") for m in methods]
    fig = Figure(figsize=(15, 7), dpi=130)
    axes = fig.subplots(2,4)
    for row, (attribute, label) in enumerate((("rms", "RMS (input units)"),("relative_l2", "Relative L2"))):
        vmax = max(float(np.nanmax(getattr(f["maps"],attribute))) for f in shown if f)
        for col, fitted in enumerate(shown):
            ax = axes[row,col]
            ax.set_title(methods[col]+" · "+label)
            ax.axis("off")
            if fitted:
                handle=ax.imshow(getattr(fitted["maps"],attribute),vmin=0,vmax=max(vmax,np.finfo(float).tiny),cmap="magma")
                fig.colorbar(handle,ax=ax,shrink=.75)
            else:
                ax.text(.5,.5,"Unavailable",transform=ax.transAxes,ha="center")
    fig.tight_layout()
    fig.savefig(output_dir/"residual_maps.png")
    fig = Figure(figsize=(15,7),dpi=130)
    axes=fig.subplots(2,4)
    for row,(attribute,label) in enumerate((("rms","RMS (input units)"),("relative_l2","Relative L2"))):
        values=np.concatenate([getattr(f["maps"],attribute).ravel() for f in shown if f])
        positive=values[np.isfinite(values)&(values>0)]
        vmax=float(positive.max()) if positive.size else 1.
        vmin=max(float(positive.min()),vmax*1e-10) if positive.size else 1e-10
        norm=LogNorm(vmin=vmin,vmax=max(vmax,vmin*10))
        for col,fitted in enumerate(shown):
            ax=axes[row,col]
            ax.set_title(methods[col]+" · "+label)
            ax.axis("off")
            if fitted:
                values=getattr(fitted["maps"],attribute)
                handle=ax.imshow(np.ma.masked_less_equal(values,0),norm=norm,cmap="magma")
                fig.colorbar(handle,ax=ax,shrink=.75)
            else:
                ax.text(.5,.5,"Unavailable",transform=ax.transAxes,ha="center")
    fig.suptitle("Shared log scales · noiseless pilot · not noise significance")
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output_dir/"residual_maps_log.png")
    fig = Figure(figsize=(15,4),dpi=130)
    axes = fig.subplots(1,4)
    for k, ax in enumerate(axes):
        ax.plot(wavelengths, baseline["spectra"][k],label="Hull fit",color="k")
        for method in methods[1:]:
            spectra=[f["spectra"][k] for f in fits.values() if f["row"]["strategy"]==method]
            if spectra:
                spectra=np.stack(spectra)
                line=ax.plot(wavelengths,np.median(spectra,axis=0),label=method)[0]
                ax.fill_between(wavelengths,spectra.min(axis=0),spectra.max(axis=0),alpha=.15,color=line.get_color())
        ax.set_title(f"Anonymous component {k+1}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction per channel")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir/"spectra.png")


@record_evaluation("output_dir")
def main(output_dir=OUTPUT_DIR):
    """Generate the scene, run blind analysis, then perform a separate truth audit."""
    output_dir=Path(output_dir)
    detector=SpectralDetector.from_profile("980_32ch")
    true_spectra=scene.detector_spectra(detector)
    grid=SamplingGrid(scene.IMAGE_SHAPE,pixel_size_um=scene.PIXEL_SIZE_UM)
    save_simulation_settings(output_dir,detector,true_spectra)
    for blur in BLUR_FWHM_UM:
        psf=None if blur==0 else GaussianPSF({name:blur for name in scene.COMPONENTS})
        spatial=render_scene(scene.build_scene(),grid,psf,oversample=3)
        true_counts=spatial.blurred_counts.reshape(-1,4)
        image=(true_counts @ true_spectra).reshape(*scene.IMAGE_SHAPE,detector.n_channels)
        folder=output_dir/f"blur_{blur:.2f}um"
        fits=analyze_image(image,detector.channel_centers,folder)
        np.savez_compressed(folder/"simulation_truth.npz",counts=true_counts,spectra=true_spectra)
        # Truth is first introduced AFTER the observed-data-only workflow.
        # This audit checks whether apparent improvements correspond to recovery.
        truth_rows=[]
        for fitted in fits.values():
            audit=evaluate_factors(image.reshape(-1,detector.n_channels),true_counts,true_spectra,
                                   fitted["counts"],fitted["spectra"])
            for k,name in enumerate(scene.COMPONENTS):
                truth_rows.append({"strategy":fitted["row"]["strategy"],"seed":fitted["row"]["seed"],
                    "component":name,"spectral_error":audit.spectral_relative_l2_per_component[k],
                    "count_rmse":audit.abundance_rmse_per_component[k]})
        expansion._write_csv(folder/"simulation_truth_audit.csv",truth_rows)
        append_truth_report(folder,truth_rows)
    print(f"Saved bounded residual-pooling pilot to {output_dir}")


def save_simulation_settings(output_dir, detector, true_spectra):
    """Keep generating truth/config separate from observed-data analysis settings."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    settings={"image_shape":scene.IMAGE_SHAPE,"pixel_size_um":scene.PIXEL_SIZE_UM,
        "blur_fwhm_um":BLUR_FWHM_UM,"oversample":3,"noise":"disabled",
        "components":scene.COMPONENTS,"spoke_counts":scene.SPOKE_COUNTS,
        "background_counts":scene.BACKGROUND_COUNTS,"spoke_seed":scene.SPOKE_SEED,
        "n_spokes":scene.N_SPOKES,"spoke_length_um":scene.SPOKE_LENGTH_UM,
        "spoke_width_um":scene.SPOKE_WIDTH_UM,"background_center_nm":scene.BACKGROUND_CENTER_NM,
        "background_sigma_nm":scene.BACKGROUND_SIGMA_NM,
        "true_unit_sum_spectra":true_spectra.tolist(),
        "wavelengths_nm":detector.channel_centers.tolist()}
    (output_dir/"simulation_settings.json").write_text(json.dumps(settings,indent=2)+"\n")


def append_truth_report(folder, rows):
    """Report optional simulation validation separately from observable diagnostics."""
    report=["", "## Separate simulation-only audit", "",
        "`simulation_truth_audit.csv` compares factors with known simulated spectra/counts. "
        "It is generated after analysis and never informs pixel selection or data-only metrics. "
        "These recovery errors are unavailable for ordinary real images. The table uses medians "
        "over available seeds, not uncertainty intervals.", "",
        "| Strategy | Component | Spectral shape error | Count RMSE |",
        "| --- | --- | --- | --- |"]
    for strategy in dict.fromkeys(r["strategy"] for r in rows):
        for component in dict.fromkeys(r["component"] for r in rows):
            group=[r for r in rows if r["strategy"]==strategy and r["component"]==component]
            shape=np.median([float(r["spectral_error"]) for r in group])
            counts=np.median([float(r["count_rmse"]) for r in group])
            report.append(f"| {strategy} | {component} | {shape:.4g} | {counts:.4g} |")
    with (Path(folder)/"summary.md").open("a") as handle:
        handle.write("\n".join(report)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    main(parser.parse_args().output_dir)
