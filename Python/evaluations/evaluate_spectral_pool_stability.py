"""Data-only spectral drift as a nested fitting pool grows (four components).

The entry point generates the existing noiseless scene. run_trajectory accepts
only measured pixels and wavelengths; truth never enters matching or metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from lumic import mcr_als, positive_least_squares
from lumic.evaluation import align_factors, compare_spectra, reconstruction_error_maps
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene
from . import evaluate_background_recovery as diagnostic
from . import evaluate_fitting_pool_expansion as expansion
from . import evaluate_radial_spoke_blur as scene


ADDED_BUDGETS = (64, 256, 512, 1024, 2048)
SAMPLING_SEED = 2026
MAX_ITERATIONS = 500


def run_trajectory(pixels, wavelengths, output_dir):
    """Track pool-dependent changes, not ground-truth recovery or uncertainty.

    Inputs are (pixels, channels) and channel wavelengths. Components are
    assigned to the fixed blind starts using unit-L2 spectral shape matching,
    then given unit-sum spectra with paired abundances rescaled. Anonymous
    IDs refer to these starting rows, not biological identities. Every ALS
    run restarts the same starts; a capped result may be incompletely optimized.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    starts, hull, anchors = diagnostic.select_starts(pixels)
    pools = [("hull", 0, np.unique(hull))]
    pools.extend((f"plus_{budget}", budget, indices) for budget, indices in
                 expansion.expanded_pools(len(pixels), hull, ADDED_BUDGETS, SAMPLING_SEED).items())
    pools.append(("full_image", len(pixels)-len(np.unique(hull)), np.arange(len(pixels))))
    results, selections, arrays = [], [], {}
    for name, budget, indices in pools:
        with tqdm(total=MAX_ITERATIONS, desc=f"Spectral stability / {name}", unit="iter", leave=False) as progress:
            _, spectra, history, diagnostics = mcr_als(pixels[indices], starts.copy(),
                nIter=MAX_ITERATIONS, normL="L1", tol_percent=expansion.TOL_PERCENT,
                solver=expansion.SOLVER, return_diagnostics=True,
                iteration_callback=lambda _: progress.update(1))
        counts = positive_least_squares(pixels, spectra, solver=expansion.SOLVER)
        # Compare anonymous components to a fixed data-derived reference.
        # Normalization transfers scale to C, preserving the reconstructed image.
        aligned = align_factors(starts, counts, spectra)
        counts, spectra = aligned.abundances, aligned.spectra
        order, matching_costs = aligned.component_order, aligned.matching_costs
        residuals = reconstruction_error_maps(pixels, counts @ spectra)
        results.append({"pool": name, "added_budget": budget, "fit_pixels": len(indices),
            "iterations": diagnostics.iterations_completed, "best_iteration": diagnostics.best_iteration,
            "status": diagnostics.stop_reason,
            "reconstruction_relative_l2": float(np.linalg.norm(residuals.residuals)/np.linalg.norm(pixels)),
            "spectra": spectra, "matching_costs": matching_costs, "component_order": order})
        selections.append({"pool": name, "fit_indices": indices.tolist(), "history": history.tolist()})
        arrays[name+"_spectra"] = spectra
        arrays[name+"_abundances"] = counts
    rows = []
    hull_spectra, full_spectra = results[0]["spectra"], results[-1]["spectra"]
    previous = hull_spectra
    for result in results:
        spectra = result["spectra"]
        # All runs already share the blind-start order. Do not rematch pairs:
        # doing so could switch anonymous component IDs along the trajectory.
        hull_change = compare_spectra(hull_spectra, spectra, match=False)
        previous_change = compare_spectra(previous, spectra, match=False)
        full_change = compare_spectra(full_spectra, spectra, match=False)
        for k in range(len(starts)):
            rows.append({key: value for key, value in result.items()
                         if key not in ("spectra", "matching_costs", "component_order")} | {
                "component_id": k+1, "matched_estimated_row": int(result["component_order"][k]),
                "matching_cost_to_start": result["matching_costs"][k],
                "shape_change_from_hull": hull_change.shape_distance[k],
                "shape_change_from_previous": previous_change.shape_distance[k],
                "shape_difference_from_full": full_change.shape_distance[k],
                "max_channel_change_from_hull": hull_change.max_channel_change[k],
                "max_channel_change_from_previous": previous_change.max_channel_change[k]})
        previous = spectra
    expansion._write_csv(output_dir/"trajectory.csv", rows)
    arrays["starting_spectra"] = starts
    np.savez_compressed(output_dir/"aligned_factors.npz", **arrays)
    (output_dir/"settings.json").write_text(json.dumps({"max_iterations": MAX_ITERATIONS,
        "tol_percent": expansion.TOL_PERCENT, "solver": expansion.SOLVER, "norm": "L1",
        "sampling_seed": SAMPLING_SEED, "added_budgets": ADDED_BUDGETS,
        "starting_spectra": starts.tolist(), "selected_start_indices": anchors.tolist(),
        "channel_centers_nm": np.asarray(wavelengths).tolist(), "selections": selections}, indent=2)+"\n")
    fig = Figure(figsize=(16, 9), dpi=130)
    axes = fig.subplots(2, 4)
    for k in range(4):
        ax = axes[0,k]
        ax.plot(wavelengths, starts[k], "k:", label="Blind start")
        for result in results:
            ax.plot(wavelengths, result["spectra"][k], label=result["pool"].replace("plus_", "Hull + "))
        ax.set_title(f"Anonymous component {k+1}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction per channel")
        ax.legend(fontsize=6)
        entries = [r for r in rows if r["component_id"]==k+1]
        ax = axes[1,k]
        for metric, label in (("shape_change_from_hull", "From hull-only fit"),
                ("shape_change_from_previous", "From previous pool"),
                ("shape_difference_from_full", "From full-image fit")):
            ax.semilogx([r["fit_pixels"] for r in entries], [r[metric] for r in entries], "o-", label=label)
        ax.set_xlabel("Total fitting pixels (log scale)")
        ax.set_ylabel("Spectral shape difference (unit L2)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Data-driven pool stability · fixed starts · one nested sampling sequence · 500-update cap")
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output_dir/"spectral_drift.png")
    report = ["# Spectral sensitivity to fitting-pool expansion", "",
        "No ground-truth spectra or abundances enter analysis. Anonymous components are aligned "
        "to fixed blind starting spectra. All fits restart those same starts; additions are nested, "
        "uniform samples without replacement, using seed 2026. All hull pixels are retained.", "",
        "[Spectral trajectory](spectral_drift.png)", "",
        "This identifies signatures whose estimates depend strongly on fitting-pool coverage. "
        "It does not identify which estimate is correct, establish biological identity, or provide "
        "confidence intervals. The full-image fit is an endpoint reference, not ground truth. "
        "Near-identical spectra can make component matching ambiguous. One trajectory is a pilot; "
        "robust conclusions need additional trajectories and convergence checks.", "",
        "500-update cap and early stopping at 0.1%. Capped-run drift can reflect incomplete "
        "optimization as well as pool sensitivity. Shape differences use unit-L2 normalization; "
        "largest-channel differences use unit-sum spectra. No noise significance is claimed.", "",
        "| Component ID | Hull-to-full shape change | Largest channel change |",
        "| --- | --- | --- |"]
    for r in rows:
        if r["pool"]=="full_image":
            report.append(f"| {r['component_id']} | {r['shape_change_from_hull']:.4g} | {r['max_channel_change_from_hull']:.4g} |")
    report.extend(["", "## Fit diagnostics", "",
                   "| Pool | Total pixels | Iterations | Stop | Full-image relative residual |",
                   "| --- | --- | --- | --- | --- |"])
    for result in results:
        report.append(f"| {result['pool']} | {result['fit_pixels']} | {result['iterations']} | "
                      f"{result['status']} | {result['reconstruction_relative_l2']:.4g} |")
    (output_dir/"summary.md").write_text("\n".join(report)+"\n")
    return rows


def main(output_dir):
    output_dir=Path(output_dir)
    detector=SpectralDetector.from_profile("980_32ch")
    spectra=scene.detector_spectra(detector)
    grid=SamplingGrid(scene.IMAGE_SHAPE,pixel_size_um=scene.PIXEL_SIZE_UM)
    for blur in expansion.BLUR_FWHM_UM:
        psf=None if blur==0 else GaussianPSF({name:blur for name in scene.COMPONENTS})
        spatial=render_scene(scene.build_scene(),grid,psf,oversample=3)
        pixels=spatial.blurred_counts.reshape(-1,4) @ spectra
        # Only observed data cross the boundary into the analysis function.
        run_trajectory(pixels,detector.channel_centers,output_dir/f"blur_{blur:.2f}um")
    print(f"Saved data-driven stability trajectories to {output_dir}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=expansion.OUTPUT_DIR/"data_driven_stability")
    main(parser.parse_args().output_dir)
