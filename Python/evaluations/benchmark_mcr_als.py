"""Compare ALS solvers on a fixed subset of the four-component spoke scene.

Run from Python with ``python -m evaluations.benchmark_mcr_als``.
Scene rendering and blind spectrum selection are outside the timed region.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from lumic import mcr_als, normalize_components
from evaluations import evaluate_radial_spoke_blur as spokes
from ._execution import record_evaluation


# Fixed iteration counts make the timing comparison fair: neither solver
# stops early. This is a short performance measurement, not a recovery sweep.
BLUR_FWHM_UM = 0.65
N_ITERATIONS = 30
REPEATS = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "mcr_als_performance"


def make_fit_inputs():
    """Use the same starts and seeded fitting subset as the spoke evaluation."""
    scene = spokes.build_scene()
    grid = spokes.SamplingGrid(spokes.IMAGE_SHAPE,
                               pixel_size_um=spokes.PIXEL_SIZE_UM)
    detector = spokes.SpectralDetector.from_profile("980_32ch")
    psf = spokes.GaussianPSF({name: BLUR_FWHM_UM for name in spokes.COMPONENTS})
    spatial = spokes.render_scene(scene, grid, psf, oversample=3)
    mixture = spokes.simulate_linear_mixture(
        spatial.blurred_counts, spokes.detector_spectra(detector),
        detector.channel_centers, fluorophore_names=spokes.COMPONENTS,
    )
    pixels = mixture.image.reshape(-1, detector.n_channels)
    state = np.random.get_state()
    np.random.seed(spokes.SPOKE_SEED)
    try:
        candidates, hull_indices = spokes.essential_spectral_pixles(
            pixels, npc=len(spokes.COMPONENTS)
        )
        starts, selected_indices = spokes.unique_profiles(
            candidates.T, nr=len(spokes.COMPONENTS),
            f=spokes.UNIQUE_PROFILES_F_PERCENT,
        )
    finally:
        np.random.set_state(state)
    chosen = np.random.default_rng(spokes.SPOKE_SEED).choice(
        len(pixels), min(spokes.MAX_FIT_PIXELS, len(pixels)), replace=False
    )
    selected_pixels = hull_indices[np.asarray(selected_indices, dtype=int)]
    chosen = np.unique(np.concatenate((chosen, selected_pixels)))
    return pixels[chosen], normalize_components(starts, do_norm="L1")


@record_evaluation("output_dir")
def benchmark(output_dir=OUTPUT_DIR):
    """Measure runtime and check factors/history against the original solver."""
    data, starts = make_fit_inputs()
    rows = []
    fits = {}
    for solver in ("nnls", "auto"):
        # Warm up imports and linear algebra before recording wall time.
        mcr_als(data, starts, nIter=2, normL="L1", solver=solver)
        elapsed = []
        for repeat in range(REPEATS):
            print(f"{solver}: repeat {repeat + 1}/{REPEATS}", flush=True)
            began = perf_counter()
            fit = mcr_als(data, starts, nIter=N_ITERATIONS, normL="L1",
                          solver=solver)
            elapsed.append(perf_counter() - began)
        fits[solver] = fit
        rows.append({
            "solver": solver,
            "pixels": len(data),
            "channels": data.shape[1],
            "components": len(starts),
            "iterations": N_ITERATIONS,
            "repeats": REPEATS,
            "median_seconds": float(np.median(elapsed)),
            "min_seconds": min(elapsed),
            "max_seconds": max(elapsed),
            "final_relative_l2": float(fit[2][-1]),
        })

    comparisons = []
    for name, original, accelerated in zip(
        ("concentrations", "spectra", "fit_history"), fits["nnls"], fits["auto"]
    ):
        # Performance gains are only meaningful when the numerical results
        # agree. This fixture has independent components; the solution is
        # sufficiently determined for comparing the factors themselves.
        np.testing.assert_allclose(accelerated, original, rtol=1e-8, atol=1e-10)
        comparisons.append((name, float(np.max(np.abs(accelerated - original)))))

    speedup = rows[0]["median_seconds"] / rows[1]["median_seconds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.md").open("w") as file:
        file.write("# MCR-ALS solver performance comparison\n\n")
        file.write(f"Fixed {BLUR_FWHM_UM:g} µm blur spoke scene: {len(data)} "
                   f"fitting pixels, {data.shape[1]} channels, {len(starts)} "
                   f"components. Both solvers run {N_ITERATIONS} updates with "
                   f"L1 normalization and no early stopping. Median wall time "
                   f"is measured over {REPEATS} "
                   f"{'repetition' if REPEATS == 1 else 'repetitions'} after warm-up. "
                   "Rendering, convex-hull selection, plotting, and full-image "
                   "projection are excluded. Timings depend on the machine "
                   "and installed libraries.\n\n")
        file.write("| Solver | Median seconds | Final relative L2 |\n"
                   "| --- | ---: | ---: |\n")
        for row in rows:
            file.write(f"| {row['solver']} | {row['median_seconds']:.4f} | "
                       f"{row['final_relative_l2']:.8g} |\n")
        file.write(f"\nMeasured speedup: **{speedup:.1f}×**.\n\n")
        file.write("| Compared output | Maximum absolute difference |\n"
                   "| --- | ---: |\n")
        for name, delta in comparisons:
            file.write(f"| {name} | {delta:.3g} |\n")
        file.write("\nBoth paths solve the same nonnegative least-squares "
                   "objective. The auto path enumerates component subsets "
                   "in batches for at most six well-conditioned components. "
                   "Larger or degenerate systems retain the original "
                   "NNLS/BVLS path. This benchmark does not establish a "
                   "speedup for those systems or measure an entire image "
                   "analysis pipeline.\n")
    print(f"Median ALS speedup: {speedup:.1f}×", flush=True)
    print(f"Saved comparison to {output_dir}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    benchmark(parser.parse_args().output_dir)
