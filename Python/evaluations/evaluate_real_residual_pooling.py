"""Inspect hull versus spatial residual additions on the minimal notebook image.

Run from Python: python -m evaluations.evaluate_real_residual_pooling
The default image is the public high-SNR COS-7 CZI. No reference spectra or ground truth guide
selection. This is a single exploratory refit, not validation of true recovery.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from lumic import (essential_spectral_pixles, mcr_als, positive_least_squares,
                   unique_profiles)
from lumic.evaluation import align_factors, compare_spectra, reconstruction_error_maps
from lumic.data import download_cos7_data
from lumic.io import load_czi
from lumic.utils import RGB_from_spec
from ._residual_pooling import candidate_masks


# Match the notebook's preprocessing, blind selection and ALS settings.
OUTPUT = Path(__file__).resolve().parent / "results/real_residual_pooling"
COMPONENTS = 4
LOW_INTENSITY_QUANTILE = .2
UNIQUE_PERCENT = .5
MAX_ITERATIONS = 500
TOL_PERCENT = .01
ADDED_BUDGET = 2048
RESIDUAL_QUANTILE = .8
OPENING_RADIUS = 1
SEED = 2026
PROJECTION_BATCH = 65536


def fit_and_project(pixels, image_shape, starts, indices, label, folder, *,
                    max_iterations=MAX_ITERATIONS, tol_percent=TOL_PERCENT,
                    norm="L2", projection_batch=PROJECTION_BATCH):
    """Fit one pool, then measure reconstruction across every original pixel.

    Projection is chunked so large images do not need multiple full spectral
    residual arrays in memory. Public reconstruction metrics are used in each
    chunk. Factors are matched to the common blind starts and given unit sum,
    preserving their product. Maps remain in raw detector units, not photons.
    """
    began = perf_counter()
    with tqdm(total=max_iterations, desc=label, unit="iter") as progress:
        _, spectra, history, info = mcr_als(
            pixels[indices], starts.copy(), nIter=max_iterations, normL=norm,
            tol_percent=tol_percent, solver="auto", return_diagnostics=True,
            iteration_callback=lambda _: progress.update(1))
    # Establish common ordering/scaling before projecting all pixels.
    n_components = len(starts)
    aligned = align_factors(starts, np.zeros((1, n_components)), spectra)
    spectra = aligned.spectra
    abundance = np.empty((len(pixels), n_components))
    rms, relative = np.empty(len(pixels)), np.empty(len(pixels))
    squared_residual, squared_signal = 0., 0.
    for first in tqdm(range(0, len(pixels), projection_batch), desc=label+" / full image"):
        last = min(first+projection_batch, len(pixels))
        observed = pixels[first:last]
        abundance[first:last] = positive_least_squares(
            observed, spectra, solver="auto", batch_size=8192)
        if not np.isfinite(abundance[first:last]).all():
            raise RuntimeError("Unresolved projection pixels; inspect solver warnings before comparing maps")
        maps = reconstruction_error_maps(observed, abundance[first:last] @ spectra)
        rms[first:last], relative[first:last] = maps.rms, maps.relative_l2
        squared_residual += float(np.sum(maps.residuals**2))
        squared_signal += float(np.sum(observed.astype(float)**2))
    rms, relative = rms.reshape(image_shape), relative.reshape(image_shape)
    row = dict(strategy=label, fit_pixels=len(indices), iterations=info.iterations_completed,
               best_iteration=info.best_iteration, stop=info.stop_reason,
               relative_l2=float(np.sqrt(squared_residual/squared_signal)),
               mean_rms=float(rms.mean()), median_rms=float(np.median(rms)),
               p95_rms=float(np.quantile(rms,.95)),
               seconds_including_projection=perf_counter()-began)
    np.savez_compressed(folder/(label+".npz"), spectra=spectra,
                        abundances=abundance.reshape(*image_shape, n_components),
                        rms=rms, relative_l2=relative, history=history, indices=indices)
    return dict(spectra=spectra, rms=rms, relative=relative, history=history, row=row)


def plot_comparison(rgb, wave, baseline, expanded, masks, added, folder):
    """Use shared display limits and disclose percentile clipping explicitly.

    Relative residual maps at very faint pixels can be unstable. No noise
    normalization is applied, so spatial patterns and improvements do not prove
    better spectra, concentrations, or statistically significant model failure.
    """
    fig = Figure(figsize=(15, 9), dpi=140)
    axes = fig.subplots(2,3)
    axes[0,0].imshow(rgb)
    axes[0,0].set_title("Observed RGB (existing RGB_from_spec helper)")
    axes[0,1].imshow(masks["high"], cmap="gray")
    axes[0,1].set_title("Top 20% residual RMS, before opening")
    axes[0,2].imshow(masks["spatial"], cmap="gray")
    y,x = np.unravel_index(added, baseline["rms"].shape)
    axes[0,2].scatter(x,y,s=1,c="red")
    axes[0,2].set_title(f"After opening; {len(added)} sampled pixels")
    vmax = max(np.quantile(f["rms"],.995) for f in (baseline,expanded))
    for ax, f, title in ((axes[1,0],baseline,"Hull-only RMS"),
                         (axes[1,1],expanded,"Hull + spatial additions RMS")):
        h=ax.imshow(f["rms"],vmin=0,vmax=vmax,cmap="magma")
        ax.set_title(title+" (shared 99.5% limits)")
        fig.colorbar(h,ax=ax,shrink=.8,label="Detector intensity units")
    delta=baseline["rms"]-expanded["rms"]
    limit=max(float(np.quantile(np.abs(delta),.995)),np.finfo(float).tiny)
    h=axes[1,2].imshow(delta,cmap="RdBu",vmin=-limit,vmax=limit)
    axes[1,2].set_title("RMS reduction: blue improves, red worsens")
    fig.colorbar(h,ax=axes[1,2],shrink=.8,label="Shared image; not independent validation")
    for ax in axes.flat: ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder/"comparison.png")

    fig=Figure(figsize=(12,9),dpi=140)
    axes=fig.subplots(2,2)
    for row,(field,label) in enumerate((("rms","RMS"),("relative","Relative L2"))):
        vmax=max(float(np.nanmax(f[field])) for f in (baseline,expanded))
        for col,f in enumerate((baseline,expanded)):
            h=axes[row,col].imshow(f[field],vmin=0,vmax=vmax,cmap="magma")
            axes[row,col].set_title(("Hull only" if col==0 else "Hull + spatial additions")+" / "+label)
            axes[row,col].axis("off")
            fig.colorbar(h,ax=axes[row,col],shrink=.8)
    fig.suptitle("Shared full-range limits; no percentile clipping")
    fig.tight_layout()
    fig.savefig(folder/"residual_maps_full_range.png")

    fig=Figure(figsize=(13,8),dpi=140)
    axes=fig.subplots(2,3)
    for k,ax in enumerate(axes.flat[:COMPONENTS]):
        ax.plot(wave,baseline["spectra"][k],label="Hull only")
        ax.plot(wave,expanded["spectra"][k],label="Hull + spatial additions")
        ax.set(title=f"Anonymous component {k+1}",xlabel="Wavelength (nm)",ylabel="Fraction per channel")
        ax.legend()
    for f,label in ((baseline,"Hull only"),(expanded,"Hull + spatial additions")):
        axes.flat[4].plot(np.arange(1,len(f["history"])+1),f["history"],label=label)
    axes.flat[4].set(title="ALS fitting-pool histories",xlabel="Iteration",ylabel="Lack of fit (%)")
    axes.flat[4].legend()
    axes.flat[5].axis("off")
    axes.flat[5].text(0,.9,"Different pools: histories are not\ndirectly comparable image scores.\n\nSame blind starts; four components;\n500-iteration cap; no ground truth.",va="top")
    fig.tight_layout()
    fig.savefig(folder/"spectra_and_convergence.png")


def run(input_path=None, output_dir=OUTPUT):
    """Use real image diagnostics for one spatially supported refit only.

    Low-intensity filtering matches the notebook and applies to hull selection
    and additional candidates, not image reporting. Opening is applied before
    excluding hull points to preserve their spatial support. Spatial sampling
    is uniform among surviving eligible pixels, so larger regions weigh more.
    The same full image supplies selection and reporting; no ground truth or
    independent accuracy claim is available. Brightness-dependent noise may
    dominate raw RMS; detector calibration is needed for noise significance.
    """
    if input_path is None:
        input_path, = download_cos7_data(snr="high", progress=True)
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)
    image=load_czi(input_path)  # First T/Z plane; fitting and maps use CYX.
    data=image.data
    wave=image.wavelengths_nm
    if wave is None:
        wave=np.arange(data.shape[0])  # Detector-channel indices for diagnostics.
    print(f"Loaded {input_path}: {data.shape}",flush=True)
    pixels=image.unfold().data
    median=np.median(pixels,axis=1)
    threshold=float(np.quantile(median,LOW_INTENSITY_QUANTILE))
    retained=np.flatnonzero((median>=threshold)&(pixels.sum(axis=1)>0))
    np.random.seed(SEED)
    candidates,local_hull=essential_spectral_pixles(pixels[retained],npc=COMPONENTS)
    hull=np.unique(retained[local_hull])
    starts,_=unique_profiles(candidates.T,COMPONENTS,UNIQUE_PERCENT)
    print(f"Retained {len(retained)} pixels; hull {len(hull)}",flush=True)
    baseline=fit_and_project(pixels,data.shape[1:],starts,hull,"hull",folder)
    eligible=np.zeros(len(pixels),dtype=bool)
    eligible[retained]=True
    eligible[hull]=False
    eligible=eligible.reshape(data.shape[1:])
    masks=candidate_masks(baseline["rms"],eligible,quantile=RESIDUAL_QUANTILE,
                          opening_radius=OPENING_RADIUS,
                          informative=baseline["row"]["relative_l2"]>1e-10)
    shortlist=np.flatnonzero(masks["spatial"]&eligible)
    if not len(shortlist):
        raise RuntimeError("No spatially supported residual candidates; hull results are saved")
    added=np.sort(np.random.default_rng(SEED).choice(shortlist,min(ADDED_BUDGET,len(shortlist)),replace=False))
    indices=np.union1d(hull,added)
    print(f"Spatial shortlist {len(shortlist)}; adding {len(added)} pixels",flush=True)
    expanded=fit_and_project(pixels,data.shape[1:],starts,indices,"spatial_residual",folder)
    np.savez_compressed(folder/"selection.npz",blind_starts=starts,hull_indices=hull,
                        added_indices=added,eligible=eligible,high_mask=masks["high"],
                        opened_mask=masks["spatial"],region_labels=masks["labels"],wavelengths_nm=wave)
    settings=dict(input_path=str(Path(input_path).resolve()),image_shape=data.shape,
                  components=COMPONENTS,median_quantile=LOW_INTENSITY_QUANTILE,
                  median_threshold=threshold,retained_pixels=len(retained),seed=SEED,
                  residual_quantile=RESIDUAL_QUANTILE,residual_threshold=masks["threshold"],
                  opening_radius=OPENING_RADIUS,spatial_candidates=len(shortlist),
                  requested_additions=ADDED_BUDGET,actual_additions=len(added),
                  max_iterations=MAX_ITERATIONS,tol_percent=TOL_PERCENT,norm="L2",
                  solver="auto",noise_normalization="none; raw detector units")
    (folder/"settings.json").write_text(json.dumps(settings,indent=2)+"\n")
    plot_comparison(RGB_from_spec(data,do_norm=True),wave,baseline,expanded,masks,added,folder)
    drift=compare_spectra(baseline["spectra"],expanded["spectra"],match=False)
    report=["# Real-data spatial residual-pool comparison","",
            "Same CZI as minimal_example.ipynb; blind selection, no manually selected reference spectra. "
            "Four components are a declared prior. Filtering and ALS settings match the notebook; "
            "the seeded SVD/hull can differ from a previous unseeded notebook run.","",
            f"Hull: {len(hull)} pixels; spatial additions: {len(added)} from {len(shortlist)} eligible candidates. "
            "Both fits restart the same spectra. Opening uses a one-pixel-radius disk (3x3 cross).","",
            "| Pool | Iterations | Stop | Full-image relative L2 | Mean RMS | P95 RMS |",
            "| --- | --- | --- | --- | --- | --- |"]
    for f in (baseline,expanded):
        r=f["row"]
        report.append(f"| {r['strategy']} | {r['iterations']} | {r['stop']} | {r['relative_l2']:.6g} | {r['mean_rms']:.6g} | {r['p95_rms']:.6g} |")
    report.extend(["",f"Fraction of image pixels with smaller RMS: {np.mean(expanded['rms']<baseline['rms']):.2%}.","",
                   "Unit-L2 spectral shape changes relative to the hull fit (anonymous component order): "
                   +", ".join(f"{v:.4g}" for v in drift.shape_distance)+".","",
                   "[Inspection comparison](comparison.png) · [Unclipped residual maps](residual_maps_full_range.png) · "
                   "[Spectra and iteration histories](spectra_and_convergence.png)","",
                   "## Interpretation limits","",
                   "These are raw detector-intensity residuals, not shot-noise-normalized significance. "
                   "A smaller or flatter residual does not establish correct spectra or concentrations. "
                   "The same pixels informed sampling and reporting, so this is exploratory rather than independent validation. "
                   "Opening can remove true thin features and retain chance clusters; it does not regularize ALS spatially. "
                   "Both maps include low-intensity pixels excluded from pool selection. Relative errors there can be unstable. "
                   "The inspection figure clips at shared 99.5th-percentile display limits, while the second figure uses full ranges. "
                   "NPZ files retain full-resolution maps, spectra, abundances, histories and exact selected indices."])
    (folder/"summary.md").write_text("\n".join(report)+"\n")
    (folder/"observed_metrics.json").write_text(json.dumps([baseline["row"],expanded["row"]],indent=2)+"\n")
    print("\n".join(report),flush=True)
    return baseline,expanded


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local CZI path. Defaults to the public high-SNR COS-7 acquisition.")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.input,args.output_dir)
