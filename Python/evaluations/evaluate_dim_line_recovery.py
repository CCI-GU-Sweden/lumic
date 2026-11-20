"""Noiseless residual-guided recovery of a dim far-red line beside bright spokes.

Run from Python: python -m evaluations.evaluate_dim_line_recovery
First bounded validation: five true contributions, initial blind count four,
three fixed blur conditions. Truth enters only the separate audit/figures.
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
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from lumic import essential_spectral_pixles, unique_profiles
from lumic.evaluation import (compare_spectra, evaluate_factors, select_residual_pixels,
    residual_neighbor_correlation, residual_spectral_correlation)
from lumic.simulation import (Scene2D, Line, Rectangle, SamplingGrid, GaussianPSF,
    SpectralDetector, load_offline_emission_spectrum, render_scene)
from lumic.utils import RGB_from_spec, select_analysis_pixels
from ._residual_workflow import convex_hull_pool, fit_pool, inspect_fit_residuals, expand_pool


BRIGHT_NAMES = ("EBFP2", "EGFP", "mCherry")
BACKGROUND = "broad_500nm_background"
COMPONENTS = (*BRIGHT_NAMES, "mNeptune", BACKGROUND)
IMAGE_SHAPE = (64, 64)
PIXEL_SIZE_UM = .12
BRIGHT_COUNTS = 900.
DIM_COUNTS = 90.
BACKGROUND_COUNTS = 24.
LINE_START_UM = (1.20, 1.26)  # On a scan row: a thin no-blur line is not missed by sampling.
LINE_END_UM = (6.40, 1.26)
LINE_WIDTH_UM = .12
SPOKE_CENTER_UM = (3.84, 4.80)
SPOKE_LENGTH_UM = 2.1
SPOKE_WIDTH_UM = .30
N_SPOKES = 9
SEED = 2026
BLUR_FWHM_UM = (0., .24, .60)
INITIAL_COMPONENTS = 4
RESIDUAL_PCS = (1,)  # Fixed observed-data rule; do not choose using truth errors.
SAMPLING = dict(per_sign_budget=128, score_quantile=.90, opening_radius=1, seed=SEED)
ALS = dict(max_iterations=500, tol_percent=.01, norm="L2")
OUTPUT = Path(__file__).resolve().parent / "results/dim_line_recovery"


def build_scene():
    """Only one shape contains mNeptune; background covers the whole field."""
    shapes=[Rectangle((3.84,3.84),(30.,30.),{BACKGROUND:BACKGROUND_COUNTS})]
    labels=np.random.default_rng(SEED).permutation(np.arange(N_SPOKES)%3)
    for k,label in enumerate(labels):
        angle=2*np.pi*k/N_SPOKES
        endpoint=(SPOKE_CENTER_UM[0]+SPOKE_LENGTH_UM*np.cos(angle),
                  SPOKE_CENTER_UM[1]+SPOKE_LENGTH_UM*np.sin(angle))
        shapes.append(Line(SPOKE_CENTER_UM,endpoint,SPOKE_WIDTH_UM,
                           {BRIGHT_NAMES[label]:BRIGHT_COUNTS}))
    shapes.append(Line(LINE_START_UM,LINE_END_UM,LINE_WIDTH_UM,{"mNeptune":DIM_COUNTS}))
    return Scene2D(COMPONENTS,tuple(shapes))


def detector_spectra(detector):
    """Counts are specified AFTER sampling; normalize detector signatures to sum 1."""
    curves=[load_offline_emission_spectrum(name) for name in COMPONENTS[:-1]]
    wavelengths=np.arange(400.,721.)
    curves.append((wavelengths,np.exp(-.5*((wavelengths-500.)/105.)**2)))
    sampled=np.array([detector.sample(*curve) for curve in curves])
    return sampled/sampled.sum(axis=1,keepdims=True)


def analyze_observed(image, wave, folder):
    """Fit without access to true spectra/maps or target-component identity.

    Preserve all usable pixels as eligible (median_quantile=0): the first
    experiment isolates weak/mixed signatures, not a low-intensity cutoff.
    Start with four, select fixed residual PC1 on both signs with opening,
    then compare four/five on one expanded pool. A five-component hull-only
    control uses the SAME starts inferred from the expanded observed pool;
    therefore it tests fitting coverage, not a wholly independent hull pipeline.
    Failures and empty additions remain explicit; no scene tuning/fallback
    changes selection to ensure recovery of the desired species.
    """
    retained=select_analysis_pixels(image,median_quantile=0.)
    pool=convex_hull_pool(image,retained,npc=INITIAL_COMPONENTS,seed=SEED)
    fits={}
    fits["four_hull"]=fit_pool(image,pool,n_components=4,output_dir=folder/"four_hull",seed=SEED,**ALS)
    diagnostics=inspect_fit_residuals(image,fits["four_hull"],wave)
    if fits["four_hull"]["row"]["relative_l2"]<=1e-10:
        return fits,"baseline_below_residual_floor"
    selection=select_residual_pixels(diagnostics["scores"],pcs=RESIDUAL_PCS,
        eligible=pool["eligible"],existing_indices=pool["indices"],**SAMPLING)
    expanded=expand_pool(pool,diagnostics,selection=selection,output_dir=folder/"selection")
    if not len(selection.indices):
        return fits,"no_spatially_supported_additions"
    pixels=np.moveaxis(image,0,-1).reshape(-1,image.shape[0])
    state=np.random.get_state()
    try:
        np.random.seed(SEED)
        candidates,_=essential_spectral_pixles(pixels[expanded["indices"]],npc=5)
        five_starts,_=unique_profiles(candidates.T,5,.5)
    finally:np.random.set_state(state)
    original_settings=json.loads((fits["four_hull"]["folder"]/"settings.json").read_text())
    four_starts=np.array(original_settings["initial_spectra"])
    runs=(("four_expanded",expanded,4,four_starts),
          ("five_expanded",expanded,5,five_starts),
          ("five_hull_same_starts",pool,5,five_starts))
    failures={}
    for name,fitting_pool,count,starts in runs:
        try:
            fits[name]=fit_pool(image,fitting_pool,n_components=count,initial_spectra=starts,
                                output_dir=folder/name,seed=SEED,**ALS)
            inspect_fit_residuals(image,fits[name],wave)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            failures[name]=dict(type=type(error).__name__,message=str(error))
    (folder/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    return fits,"completed" if not failures else "partial_failure"


def summarize_fit_structure(image, fit):
    """Save observed-data correlations for this small simulation, without truth.

    The explicit relative floor suppresses correlations of near-zero arithmetic
    residuals. It is a reporting convention, not a calibrated noise threshold.
    """
    with np.load(fit["folder"]/"fit.npz") as saved:
        abundance=saved["abundances"].reshape(*image.shape[1:],fit["n_components"])
    residual=np.moveaxis(image,0,-1)-abundance@fit["spectra"]
    informative=fit["row"]["relative_l2"]>1e-10
    spectral=residual_spectral_correlation(residual) if informative else np.full(image.shape[1:],np.nan)
    neighbor=residual_neighbor_correlation(residual) if informative else np.nan
    valid=spectral[np.isfinite(spectral)]
    np.savez_compressed(fit["folder"]/"residual_correlations.npz",
        spectral_lag1_correlation=spectral,neighbor_correlation=neighbor,
        residual_above_floor=informative,relative_floor=1e-10)
    return dict(neighbor_correlation=float(neighbor),
                mean_spectral_correlation=float(valid.mean()) if valid.size else np.nan,
                residual_above_floor=informative)


def truth_assignment(true_spectra,spectra):
    """Post-fit rectangular unit-L2 matching; never used to pick starts/pixels."""
    a=true_spectra/np.linalg.norm(true_spectra,axis=1,keepdims=True)
    norms=np.linalg.norm(spectra,axis=1,keepdims=True)
    b=np.divide(spectra,norms,out=np.zeros_like(spectra),where=norms>0)
    cost=np.sum((a[:,None]-b[None])**2,axis=-1)
    rows,cols=linear_sum_assignment(cost)
    return rows,cols,cost


def audit_initial_starts(true_spectra,folder):
    """Record whether the desired identity was absent at selection, before ALS."""
    starts=np.asarray(json.loads((folder/"four_hull/settings.json").read_text())["initial_spectra"])
    rows,cols,cost=truth_assignment(true_spectra,starts)
    result=dict(unmatched_truth=[name for k,name in enumerate(COMPONENTS) if k not in rows],
        assignments=[dict(start_index=int(col),assigned_truth=COMPONENTS[row],
                          shape_distance=float(np.sqrt(cost[row,col]))) for row,col in zip(rows,cols)])
    (folder/"initial_start_truth_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def audit_fit(true_counts,true_spectra,fit):
    """Truth-only audit AFTER blind fitting, including under-specified results.

    Equal-count fits use the shared factor metrics. For four/five rectangular
    matching, assign each recovered spectrum to a distinct truth row by unit-L2
    distance; one row remains unmatched. This records the best assignment, not
    proof the omitted identity was uniquely determined. Unmatched predicted
    concentration is displayed as zero, not claimed to be recovered correctly.
    """
    with np.load(fit["folder"]/"fit.npz") as saved:counts=saved["abundances"].reshape(-1,fit["n_components"])
    truth=true_counts.reshape(-1,len(COMPONENTS))
    spectra=fit["spectra"]
    aligned_counts=np.zeros_like(truth);aligned_spectra=np.zeros_like(true_spectra)
    if len(spectra)==len(COMPONENTS):
        metrics=evaluate_factors(truth@true_spectra,truth,true_spectra,counts,spectra)
        aligned_counts=metrics.estimated_abundances
        aligned_spectra=metrics.estimated_spectra
        assigned=set(range(len(COMPONENTS)))
    else:
        rows,cols,_=truth_assignment(true_spectra,spectra)
        aligned_counts[:,rows]=counts[:,cols]
        aligned_spectra[rows]=spectra[cols]
        assigned=set(rows)
    comparison=compare_spectra(true_spectra,aligned_spectra,match=False)
    rows=[]
    for k,name in enumerate(COMPONENTS):
        rows.append(dict(component=name,assigned=k in assigned,
            shape_distance=float(comparison.shape_distance[k]) if k in assigned else None,
            max_channel_change=float(comparison.max_channel_change[k]) if k in assigned else None,
            count_rmse=float(np.sqrt(np.mean((aligned_counts[:,k]-truth[:,k])**2))),
            truth_peak_counts=float(truth[:,k].max())))
    return rows,aligned_counts.reshape(true_counts.shape),aligned_spectra


def plot_case(image,counts,spectra,wave,fits,audits,folder):
    """Separate observable residuals from simulation-only recovery displays."""
    fig=Figure(figsize=(14,8),dpi=130);axes=fig.subplots(2,3)
    for k,name in enumerate(COMPONENTS):
        h=axes.flat[k].imshow(counts[...,k],cmap="viridis")
        axes.flat[k].set_title("True "+name);fig.colorbar(h,ax=axes.flat[k],shrink=.8)
    axes.flat[5].imshow(RGB_from_spec(image,do_norm=True));axes.flat[5].set_title("Observed spectral RGB")
    for ax in axes.flat:ax.axis("off")
    fig.tight_layout();fig.savefig(folder/"scene.png")
    names=list(fits)
    fig=Figure(figsize=(4*len(names),7),dpi=130);axes=fig.subplots(2,len(names),squeeze=False)
    vmax=max(np.quantile(f["rms"],.995) for f in fits.values())
    for col,name in enumerate(names):
        fit=fits[name]
        h=axes[0,col].imshow(fit["rms"],cmap="magma",vmin=0,vmax=max(vmax,1e-12))
        axes[0,col].set_title(name+" / residual RMS");fig.colorbar(h,ax=axes[0,col],shrink=.8)
        axes[0,col].axis("off")
        with np.load(fit["folder"]/"residual_diagnostics.npz") as diagnostic:
            score=diagnostic["own_scores"][...,0]
        limit=max(float(np.quantile(np.abs(score),.995)),1e-12)
        h=axes[1,col].imshow(score,cmap="RdBu_r",vmin=-limit,vmax=limit)
        axes[1,col].set_title("Fresh residual PC1 / independent basis & scale")
        axes[1,col].axis("off");fig.colorbar(h,ax=axes[1,col],shrink=.8)
    fig.tight_layout();fig.savefig(folder/"observable_comparison.png")
    fig=Figure(figsize=(4*len(names),12),dpi=130);axes=fig.subplots(3,len(names),squeeze=False)
    # Shared limits prevent separately stretched maps from hiding count errors.
    maximum=max(float(counts[...,3].max()),
                *(float(audits[name][1][...,3].max()) for name in names),1e-12)
    error_limit=max(*(float(np.max(np.abs(audits[name][1][...,3]-counts[...,3])))
                      for name in names),1e-12)
    for col,name in enumerate(names):
        _,recovered,recovered_spectra=audits[name]
        axes[0,col].plot(wave,spectra[3],"k",label="True mNeptune")
        axes[0,col].plot(wave,recovered_spectra[3],label="Matched recovered (zero if unmatched)")
        axes[0,col].set(title=name,xlabel="Wavelength (nm)",ylabel="Fraction per channel")
        axes[0,col].legend(fontsize=7)
        h=axes[1,col].imshow(recovered[...,3],vmin=0,vmax=maximum,cmap="viridis")
        axes[1,col].set_title("Truth-audit matched mNeptune contribution")
        fig.colorbar(h,ax=axes[1,col],shrink=.8)
        difference=recovered[...,3]-counts[...,3]
        h=axes[2,col].imshow(difference,cmap="RdBu_r",vmin=-error_limit,vmax=error_limit)
        axes[2,col].set_title("Contribution error / truth required")
        fig.colorbar(h,ax=axes[2,col],shrink=.8)
        for ax in axes[1:,col]:ax.axis("off")
    fig.tight_layout();fig.savefig(folder/"mneptune_truth_audit.png")


def run(output=OUTPUT, *, blur_fwhm_um=BLUR_FWHM_UM):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not blur_fwhm_um or not np.isfinite(blur_fwhm_um).all() or any(value<0 for value in blur_fwhm_um):
        raise ValueError("blur widths must be a nonempty sequence of finite nonnegative values")
    detector=SpectralDetector.from_profile("980_32ch")
    spectra=detector_spectra(detector);wave=detector.channel_centers
    settings=dict(components=COMPONENTS,image_shape=IMAGE_SHAPE,pixel_size_um=PIXEL_SIZE_UM,
        bright_counts=BRIGHT_COUNTS,dim_counts=DIM_COUNTS,background_counts=BACKGROUND_COUNTS,
        line_start_um=LINE_START_UM,line_end_um=LINE_END_UM,line_width_um=LINE_WIDTH_UM,
        spoke_center_um=SPOKE_CENTER_UM,spoke_length_um=SPOKE_LENGTH_UM,spoke_width_um=SPOKE_WIDTH_UM,
        n_spokes=N_SPOKES,seed=SEED,blur_fwhm_um=blur_fwhm_um,initial_components=4,
        residual_pcs=RESIDUAL_PCS,sampling=SAMPLING,als=ALS,detector=detector.name,
        median_quantile=0,noise="none",counts="expected detected counts after binning; signatures unit sum")
    settings["versions"]=dict(python=platform.python_version(),numpy=np.__version__,
                              scipy=scipy.__version__,sklearn=sklearn.__version__)
    (output/"settings.json").write_text(json.dumps(settings,indent=2)+"\n")
    summary=[]
    for blur in blur_fwhm_um:
        folder=output/f"blur_{blur:.2f}um";folder.mkdir(exist_ok=True)
        print(f"\nCondition FWHM={blur:g} um / width={LINE_WIDTH_UM:g} um",flush=True)
        psf=None if blur==0 else GaussianPSF({name:blur for name in COMPONENTS})
        spatial=render_scene(build_scene(),SamplingGrid(IMAGE_SHAPE,PIXEL_SIZE_UM),psf,oversample=7)
        counts=spatial.blurred_counts
        image=np.moveaxis(counts@spectra,-1,0)
        np.savez_compressed(folder/"simulation_truth.npz",counts=counts,spectra=spectra,wavelengths_nm=wave)
        try:
            fits,status=analyze_observed(image,wave,folder)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            (folder/"analysis_failure.json").write_text(json.dumps(dict(type=type(error).__name__,message=str(error)),indent=2)+"\n")
            summary.append(dict(blur_fwhm_um=blur,condition_status="failed",message=str(error)))
            continue
        initial_audit=audit_initial_starts(spectra,folder)
        audits={name:audit_fit(counts,spectra,fit) for name,fit in fits.items()}
        for name,fit in fits.items():
            rows,recovered,recovered_spectra=audits[name]
            with (fit["folder"]/"truth_audit.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            np.savez_compressed(fit["folder"]/"matched_truth_audit.npz",counts=recovered,spectra=recovered_spectra)
            neptune=rows[3]
            observed_row={**fit["row"],**summarize_fit_structure(image,fit),"strategy":name}
            summary.append(dict(blur_fwhm_um=blur,condition_status=status,**observed_row,
                unmatched_truth=[row["component"] for row in rows if not row["assigned"]],
                initial_unmatched_truth=initial_audit["unmatched_truth"],
                mneptune_assigned=neptune["assigned"],mneptune_shape_distance=neptune["shape_distance"],
                mneptune_count_rmse=neptune["count_rmse"],
                max_true_mneptune_fraction=float(np.max(counts[...,3]/counts.sum(axis=-1)))))
        plot_case(image,counts,spectra,wave,fits,audits,folder)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Dim far-red line: noiseless residual-guided recovery","",
        "Five true contributions: EBFP2/EGFP/mCherry bright spokes, one dim mNeptune line, "
        "and broad background everywhere. True data enters only the post-fit audit and figures. "
        "No brightness/geometry tuning is performed to force mNeptune to be omitted.","",
        "| Blur FWHM | Fit | Pool pixels | Iterations / stop | Relative fit | Unmatched truth | mNeptune shape error | Neighbor correlation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        if "strategy" not in row:
            lines.append(f"| {row['blur_fwhm_um']} | FAILED: {row['message']} | | | | | | |")
            continue
        shape="unmatched" if row["mneptune_shape_distance"] is None else f"{row['mneptune_shape_distance']:.4g}"
        lines.append(f"| {row['blur_fwhm_um']} | {row['strategy']} | {row['fit_pixels']} | "
            f"{row['iterations']} / {row['stop']} | {row['relative_l2']:.4g} | {', '.join(row['unmatched_truth']) or 'none'} | {shape} | {row['neighbor_correlation']:.4g} |")
    lines.extend(["","## What to inspect","",
        "Each blur folder has scene.png, observable_comparison.png and mneptune_truth_audit.png. "
        "Each fit has fresh residual_pca.png, factors, residual_correlations.npz and a simulation-only truth audit. "
        "selection/ records masks, exact sampled indices and explicit group shortfalls. "
        "initial_start_truth_audit.json separately records the starting spectra before ALS, "
        "so initial selection is not confused with the optimized four-component result.","",
        "Four hull/expanded fits share the same original starts. Five hull/expanded fits share "
        "the starts inferred from the expanded observed pool; the hull-only control thus shares "
        "initialization information and is not an independent blind pipeline. No-sample or "
        "near-zero residual cases stop explicitly. Rectangular truth matching only provides a "
        "best assignment; it is not proof of a pure missing species. Inspect all component audits.","",
        "This first phase has no measurement noise and does not establish general recovery. "
        "Correlations describe structure, not statistical significance or correct factors; values below the 1e-10 relative residual floor are NaN. Thin structures may disappear under opening; blur reduces peak line intensity as "
        "well as purity. Display percentiles and independent PCA bases need care. A low fit "
        "error or smooth map does not certify correct signatures. Higher counts/noise, repeated "
        "sampling and an excess sixth component remain subsequent controlled comparisons."])
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--blur-fwhm",type=float,nargs="+",default=BLUR_FWHM_UM,
                        help="Explicit blur conditions in micrometres; use a new output directory for an extension")
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output_dir,blur_fwhm_um=args.blur_fwhm)
