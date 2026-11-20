"""Compare four/five components with additions to the ORIGINAL hull.

Run from Python: python -m evaluations.evaluate_real_extra_component
Residual PC1/PC2 of the original hull fit guide pixel selection. Fit original
measured spectra, never signed residuals. This is an exploratory local pilot.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from lumic import essential_spectral_pixles, unique_profiles
from lumic.evaluation import compare_spectra, select_residual_pixels
from lumic.io import load_czi
from .evaluate_real_residual_pooling import OUTPUT as BASE_OUTPUT, fit_and_project, UNIQUE_PERCENT
from .inspect_real_residual_structure import (project_residuals, summarize_residuals,
                                             plot_directions, RESULT_FILES as RESIDUAL_FILES,
                                             run as inspect_residuals)
from ._execution import record_evaluation, prepare_prerequisites


SEEDS = (2026, 2027)
PER_SIGN_BUDGET = 512
SCORE_QUANTILE = .9
OPENING_RADIUS = 1
OUTPUT = BASE_OUTPUT.parent / "real_extra_component"
RESULT_FILES = (*RESIDUAL_FILES, "extra_component/observed_metrics.json", *(
    f"extra_component/seed_{seed}/{name}"
    for seed in SEEDS for name in (
        "four.npz", "five.npz", "settings.json", "five_residual_pca.png",
        "five_residual_diagnostics.npz", "spectra_and_extra_map.png", "five_component_maps.png")))


def select_additions(scores, eligible, seed, *, n_directions=2, pcs=None,
                     per_sign_budget=PER_SIGN_BUDGET, score_quantile=SCORE_QUANTILE,
                     opening_radius=OPENING_RADIUS):
    """Adapt public residual selection to the existing evaluation tuple format.

    These callers already exclude their current pool from eligibility.
    Thresholding, opening, sampling and provenance now live in one public
    implementation; this wrapper only supplies pilot defaults/result format.
    """
    pcs = list(range(1, n_directions+1)) if pcs is None else pcs
    result = select_residual_pixels(scores, pcs=pcs, eligible=eligible,
        per_sign_budget=per_sign_budget, score_quantile=score_quantile,
        opening_radius=opening_radius, seed=seed)
    return result.indices, result.opened_masks, result.groups


def match_extra(four,five):
    """Match n spectra to n+1; unmatched row is a candidate, not identity.

    Rectangular assignment uses unit-L2 spectral distance. With split or
    similar components this matching can be ambiguous. Calling the unmatched
    row 'extra' does not establish a new fluorophore or a pure recovered spectrum.
    """
    a=four/np.linalg.norm(four,axis=1,keepdims=True)
    b=five/np.linalg.norm(five,axis=1,keepdims=True)
    rows,cols=linear_sum_assignment(np.sum((a[:,None]-b[None])**2,axis=-1))
    order=cols[np.argsort(rows)]
    extra=int(np.setdiff1d(np.arange(len(five)),order)[0])
    return np.r_[order,extra]


def plot_case(wave, baseline, four, five, abundance, order, scores, added, folder):
    """Show changes on fixed residual directions and the candidate extra factor."""
    fig=Figure(figsize=(15,11),dpi=140)
    axes=fig.subplots(3,3)
    for col,f in enumerate((baseline,four,five)):
        vmax=max(np.quantile(g["rms"],.995) for g in (baseline,four,five))
        h=axes[0,col].imshow(f["rms"],cmap="magma",vmin=0,vmax=vmax)
        axes[0,col].set_title(("Original hull / 4","PC-expanded pool / 4","Same pool / 5")[col]+" / RMS")
        fig.colorbar(h,ax=axes[0,col],shrink=.8)
    for k in range(2):
        limit=max(np.quantile(np.abs(s[...,k]),.995) for s in scores)
        for col,s in enumerate(scores):
            h=axes[k+1,col].imshow(s[...,k],cmap="RdBu_r",vmin=-limit,vmax=limit)
            axes[k+1,col].set_title(f"Original hull residual PC{k+1} / signed amplitude")
            fig.colorbar(h,ax=axes[k+1,col],shrink=.8)
    for ax in axes.flat: ax.axis("off")
    fig.suptitle("Shared original-hull PCA basis/centre; shared 99.5% display limits")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(folder/"residual_comparison.png")

    fig=Figure(figsize=(15,10),dpi=140)
    axes=fig.subplots(2,3)
    for k in range(4):
        ax=axes.flat[k]
        ax.plot(wave,four["spectra"][k],label="4-component fit")
        ax.plot(wave,five["spectra"][order[k]],label="Matched 5-component fit")
        ax.set(title=f"Matched anonymous component {k+1}",xlabel="Wavelength (nm)",ylabel="Fraction per channel")
        ax.legend(fontsize=8)
    axes.flat[4].plot(wave,five["spectra"][order[-1]])
    axes.flat[4].set(title="Unmatched candidate extra spectrum",xlabel="Wavelength (nm)",ylabel="Fraction per channel")
    h=axes.flat[5].imshow(abundance[...,order[-1]],cmap="viridis",vmin=0,
        vmax=np.quantile(abundance[...,order[-1]],.995))
    axes.flat[5].set_title("Candidate extra contribution (99.5% limits)")
    axes.flat[5].axis("off");fig.colorbar(h,ax=axes.flat[5],shrink=.8,label="Detector intensity units")
    fig.tight_layout();fig.savefig(folder/"spectra_and_extra_map.png")

    fig=Figure(figsize=(15,8),dpi=130)
    axes=fig.subplots(2,3)
    for k in range(5):
        h=axes.flat[k].imshow(abundance[...,order[k]],cmap="viridis",vmin=0,
            vmax=max(np.quantile(abundance[...,order[k]],.995),np.finfo(float).tiny))
        axes.flat[k].set_title(f"Five-component contribution {k+1}"+(" (unmatched)" if k==4 else ""))
        fig.colorbar(h,ax=axes.flat[k],shrink=.8)
    axes.flat[5].imshow(baseline["rms"],cmap="magma")
    y,x=np.unravel_index(added,baseline["rms"].shape)
    axes.flat[5].scatter(x,y,c="cyan",s=1)
    axes.flat[5].set_title(f"Original-hull RMS + {len(added)} PC-guided additions")
    for ax in axes.flat:ax.axis("off")
    fig.tight_layout();fig.savefig(folder/"five_component_maps.png")


@record_evaluation("root")
def run(root=OUTPUT, *, input_path=None, reuse_prerequisites=False):
    """Hold the fitting pool fixed when comparing four versus five components.

    Uses only the ORIGINAL hull fit and its residual PCA; the earlier
    RMS-expanded fit does not enter selection or initialization. For each seed
    retain all original hull indices and add PC-guided pixels. Five-dimensional
    hull selection within that pool supplies five blind starts; four starts
    remain those saved for the original fit. Differences therefore include
    model rank and initialization, not just rank. Repeat two sampling seeds as
    a limited sensitivity check, not a bootstrap/confidence estimate.
    """
    root=Path(root)
    prepare_prerequisites(root, stage="inspect_real_residual_structure",
        required=RESIDUAL_FILES, reuse=reuse_prerequisites, input_path=input_path,
        build=lambda: inspect_residuals(root, input_path=input_path))
    output=root/"extra_component"
    output.mkdir(exist_ok=True)
    settings=json.loads((root/"settings.json").read_text())
    pixels=load_czi(settings["input_path"]).unfold().data
    with np.load(root/"selection.npz") as f:
        hull,eligible,starts,wave=f["hull_indices"],f["eligible"],f["blind_starts"],f["wavelengths_nm"]
    with np.load(root/"hull.npz") as f:
        baseline=dict(rms=f["rms"],spectra=f["spectra"])
    with np.load(root/"residual_structure/hull_diagnostics.npz") as f:
        scores=f["own_scores"];mean=f["mean_residual"];basis=f["loadings"]
    records,extras=[],[]
    for seed in SEEDS:
        folder=output/f"seed_{seed}";folder.mkdir(exist_ok=True)
        added,masks,details=select_additions(scores,eligible,seed)
        pool=np.union1d(hull,added)
        assert np.isin(hull,pool).all() and len(pool)==len(hull)+len(added)
        np.random.seed(seed)
        candidates,candidate_indices=essential_spectral_pixles(pixels[pool],npc=5)
        five_starts,_=unique_profiles(candidates.T,5,UNIQUE_PERCENT)
        print(f"Seed {seed}: original hull {len(hull)} + {len(added)} additions; "
              f"5D initialization hull {len(candidate_indices)}",flush=True)
        four=fit_and_project(pixels,baseline["rms"].shape,starts,pool,"four",folder)
        five=fit_and_project(pixels,baseline["rms"].shape,five_starts,pool,"five",folder)
        records.extend([dict(seed=seed,**f["row"]) for f in (four,five)])
        with np.load(folder/"five.npz") as f: abundance=f["abundances"]
        # Refit PCA to the NEW residuals: the fixed old basis can miss patterns
        # that become prominent after another component has been fitted.
        remaining=summarize_residuals(pixels,abundance.reshape(-1,5),
                                     five["spectra"],five["rms"],{})
        own_scores=project_residuals(pixels,abundance.reshape(-1,5),five["spectra"],
                                    remaining["mean"],remaining["basis"],baseline["rms"].shape)
        plot_directions(wave,remaining,own_scores,folder/"five_residual_pca.png",
                        "Five-component fit / fresh centred residual PCA")
        np.savez_compressed(folder/"five_residual_diagnostics.npz",mean_residual=remaining["mean"],
            covariance=remaining["covariance"],eigenvalues=remaining["eigenvalues"],
            loadings=remaining["basis"],own_scores=own_scores,
            spectral_lag1_correlation=remaining["correlation"],wavelengths_nm=wave)
        order=match_extra(four["spectra"],five["spectra"])
        extras.append(five["spectra"][order[-1]])
        projections=[scores]
        for name,fitted in (("four",four),("five",five)):
            with np.load(folder/(name+".npz")) as f:
                projected=project_residuals(pixels,f["abundances"].reshape(-1,len(fitted["spectra"])),
                    fitted["spectra"],mean,basis,baseline["rms"].shape)
            projections.append(projected)
        plot_case(wave,baseline,four,five,abundance,order,projections,added,folder)
        np.savez_compressed(folder/"selection_and_scores.npz",hull_indices=hull,added_indices=added,
            four_starts=starts,five_starts=five_starts,five_matching_order=order,
            four_hull_basis_scores=projections[1],five_hull_basis_scores=projections[2],**masks)
        (folder/"settings.json").write_text(json.dumps(dict(seed=seed,base_pool="original hull only",
            original_hull_pixels=len(hull),actual_additions=len(added),groups=details,
            initialization_hull_indices=pool[candidate_indices].tolist(),score_quantile=SCORE_QUANTILE,
            opening_radius=OPENING_RADIUS,four_start_source="original saved blind starts",
            five_start_source="5D hull then unique_profiles within PC-expanded original-hull pool"),indent=2)+"\n")
    change=compare_spectra(np.array([extras[0]]),np.array([extras[1]]),match=False)
    report=["# Four versus five components: original hull plus residual-PC additions","",
        "The original 88-pixel hull is the base. The previous RMS-expanded fit/pool is not used. "
        "Original-hull residual PC1/PC2, each positive and negative, guide up to 512 additions "
        "per group after opening. Both component counts use exactly the same fitting pool "
        "within a seed. Five starts come from a 5D hull of that pool; four starts are unchanged.","",
        "| Seed | Components | Pool pixels | Iterations | Stop | Full-image relative L2 | Mean RMS |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        report.append(f"| {r['seed']} | {r['strategy']} | {r['fit_pixels']} | {r['iterations']} | {r['stop']} | {r['relative_l2']:.6g} | {r['mean_rms']:.6g} |")
    report.extend(["",f"Unmatched candidate spectral shape distance between two seeds: {change.shape_distance[0]:.6g}. "
        "Matching can be ambiguous; this is sensitivity, not a confidence interval.","",
        "For each seed inspect `residual_comparison.png`, `spectra_and_extra_map.png`, "
        "and `five_component_maps.png`. `five_residual_pca.png` independently refits PCA "
        "to the five-component residuals to expose remaining patterns; its PC order/basis "
        "cannot be directly matched to the old PCs. Signed comparison maps use the original hull basis "
        "and centre for all fits. All original measured spectra, not residual spectra, enter ALS.","",
        "## Limits","",
        "Rank and blind initialization both change. A fifth factor may split an existing "
        "signature, capture detector/background effects or an omitted component. The unmatched "
        "spectrum is not automatically a new fluorophore or a pure recovered signature. "
        "A smaller same-image residual is expected with more flexibility and does not prove "
        "recovery. Raw PC scores are not noise-normalized; thin real features can be lost by "
        "opening. Two sampling seeds are a bounded sensitivity check, not robust significance "
        "or independent validation. Component identity requires controls/references. "
        "The 500-update cap and notebook stopping settings are retained; outputs remain local."])
    (output/"summary.md").write_text("\n".join(report)+"\n")
    (output/"observed_metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    print("\n".join(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", type=Path, default=OUTPUT)
    parser.add_argument("--input", type=Path, help="Local CZI; otherwise download public high-SNR COS-7 data.")
    parser.add_argument("--reuse-prerequisites", action="store_true", help="Reuse completed baseline and residual inspection in the output directory.")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output_dir, input_path=args.input, reuse_prerequisites=args.reuse_prerequisites)
