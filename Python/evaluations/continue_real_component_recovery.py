"""Second residual-guided round: five versus six components on one new pool.

Run from Python: python -m evaluations.continue_real_component_recovery
Prepare five-component runs, then continue using their fresh residual PC1.
Organize four/five/six results under component_recovery/. Outputs are local.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from lumic import essential_spectral_pixles, unique_profiles
from lumic.evaluation import compare_spectra
from lumic.io import load_czi
from .evaluate_real_extra_component import (SEEDS, select_additions, match_extra,
    RESULT_FILES as EXTRA_FILES, run as run_extra_component)
from .evaluate_real_residual_pooling import OUTPUT as BASE_OUTPUT, fit_and_project, UNIQUE_PERCENT
from .inspect_real_residual_structure import summarize_residuals, project_residuals, plot_directions
from ._execution import record_evaluation, prepare_prerequisites


OUTPUT = BASE_OUTPUT.parent / "real_component_recovery"


def copy_artifacts(source, target, names):
    """Create a clearly labelled result snapshot without altering saved inputs."""
    target.mkdir(parents=True,exist_ok=True)
    for old,new in names.items():
        if (source/old).exists(): shutil.copy2(source/old,target/new)
    (target/"provenance.json").write_text(json.dumps(dict(source=str(source.resolve()),
        files=names,note="Snapshot of prerequisite stage; originals preserved for reproducibility"),indent=2)+"\n")


def fresh_pca(pixels, folder, fit_name, wave, label):
    """Recompute centred residual directions and verify saved reconstruction RMS."""
    with np.load(folder/(fit_name+".npz")) as fit:
        abundance=fit["abundances"];spectra=fit["spectra"];rms=fit["rms"]
    count=len(spectra)
    stats=summarize_residuals(pixels,abundance.reshape(-1,count),spectra,rms,{})
    scores=project_residuals(pixels,abundance.reshape(-1,count),spectra,
                            stats["mean"],stats["basis"],rms.shape)
    np.testing.assert_allclose(scores.reshape(-1,6).var(axis=0,ddof=1),
                               stats["eigenvalues"][:6],rtol=1e-10)
    plot_directions(wave,stats,scores,folder/"residual_pca.png",label)
    np.savez_compressed(folder/"residual_diagnostics.npz",mean_residual=stats["mean"],
        covariance=stats["covariance"],eigenvalues=stats["eigenvalues"],loadings=stats["basis"],
        own_scores=scores,spectral_lag1_correlation=stats["correlation"],wavelengths_nm=wave)
    fractions=stats["eigenvalues"]/stats["eigenvalues"].sum()
    return dict(pc1_fraction=float(fractions[0]),first_six_fraction=float(fractions[:6].sum()))


def plot_six(wave, previous_spectra, spectra, abundance, folder):
    """Show all six spectra/maps and a matched candidate extra contribution.

    The rectangular spectral assignment is descriptive. Similar or split
    spectra can make the unmatched row an ambiguous identity. Contributions
    are in detector intensity units, not molecular concentrations.
    """
    order=match_extra(previous_spectra,spectra)
    fig=Figure(figsize=(15,16),dpi=130)
    axes=fig.subplots(6,2)
    for k,(curve,image) in enumerate(axes):
        curve.plot(wave,spectra[order[k]],label="6-component fit")
        if k<5:curve.plot(wave,previous_spectra[k],"--",label="Matched 5-component control")
        curve.set(title=f"Component {k+1}"+(" / unmatched candidate" if k==5 else ""),
                  xlabel="Wavelength (nm)",ylabel="Fraction per channel")
        curve.legend(fontsize=8)
        values=abundance[...,order[k]]
        h=image.imshow(values,cmap="viridis",vmin=0,
            vmax=max(float(np.quantile(values,.995)),np.finfo(float).tiny))
        image.set_title("Contribution; 99.5% display limit");image.axis("off")
        fig.colorbar(h,ax=image,shrink=.8,label="Detector intensity units")
    fig.tight_layout();fig.savefig(folder/"spectra_and_contributions.png")
    np.save(folder/"matching_order.npy",order)
    return spectra[order[-1]]


def plot_round_comparison(wave, before, control, six, source_pca, pixels,
                          control_folder, six_folder, folder):
    """Hold the pre-round five-component residual basis, mean and scales fixed."""
    with np.load(source_pca) as f:mean,basis,scores=f["mean_residual"],f["loadings"],f["own_scores"]
    maps=[scores[...,0]]
    for path,count in ((control_folder,5),(six_folder,6)):
        with np.load(path/"fit.npz") as f:
            values=project_residuals(pixels,f["abundances"].reshape(-1,count),
                f["spectra"],mean,basis[:1],before["rms"].shape)
        maps.append(values[...,0])
    fig=Figure(figsize=(15,9),dpi=130);axes=fig.subplots(2,3)
    vmax=max(np.quantile(f["rms"],.995) for f in (before,control,six))
    limit=max(np.quantile(np.abs(values),.995) for values in maps)
    for col,(f,title,values) in enumerate(zip((before,control,six),
        ("Round 1 / five components","Round 2 pool / five components","Round 2 pool / six components"),maps)):
        h=axes[0,col].imshow(f["rms"],cmap="magma",vmin=0,vmax=vmax)
        axes[0,col].set_title(title+" / RMS");fig.colorbar(h,ax=axes[0,col],shrink=.8)
        h=axes[1,col].imshow(values,cmap="RdBu_r",vmin=-limit,vmax=limit)
        axes[1,col].set_title("Pre-round residual PC1 / fixed basis and centre")
        fig.colorbar(h,ax=axes[1,col],shrink=.8)
    for ax in axes.flat:ax.axis("off")
    fig.suptitle("Shared 99.5% display limits; these maps are not a noise-significance test")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(folder/"round2_comparison.png")


@record_evaluation("root")
def run(root=OUTPUT, *, input_path=None, reuse_prerequisites=False):
    """Retain round-one pool, add both signs of its residual PC1, fit rank six.

    All original hull and first-round additions survive. At most 512 new
    pixels per PC1 sign are sampled after opening. Fit the measured spectra,
    not signed residuals. Five and six components use the same second-round
    pool; the five-component control restarts the prior fitted spectra while
    six starts are independently selected via a six-dimensional hull. Rank,
    initialization and pool changes are documented rather than conflated.
    No automatic claim that six components remove all structure is made.
    """
    root=Path(root)
    prepare_prerequisites(root, stage="evaluate_real_extra_component",
        required=EXTRA_FILES, reuse=reuse_prerequisites, input_path=input_path,
        build=lambda: run_extra_component(root, input_path=input_path))
    out=root/"component_recovery"
    out.mkdir(exist_ok=True)
    settings=json.loads((root/"settings.json").read_text())
    pixels=load_czi(settings["input_path"]).unfold().data
    with np.load(root/"selection.npz") as f:eligible=f["eligible"];wave=f["wavelengths_nm"]
    copy_artifacts(root,out/"components_4/hull_only",
        {"hull.npz":"fit.npz","settings.json":"settings.json"})
    copy_artifacts(root/"residual_structure",out/"components_4/hull_only",
        {"hull_pca.png":"residual_pca.png","hull_diagnostics.npz":"residual_diagnostics.npz"})
    records=json.loads((root/"extra_component/observed_metrics.json").read_text())
    extras=[]
    for seed in SEEDS:
        prior=root/"extra_component"/f"seed_{seed}"
        four1=out/"components_4/round1_pc_pool"/f"seed_{seed}"
        five1=out/"components_5/round1_pc_pool"/f"seed_{seed}"
        five2=out/"components_5/round2_pc1_pool"/f"seed_{seed}"
        six2=out/"components_6/round2_pc1_pool"/f"seed_{seed}"
        copy_artifacts(prior,four1,{"four.npz":"fit.npz","settings.json":"selection_settings.json"})
        copy_artifacts(prior,five1,{"five.npz":"fit.npz","five_residual_pca.png":"residual_pca.png",
            "five_residual_diagnostics.npz":"residual_diagnostics.npz",
            "spectra_and_extra_map.png":"spectra_and_extra_map.png","five_component_maps.png":"contribution_maps.png",
            "settings.json":"selection_settings.json"})
        fresh_pca(pixels,four1,"fit",wave,"Four components / first PC-expanded pool")
        with np.load(five1/"fit.npz") as f:
            pool=f["indices"];starts=f["spectra"];before=dict(rms=f["rms"])
        with np.load(five1/"residual_diagnostics.npz") as f:scores=f["own_scores"][...,:1]
        candidates=eligible.copy();candidates.flat[pool]=False
        added,masks,details=select_additions(scores,candidates,seed,n_directions=1)
        new_pool=np.union1d(pool,added)
        assert np.isin(pool,new_pool).all() and len(new_pool)==len(pool)+len(added)
        np.random.seed(seed)
        hull,local_indices=essential_spectral_pixles(pixels[new_pool],npc=6)
        six_starts,_=unique_profiles(hull.T,6,UNIQUE_PERCENT)
        print(f"Seed {seed}: retaining {len(pool)} pixels and adding {len(added)} from five-fit residual PC1",flush=True)
        five2.mkdir(parents=True,exist_ok=True);six2.mkdir(parents=True,exist_ok=True)
        control=fit_and_project(pixels,before["rms"].shape,starts,new_pool,"fit",five2)
        six=fit_and_project(pixels,before["rms"].shape,six_starts,new_pool,"fit",six2)
        for count,path,fitted in ((5,five2,control),(6,six2,six)):
            diagnostics=fresh_pca(pixels,path,"fit",wave,f"{count} components / second PC1-expanded pool")
            records.append(dict(seed=seed,round=2,components=count,**fitted["row"],**diagnostics))
            np.savez_compressed(path/"selection.npz",previous_indices=pool,added_indices=added,
                fitting_indices=new_pool,initial_spectra=starts if count==5 else six_starts,**masks)
            (path/"settings.json").write_text(json.dumps(dict(seed=seed,components=count,
                previous_pool="original hull + first-round PC1/PC2 additions",previous_pixels=len(pool),
                guidance="fresh residual PC1 from preceding five-component fit; both signs",
                new_additions=len(added),groups=details,opening_radius=1,score_quantile=.9,
                max_iterations=500,tol_percent=.01,norm="L2",solver="auto",
                initialization="previous fitted five spectra" if count==5 else "6D hull + unique_profiles",
                six_initialization_hull_indices=new_pool[local_indices].tolist()),indent=2)+"\n")
        with np.load(six2/"fit.npz") as f:abundance=f["abundances"]
        extras.append(plot_six(wave,control["spectra"],six["spectra"],abundance,six2))
        plot_round_comparison(wave,before,control,six,five1/"residual_diagnostics.npz",
                              pixels,five2,six2,six2)
    distance=compare_spectra(np.array([extras[0]]),np.array([extras[1]]),match=False).shape_distance[0]
    (out/"observed_metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    baseline=json.loads((root/"observed_metrics.json").read_text())[0]
    report=["# Real-image component recovery: four, five and six components","",
        "Organized snapshots preserve prior runs. Every component folder contains fit.npz "
        "and residual_pca.png (fresh centred residual decomposition for that fit). Original "
        "input folders remain available to earlier evaluation scripts.","",
        "## Primary inspection figures (seed 2026)","",
        "- [Four components: original hull residual PCA](components_4/hull_only/residual_pca.png)",
        "- [Five components: first-round residual PCA](components_5/round1_pc_pool/seed_2026/residual_pca.png)",
        "- [Six components: second-round residual PCA](components_6/round2_pc1_pool/seed_2026/residual_pca.png)",
        "- [All six spectra and contribution maps](components_6/round2_pc1_pool/seed_2026/spectra_and_contributions.png)",
        "- [Second-round comparison on a fixed residual basis](components_6/round2_pc1_pool/seed_2026/round2_comparison.png)","",
        "```text","components_4/hull_only/","components_4/round1_pc_pool/seed_2026|2027/",
        "components_5/round1_pc_pool/seed_2026|2027/",
        "components_5/round2_pc1_pool/seed_2026|2027/  # same-pool control",
        "components_6/round2_pc1_pool/seed_2026|2027/","```","",
        "| Seed | Fit | Pool pixels | Iterations | Stop | Full-image relative L2 | Mean RMS |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| original | hull / 4 | {baseline['fit_pixels']} | {baseline['iterations']} | {baseline['stop']} | {baseline['relative_l2']:.6g} | {baseline['mean_rms']:.6g} |"]
    for row in records:
        name=f"round 2 / {row['components']}" if row.get("round")==2 else "round 1 / "+row["strategy"]
        report.append(f"| {row['seed']} | {name} | {row['fit_pixels']} | {row['iterations']} | {row['stop']} | {row['relative_l2']:.6g} | {row['mean_rms']:.6g} |")
    report.extend(["",f"Unmatched sixth candidate spectral distance between sampling seeds: {distance:.6g}. "
        "This assignment is descriptive and may be ambiguous.","",
        "Second round retains the original hull and all first-round additions. It samples "
        "up to 512 extra pixels per sign of fresh five-fit residual PC1, after opening. "
        "Five/six controls share the new pool; initialization differs. Original measured "
        "spectra enter ALS. The previous RMS-expanded four-component run is never used.","",
        "Fresh PCA plots have independent directions and scales: look for remaining patterns, "
        "not direct PC-number correspondence. round2_comparison.png fixes the preceding "
        "five-fit basis/mean and shares scales to inspect change. Centring removes mean "
        "residual bias, which is retained in residual_diagnostics.npz. None of these "
        "raw residual maps are noise-whitened or significance tests. More factors can split "
        "known spectra or capture detector/background effects; structure-free PCA is neither "
        "guaranteed nor proof of correct recovery. Display clips at 99.5%; arrays are unclipped.","",
        "## User-provided sample interpretation","",
        "The previously unexpected far-red contribution is interpreted by the user as the "
        "actin-attached fluorophore believed to be quenched, supported by its spatial features. "
        "This is sample context, not an identity established by blind factorization."])
    (out/"README.md").write_text("\n".join(report)+"\n")
    print("\n".join(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", type=Path, default=OUTPUT)
    parser.add_argument("--input", type=Path, help="Local CZI; otherwise download public high-SNR COS-7 data.")
    parser.add_argument("--reuse-prerequisites", action="store_true", help="Reuse completed four/five-component prerequisites in the output directory.")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output_dir, input_path=args.input, reuse_prerequisites=args.reuse_prerequisites)
