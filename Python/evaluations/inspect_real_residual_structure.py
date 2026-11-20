"""Plot spectral and spatial structure of the saved real-image residuals.

Run from Python: python -m evaluations.inspect_real_residual_structure
Uses saved hull/spatial-pool fits; never reruns ALS. Outputs remain local.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from lumic.evaluation import residual_pca_from_factors, residual_spectral_correlation
from lumic.io import load_czi
from .evaluate_real_residual_pooling import OUTPUT, PROJECTION_BATCH


DIRECTIONS = 6
METHODS = ("hull", "spatial_residual")


def residual_chunks(pixels, abundance, spectra):
    """Retain signed observed-minus-fitted spectra; no absolute value/clipping."""
    for first in range(0,len(pixels),PROJECTION_BATCH):
        last=min(first+PROJECTION_BATCH,len(pixels))
        yield first,last,pixels[first:last].astype(float)-abundance[first:last]@spectra


def summarize_residuals(pixels, abundance, spectra, rms, groups):
    """Accumulate centred PCA covariance, signed means and spectral correlation.

    PCA is across raw detector channels after subtracting their population
    mean residual. It is not the brightness-normalized hull projection. The
    separate mean spectra preserve constant errors that centred PCA removes.
    Region masks are fixed from the hull fit, not biological segmentations.
    Lag-one correlation is Pearson correlation between consecutive-channel
    residual vectors within each pixel. With 32 channels, individual values
    are noisy; fitted/noise correlations preclude a universal significance
    cutoff. Channel ordering and wavelength gaps affect this diagnostic.
    """
    channels=pixels.shape[1]
    pca=residual_pca_from_factors(pixels,abundance,spectra,
        n_components=min(DIRECTIONS,channels,len(pixels)-1),
        batch_size=PROJECTION_BATCH,return_scores=False)
    region_sums={name:np.zeros(channels) for name in groups}
    correlation=np.full(len(pixels),np.nan)
    for first,last,residual in tqdm(residual_chunks(pixels,abundance,spectra),
                                    total=(len(pixels)+PROJECTION_BATCH-1)//PROJECTION_BATCH,
                                    desc="Residual statistics"):
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2,axis=1)),
                                   rms.ravel()[first:last],rtol=1e-10,atol=1e-8)
        for name,mask in groups.items():
            region_sums[name]+=residual[mask.ravel()[first:last]].sum(axis=0)
        correlation[first:last]=residual_spectral_correlation(residual)
    means={name:region_sums[name]/mask.sum() if mask.any() else np.full(channels,np.nan)
           for name,mask in groups.items()}
    return dict(mean=pca.mean_residual,covariance=pca.covariance,
                eigenvalues=pca.eigenvalues,basis=pca.loadings,
                means=means,correlation=correlation.reshape(rms.shape))


def project_residuals(pixels, abundance, spectra, mean, basis, shape):
    """Map signed residual patterns using a specified centre and PCA basis.

    Scores have raw intensity units; loadings are unit-L2 vectors. Scores are
    pattern amplitudes, not abundances. A shared centre/basis lets two fits be
    compared without confounding sign/rotation changes from refitting PCA.
    """
    scores=np.empty((len(pixels),len(basis)))
    for first,last,residual in residual_chunks(pixels,abundance,spectra):
        scores[first:last]=(residual-mean)@basis.T
    return scores.reshape(*shape,len(basis))


def plot_directions(wave,stats,scores,path,title):
    """Pair each signed spectral loading with its signed spatial score map."""
    directions=len(stats["basis"])
    fig=Figure(figsize=(11,3*directions),dpi=130)
    axes=fig.subplots(directions,2,squeeze=False)
    total=stats["eigenvalues"].sum()
    fractions=stats["eigenvalues"]/total if total else np.zeros_like(stats["eigenvalues"])
    for k,(curve,image) in enumerate(axes):
        curve.plot(wave,stats["basis"][k])
        curve.axhline(0,color="gray",lw=.5)
        curve.set(title=f"Residual PC{k+1}: {fractions[k]:.1%} of centred variance",
                  xlabel="Wavelength (nm)",ylabel="Signed unit-L2 loading")
        limit=max(float(np.quantile(np.abs(scores[...,k]),.995)),np.finfo(float).tiny)
        h=image.imshow(scores[...,k],cmap="RdBu_r",vmin=-limit,vmax=limit)
        image.set_title("Signed score map; symmetric 99.5% display limits")
        image.axis("off")
        fig.colorbar(h,ax=image,shrink=.8,label="Detector intensity units")
    fig.suptitle(title+" — raw residuals, no noise whitening",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(path)


def run(folder=OUTPUT):
    """Inspect fixed fits; describe unexplained structure, not missing identity.

    Random noise need not be spectrally flat or uncorrelated after fitting.
    PCA can reveal detector effects, inaccurate known spectra, nonlinearities
    and omitted components. Directions are signed and need not be physical
    pure spectra. More residual PCs do not count additional fluorophores.
    """
    folder=Path(folder)
    settings=json.loads((folder/"settings.json").read_text())
    output=folder/"residual_structure"
    output.mkdir(exist_ok=True)
    pixels=load_czi(settings["input_path"]).unfold().data
    with np.load(folder/"selection.npz") as selection:
        wave=selection["wavelengths_nm"]
        high,opened=selection["high_mask"],selection["opened_mask"]
    groups={"supported high residual":opened,
            "high residual removed by opening":high&~opened,
            "remaining pixels":~high}
    shape=high.shape
    stats,own_scores,common_scores={}, {}, {}
    for method in METHODS:
        with np.load(folder/(method+".npz")) as fit:
            abundance=fit["abundances"].reshape(-1,settings["components"])
            spectra,rms=fit["spectra"],fit["rms"]
        stats[method]=summarize_residuals(pixels,abundance,spectra,rms,groups)
        own_scores[method]=project_residuals(pixels,abundance,spectra,
                            stats[method]["mean"],stats[method]["basis"],shape)
        common_scores[method]=(own_scores[method] if method=="hull" else
            project_residuals(pixels,abundance,spectra,stats["hull"]["mean"],stats["hull"]["basis"],shape))
        # Independent decomposition describes each fit; only common maps
        # below have the same basis, centre and display limits across fits.
        plot_directions(wave,stats[method],own_scores[method],
                        output/(method+"_pca.png"),method+" / centred residual PCA")
        np.savez_compressed(output/(method+"_diagnostics.npz"),
            mean_residual=stats[method]["mean"],covariance=stats[method]["covariance"],
            eigenvalues=stats[method]["eigenvalues"],loadings=stats[method]["basis"],
            own_scores=own_scores[method],hull_basis_scores=common_scores[method],
            spectral_lag1_correlation=stats[method]["correlation"],wavelengths_nm=wave)

    fig=Figure(figsize=(15,3*DIRECTIONS),dpi=130)
    axes=fig.subplots(DIRECTIONS,3)
    for k in range(DIRECTIONS):
        axes[k,0].plot(wave,stats["hull"]["basis"][k])
        axes[k,0].axhline(0,color="gray",lw=.5)
        axes[k,0].set(title=f"Hull residual PC{k+1}",xlabel="Wavelength (nm)",ylabel="Signed loading")
        limit=max(float(np.quantile(np.abs(common_scores[m][...,k]),.995)) for m in METHODS)
        for col,method in enumerate(METHODS,1):
            h=axes[k,col].imshow(common_scores[method][...,k],cmap="RdBu_r",
                vmin=-max(limit,np.finfo(float).tiny),vmax=max(limit,np.finfo(float).tiny))
            axes[k,col].set_title(method+" / common hull basis and centre")
            axes[k,col].axis("off")
            fig.colorbar(h,ax=axes[k,col],shrink=.8)
    fig.suptitle("Residual patterns before/after pooling — shared symmetric 99.5% limits")
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(output/"pca_common_basis_comparison.png")

    fig=Figure(figsize=(14,10),dpi=130)
    axes=fig.subplots(2,2)
    for col,method in enumerate(METHODS):
        axes[0,col].plot(wave,stats[method]["mean"],"k--",label="Whole image")
        for name,mean in stats[method]["means"].items():
            axes[0,col].plot(wave,mean,label=name)
        axes[0,col].axhline(0,color="gray",lw=.5)
        axes[0,col].set(title=method+" / mean signed residual spectra",
                        xlabel="Wavelength (nm)",ylabel="Detector intensity units")
        axes[0,col].legend(fontsize=8)
        h=axes[1,col].imshow(stats[method]["correlation"],cmap="RdBu_r",vmin=-1,vmax=1)
        axes[1,col].set_title(method+" / consecutive-channel residual correlation")
        axes[1,col].axis("off")
        fig.colorbar(h,ax=axes[1,col],shrink=.8)
    # Shared limits keep the signed-mean magnitude comparison honest.
    low=min(axes[0,col].get_ylim()[0] for col in range(2))
    high=max(axes[0,col].get_ylim()[1] for col in range(2))
    for col in range(2): axes[0,col].set_ylim(low,high)
    fig.tight_layout()
    fig.savefig(output/"means_and_spectral_correlation.png")
    report=["# Real-image residual structure","",
        "Signed residuals reconstructed from existing saved fits; no ALS rerun. "
        "Every chunk was checked against the saved full-resolution RMS map.","",
        "[PCA common-basis comparison](pca_common_basis_comparison.png) · "
        "[Hull PCA](hull_pca.png) · [Expanded-pool PCA](spatial_residual_pca.png) · "
        "[Signed means and spectral correlation](means_and_spectral_correlation.png)","",
        "| Fit | Mean spectral lag-one correlation | PC1 variance | First six PCs variance |",
        "| --- | --- | --- | --- |"]
    for method in METHODS:
        fractions=stats[method]["eigenvalues"]/stats[method]["eigenvalues"].sum()
        report.append(f"| {method} | {np.nanmean(stats[method]['correlation']):.4g} | {fractions[0]:.2%} | {fractions[:DIRECTIONS].sum():.2%} |")
    report.extend(["","## Interpretation","",
        "PCA centres raw signed detector-channel residuals, without brightness normalization "
        "or noise whitening. The mean spectra are shown separately so constant spectral bias "
        "is not hidden. Six directions are displayed, not asserted to be significant components. "
        "Eigenvalues describe centred residual variance, not molecular abundances or species counts.","",
        "The independent PCA plots can rotate/change sign/order between fits. The paired "
        "comparison instead fixes the hull basis AND mean, with shared symmetric display limits. "
        "Positive/negative scores indicate opposite deviations along the same pattern. Sign is "
        "arbitrary; it is fixed by making the largest absolute loading positive. Display clips "
        "at the 99.5th percentile; full-resolution NPZ scores are not clipped.","",
        "Regional means use the same hull-derived masks for both fits: surviving high-residual "
        "regions, high-residual pixels removed by opening, and remaining pixels. These are "
        "not manually identified nuclei, lines or background. Opening can remove real thin features.","",
        "Lag-one correlation measures spectral structure, not its significance or origin. "
        "Fitting induces residual correlations even with independent measurement noise. Noise "
        "variance/covariance, unequal channel spacing, detector discontinuities and brightness "
        "must be considered before a noise-null test. A flat nonzero residual can also be "
        "systematic background. Mean correlation averages pixels; localized maps are more useful.","",
        "Repeated spectral/spatial patterns motivate investigating an omitted component, "
        "inaccurate existing signatures or detector/model mismatch. Residual PCs can be signed "
        "distortions of missing spectra and cannot directly supply pure nonnegative signatures. "
        "This is exploratory on the same image; no automatic component-count or recovery claim."])
    (output/"summary.md").write_text("\n".join(report)+"\n")
    (output/"settings.json").write_text(json.dumps(dict(directions=DIRECTIONS,
        source_settings=settings,groups={k:int(v.sum()) for k,v in groups.items()},
        pca="centred raw signed residuals; no whitening",comparison_basis="hull mean and loadings",
        spectral_correlation="Pearson adjacent-channel vectors within each pixel",
        noise_significance="not assessed"),indent=2)+"\n")
    print("\n".join(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args.results_dir)
