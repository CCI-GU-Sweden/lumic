"""Experimental notebook workflow; deliberately outside the public LUMIC API.

Separate component count, residual-PC selection and solver settings. Existing
public numerical functions and evaluation prototypes perform the calculations.
No sample-specific paths, biological identities or automatic rank stopping.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic import essential_spectral_pixles, unfold_spec_data, unique_profiles
from lumic.utils import AnalysisPixelSelection
from lumic.evaluation import (residual_pca_from_factors, select_residual_pixels,
                              ResidualPixelSelection)
from .evaluate_real_residual_pooling import fit_and_project
from .inspect_real_residual_structure import plot_directions


def _pixels(image):
    """Notebook convention is CZI-compatible (channel, y, x)."""
    image=np.asarray(image)
    if (image.ndim!=3 or min(image.shape)<1 or image.shape[0]<2
            or np.iscomplexobj(image) or not np.isfinite(image).all() or np.any(image<0)):
        raise ValueError("image must be finite nonnegative (channels, y, x) data")
    return unfold_spec_data(image)


def convex_hull_pool(image, retained, *, npc=4, seed=2026):
    """Find a hull from an explicitly supplied analysis-pixel selection.

    No intensity filtering occurs here. select_analysis_pixels supplies the
    retained original indices and selection mask. Translate hull-local indices
    back to full-image coordinates. Later expansions use the same retained mask.
    Hull dimension npc and fitted count are separate concepts. The prototype
    currently uses finite nonnegative channel-first images; selection alone
    does not make invalid pixels safe for full-image solver projection.
    """
    pixels=_pixels(image)
    if (not isinstance(retained, AnalysisPixelSelection) or retained.image_shape!=image.shape
            or retained.spectral_axis!=0):
        raise ValueError("retained must describe this channel-first image")
    if len(retained.indices)<npc:
        raise ValueError("too few retained pixels for the requested hull projection")
    state=np.random.get_state()
    try:
        np.random.seed(seed)
        _,local=essential_spectral_pixles(pixels[retained.indices],npc=npc)
    finally:np.random.set_state(state)
    return dict(indices=np.unique(retained.indices[local]),eligible=retained.mask.copy(),
                median_threshold=retained.median_threshold,median_quantile=retained.median_quantile,
                exclusion_counts=retained.exclusion_counts,npc=npc,seed=seed)


def fit_pool(image, pool, *, n_components, output_dir, initial_spectra=None,
             seed=2026, max_iterations=500, tol_percent=.01, norm="L2", unique_percent=.5):
    """Fit original measured spectra on a pool, then project across the image.

    If initial_spectra is absent, use an n_components-dimensional hull within
    the supplied pool, then unique_profiles to choose blind starts. A control
    fit can instead restart a previous fit's spectra. Returned spectra have
    unit sum; contributions use detector intensity units, not necessarily
    photon counts or molecular concentrations. Unresolved projections fail
    explicitly rather than yielding apparently valid residual maps.

    Writes fit.npz, metrics.json and settings.json. The result's indices retain
    original flattened image coordinates. Fits with different component counts
    can be compared on one pool, but initialization may still differ. No true
    signatures or concentrations enter this function.
    """
    pixels=_pixels(image)
    indices=np.asarray(pool["indices"])
    if (indices.ndim!=1 or not indices.size or indices.dtype.kind not in "iu"
            or np.any(indices<0) or np.any(indices>=len(pixels)) or len(np.unique(indices))!=len(indices)):
        raise ValueError("pool indices must be distinct valid image pixel indices")
    if isinstance(n_components,(bool,np.bool_)) or not isinstance(n_components,(int,np.integer)) or not 2<=n_components<=min(pixels.shape[1],len(indices)):
        raise ValueError("n_components must be an integer between 2 and available channels/pixels")
    if initial_spectra is None:
        state=np.random.get_state()
        try:
            np.random.seed(seed)
            candidates,_=essential_spectral_pixles(pixels[indices],npc=n_components)
            starts,_=unique_profiles(candidates.T,n_components,unique_percent)
        finally:np.random.set_state(state)
    else:
        starts=np.asarray(initial_spectra,dtype=float)
        if starts.shape!=(n_components,pixels.shape[1]):
            raise ValueError("initial_spectra must match requested components and image channels")
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    fit=fit_and_project(pixels,image.shape[1:],starts,indices,"fit",folder,
                        max_iterations=max_iterations,tol_percent=tol_percent,norm=norm)
    fit.update(indices=indices.copy(),folder=folder,n_components=n_components)
    (folder/"metrics.json").write_text(json.dumps(fit["row"],indent=2)+"\n")
    (folder/"settings.json").write_text(json.dumps(dict(n_components=n_components,seed=seed,
        max_iterations=max_iterations,tol_percent=tol_percent,norm=norm,solver="auto",
        initialization="blind hull + unique_profiles" if initial_spectra is None else "provided previous spectra",
        image_shape=image.shape,pool_pixels=len(indices),initial_spectra=starts.tolist()),indent=2)+"\n")
    return fit


def inspect_fit_residuals(image, fit, wavelengths, *, n_components=None):
    """Plot/save public residual PCA for a prototype fit; no PCA math lives here.

    Centre raw residuals across pixels, without brightness normalization or
    noise whitening. Covariance is accumulated in chunks; no full spectral
    residual cube is retained. Mean signed residuals are saved separately.
    Scores are signed pattern amplitudes, not nonnegative concentrations.
    Fresh directions can rotate/reorder between fits, so PC numbers are local
    to this diagnostic result. Fitting itself can induce noise correlations.
    Structured directions suggest unexplained variation, not species identity
    or a statistically established component count.
    """
    pixels=_pixels(image);folder=fit["folder"]
    wavelengths=np.asarray(wavelengths)
    if wavelengths.shape!=(image.shape[0],) or not np.isfinite(wavelengths).all():
        raise ValueError("wavelengths must match the image channels")
    if n_components is None:
        n_components=min(6,image.shape[0],len(pixels)-1)
    with np.load(folder/"fit.npz") as saved:abundance=saved["abundances"].reshape(-1,fit["n_components"])
    result=residual_pca_from_factors(image,abundance,fit["spectra"],
                                   n_components=n_components,spectral_axis=0)
    stats=dict(mean=result.mean_residual,basis=result.loadings,eigenvalues=result.eigenvalues)
    plot_directions(wavelengths,stats,result.scores,folder/"residual_pca.png",
                    f"{fit['n_components']} components / fresh centred residual PCA")
    np.savez_compressed(folder/"residual_diagnostics.npz",mean_residual=result.mean_residual,
        covariance=result.covariance,eigenvalues=result.eigenvalues,loadings=result.loadings,
        own_scores=result.scores,explained_variance_ratio=result.explained_variance_ratio,
        wavelengths_nm=wavelengths)
    return dict(scores=result.scores,mean=result.mean_residual,basis=result.loadings,eigenvalues=result.eigenvalues,
                source_folder=folder,source_indices=fit["indices"].copy())


def plot_factors(fit, wavelengths):
    """Pair recovered spectral shapes with full-image component contributions.

    Anonymous components are ordered by the current fit's blind starts, not
    identified by biology. Each map has its own disclosed 99.5% display limit.
    Spectra have unit sum, making amplitudes contributions across the measured
    detector channels. The saved arrays remain unclipped.
    """
    with np.load(fit["folder"]/"fit.npz") as saved:abundance=saved["abundances"]
    fig=Figure(figsize=(11,3*fit["n_components"]),dpi=120)
    axes=fig.subplots(fit["n_components"],2)
    for k,(curve,image) in enumerate(axes):
        curve.plot(wavelengths,fit["spectra"][k])
        curve.set(title=f"Anonymous component {k+1}",xlabel="Wavelength (nm)",ylabel="Fraction per channel")
        values=abundance[...,k]
        h=image.imshow(values,cmap="viridis",vmin=0,
            vmax=max(float(np.quantile(values,.995)),np.finfo(float).tiny))
        image.set_title("Contribution / 99.5% display limit");image.axis("off")
        fig.colorbar(h,ax=image,shrink=.8,label="Detector intensity units")
    fig.tight_layout();fig.savefig(fit["folder"]/"spectra_and_contributions.png")


def expand_pool(pool, diagnostics, *, selection=None, pcs=None, per_sign_budget=512,
                score_quantile=.9, opening_radius=1, seed=2026, output_dir):
    """Combine/save a public residual selection as a new prototype fitting pool.

    Select both signs of each requested PC, apply opening, then intersect with
    retained-intensity eligibility excluding every existing pool pixel. Keep
    all existing pixels and record new samples without duplicates. Selected
    PCs do not specify the number of extra components: set that separately in
    fit_pool. Opening can erase genuine small/thin features or retain chance
    noise clusters. Uniform sampling within masks gives large regions more
    weight; it does not guarantee diversity or noise significance.

    Empty candidates produce no additions for that group and are explicitly
    recorded. No unrelated random pixels silently replace them. Fixed PC/sign
    order influences overlapping groups. Save masks/indices/settings so a
    human can review exactly which pixels were added.
    Prefer passing selection from select_residual_pixels explicitly. For old
    prototype callers, supplying pcs delegates to that same public API first.
    """
    if not np.array_equal(pool["indices"],diagnostics["source_indices"]):
        raise ValueError("diagnostics must belong to the current pool; inspect fresh residual PCA first")
    if selection is None:
        selection=select_residual_pixels(diagnostics["scores"],pcs=pcs,
            eligible=pool["eligible"],existing_indices=pool["indices"],seed=seed,
            per_sign_budget=per_sign_budget,score_quantile=score_quantile,opening_radius=opening_radius)
    elif pcs is not None:
        raise ValueError("pass a selection or pcs, not both")
    expected_eligibility=pool["eligible"].copy()
    expected_eligibility.flat[pool["indices"]]=False
    if (not isinstance(selection,ResidualPixelSelection)
            or not np.array_equal(selection.existing_indices,np.unique(pool["indices"]))
            or not np.array_equal(selection.eligible_mask,expected_eligibility)):
        raise ValueError("selection must exclude the current fitting pool and match its spatial shape")
    added,masks,details=selection.indices,selection.opened_masks,selection.groups
    indices=np.union1d(pool["indices"],added)
    result={**pool,"indices":indices,"added_indices":added}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"selection.npz",previous_indices=pool["indices"],added_indices=added,
                        fitting_indices=indices,**masks)
    (folder/"selection_settings.json").write_text(json.dumps(dict(**selection.settings,
        source_fit=str(diagnostics["source_folder"]),groups=details,
        previous_pixels=len(pool["indices"]),added_pixels=len(added),total_pixels=len(indices)),indent=2)+"\n")
    fig=Figure(figsize=(5*len(masks),5),dpi=120)
    axes=np.atleast_1d(fig.subplots(1,len(masks)))
    for ax,(name,mask) in zip(axes,masks.items()):
        ax.imshow(mask,cmap="gray");ax.set_title(name+" / after opening");ax.axis("off")
    fig.tight_layout();fig.savefig(folder/"selection_masks.png")
    return result
