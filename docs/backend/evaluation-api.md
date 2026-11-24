# Evaluation API

| What you have | What you can evaluate | Use |
| --- | --- | --- |
| An experimental image and its fitted reconstruction | How closely the fit reproduces the measured signal and where it leaves structured errors | `reconstruction_error_maps()`, residual PCA, and residual correlations |
| Two fits or a set of reference spectra on the same detector grid | How similar the component spectra are and which rows correspond | `compare_spectra()` and `match_components()` |
| A simulation with known spectra and abundance maps | How accurately the fit recovers the true spectra and abundances, in addition to how well it fits the image | `evaluate_factors()` and `concentration_error_maps()` |

## Experimental images: assess the fit without ground truth

On a typical experimental image, you have the measured detector signal
but do not know the true contribution of each fluorophore at each pixel.
You can check where the model fits poorly and compare spectra across fits.
Even a close fit can come from different combinations of spectra and
abundances. These checks cannot tell you the true abundance error.

### Locate unexplained signal in space and across channels

Pass observed and reconstructed arrays with identical shapes, the same
channel order and intensity units, and channels last. The residual is
**observed minus reconstructed**. Positive values indicate underestimated
signal. RMS measures the mismatch in the input intensity units, while
relative L2 is dimensionless. Use `relative_valid` to find pixels where
the observed signal was strong enough for a relative error to be reported.

```python
import matplotlib.pyplot as plt

from lumic.evaluation import reconstruction_error_maps, residual_pca_from_residuals

# observed and predicted are Y × X × C arrays on the same detector grid.
errors = reconstruction_error_maps(observed, predicted)
pca = residual_pca_from_residuals(errors.residuals, n_components=2)

fig, (ax_error, ax_scores, ax_loading) = plt.subplots(1, 3, figsize=(10, 3))
rms = ax_error.imshow(errors.rms)
ax_error.set_title("Residual RMS")
fig.colorbar(rms, ax=ax_error, label="Detector intensity")

pc1_scores = pca.scores[..., 0]
limit = abs(pc1_scores).max() or 1.0
scores = ax_scores.imshow(pc1_scores, cmap="RdBu_r", vmin=-limit, vmax=limit)
ax_scores.set_title(f"PC1 scores ({pca.explained_variance_ratio[0]:.1%})")
fig.colorbar(scores, ax=ax_scores, label="Detector intensity")

ax_loading.plot(pca.loadings[0])
ax_loading.axhline(0, color="gray", linewidth=1)
ax_loading.set(xlabel="Detector channel index", ylabel="PC1 loading")
fig.tight_layout()
```

`errors.rms` shows **where** the model leaves a large error. The score map
shows **where a particular residual pattern** is strong, and the matching
loading shows **which channels** make up that pattern. Plot the loading
against wavelength coordinates when they are available. The percentage in
the score panel describes the fraction of centered residual variance
explained by PC1. Inspect other PCs as well.

A spatially coherent score map paired with a structured loading can
suggest a missing component, especially if the pattern recurs in another
acquisition. Background, detector artifacts, or a mismatched reference
spectrum can also leave structured residuals. PCA loadings can have both
signs and are directions of variation. Use the pattern to decide what to
inspect or test in a follow-up fit.

PCA removes the mean before fitting. Inspect `pca.mean_residual` separately
for a systematic channel offset. `residual_neighbor_correlation()` summarizes
similarity between adjacent pixels in a 2D image, while
`residual_spectral_correlation()` maps correlation between consecutive
channels within each pixel. The same checks also work on simulated data,
including noisy simulations.
See the [evaluation walkthrough](evaluate-results.md) for the plots and a
controlled [missing-component example](evaluate-results.md#what-does-a-missing-component-look-like).

::: lumic.evaluation.reconstruction_error_maps
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.ReconstructionErrorMaps
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false
      members: [residuals, rms, relative_l2, relative_valid]

::: lumic.evaluation.residual_pca_from_residuals
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.ResidualPCAResult
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false
      members: [mean_residual, loadings, scores, explained_variance, explained_variance_ratio, eigenvalues, covariance, spatial_shape, n_samples]

::: lumic.evaluation.residual_neighbor_correlation
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.residual_spectral_correlation
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

### Compare component spectra

Use two `(K, C)` matrices with the same component count and detector
channels in the same order. `match_components()` returns estimated row
indices in reference order. Use those indices to reorder estimated spectra
and their paired abundance columns together. `compare_spectra()` reports
changes in spectral shape after removing scale differences. Use
`match=False` when rows already correspond and you want to preserve that
pairing. Agreement with a previous fit or a reference spectrum tells you
about spectral similarity. It cannot validate the abundance maps.

::: lumic.evaluation.match_components
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.compare_spectra
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.SpectralComparison
    options:
      heading_level: 4
      show_root_heading: true
      show_root_full_path: false
      members: [component_order, matching_costs, shape_distance, max_channel_change, max_channel_index]

## Simulated images: measure recovery against known truth

A simulation provides the component spectra and abundance maps that
generated the clean detector signal. Keep those factors when adding noise
so you can compare the fit against known truth. These recovery functions
also apply to an experimental validation dataset if it supplies independent
ground truth for both quantities.

`evaluate_factors()` summarizes recovery from flattened observations
`(N, C)`, abundance matrices `(N, K)`, and spectra `(K, C)`.
`concentration_error_maps()` also accepts spatial abundance maps with
components last. It returns signed and absolute errors in true component
order and masks relative errors where the reference concentration is
too small.

Both functions match estimated components to the reference shapes, then
put spectra on a unit-sum scale and transfer the removed scale to their
paired abundances. This preserves the predicted detector signal while
making abundances comparable. For a detector image in count units, the
aligned abundance scale is total detector counts per component.
See [Evaluate simulation recovery](evaluate-simulation-recovery.md).

::: lumic.evaluation.evaluate_factors
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.FactorEvaluation
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [component_order, matching_costs, estimated_abundances, estimated_spectra, reconstruction_relative_l2, spectral_relative_l2_per_component, spectral_relative_l2_global, max_channel_absolute, max_error_component_index, max_error_channel_index, abundance_rmse_per_component, abundance_rmse_global, abundance_relative_l2_per_component, abundance_relative_l2_global]

::: lumic.evaluation.concentration_error_maps
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.evaluation.ConcentrationErrorMaps
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [signed, absolute, relative, relative_valid, component_order]
