# Comparing spectral fits

## When to use these functions

`align_factors` and `compare_spectra` support both real data and simulations.
They help compare fits made with different pixel pools, sampling seeds,
initial spectra, or solver settings. They do not need known concentrations.

| Function | What it answers | What it cannot establish |
| --- | --- | --- |
| `align_factors` | Which fitted rows correspond best to the reference, and how can their spectra and abundances share a consistent scale? | Biological identity, missing components, recovery accuracy, or confidence intervals. |
| `compare_spectra` | How much did each spectral shape change, and what is its largest single-channel change? | Which fit is correct, whether changes are statistically significant, or whether ALS converged. |

Use `evaluate_factors` and `concentration_error_maps` when you **do** have
ground truth and want recovery accuracy. Use `reconstruction_error_maps` to
inspect measured-minus-predicted signal without requiring ground truth.
A good reconstruction or stable signature can still conceal inaccurate
spectra and concentrations.

## Check reconstruction fit on any image

Given measured data `D`, abundances `C`, and spectra `S`, the fitted signal
is `C @ S`. Use `reconstruction_error_maps(D, C @ S)` for flattened data
or pass two matching images with detector channels last.

| Output | What to inspect |
| --- | --- |
| `residuals` | Signed difference `D - C @ S` in each channel. Positive values show where the model underestimates the measured signal. |
| `rms` | Per-pixel root-mean-square error across channels, in detector intensity units. Use it to locate large absolute mismatches. |
| `relative_l2` | Per-pixel residual norm divided by the measured signal norm. Use `relative_valid` to identify where the ratio is defined. Pixels at or below `min_signal` have a NaN ratio. |

A whole-image relative L2 error uses a single norm over all pixels and
channels. It will generally differ from the average of the per-pixel relative
errors. Report the aggregation you used. For a simulation with noise, comparing
to the measured image tells you how well the fit explains the realized data.
Comparing the prediction to the saved clean image answers how well it
reconstructs the expected signal.

## Measure factor recovery when ground truth is known

`evaluate_factors` reports summary errors from flattened pixels.
`concentration_error_maps` shows where the abundance estimates differ from
truth. Both use the same matching and scaling. First, match estimated rows
to the reference by minimizing the sum of squared distances between unit-L2
spectral shapes. Then put spectra on a unit-sum scale and transfer the
removed scale to the paired abundance columns. This preserves each factor
pair's reconstruction. For count-domain simulations, the aligned abundances
are total contributions across detector channels in counts.

| `evaluate_factors` output | Meaning |
| --- | --- |
| `reconstruction_relative_l2` | `norm(D - C @ S) / norm(D)` over all supplied pixels and channels. Use the observed data for the fit score. |
| `spectral_relative_l2_per_component`, `spectral_relative_l2_global` | Distance between matched unit-L2 spectral shapes, per component and as the root mean square of those distances across components. |
| `max_channel_absolute`, `max_error_component_index`, `max_error_channel_index` | Largest absolute difference between matched unit-L2 spectra, with its zero-based position in reference order. |
| `abundance_rmse_per_component`, `abundance_rmse_global` | Typical magnitude of abundance errors in detector intensity units. The global score pools pixels and components. |
| `abundance_relative_l2_per_component`, `abundance_relative_l2_global` | L2 norm of the abundance error divided by the corresponding truth norm. A zero truth norm gives NaN. |

```python
import numpy as np
from lumic.evaluation import concentration_error_maps, evaluate_factors

true_spectra = np.array([[1., 0., 0.], [0., 0., 1.]])
true_abundances = np.array([[10., 0.], [2., 8.], [0., 10.]])
observed = true_abundances @ true_spectra

estimated_spectra = np.array([[0., 0., 2.], [1., 0., 0.]])
estimated_abundances = np.array([[0.25, 9.], [4., 3.], [5., 0.]])
metrics = evaluate_factors(
    observed, true_abundances, true_spectra, estimated_abundances, estimated_spectra,
)
maps = concentration_error_maps(
    true_abundances, true_spectra, estimated_abundances, estimated_spectra,
)
print(metrics.component_order)  # [1, 0]
print(metrics.spectral_relative_l2_per_component)  # [0., 0.]
print(maps.signed[0])  # [-1., 0.5] for the first pixel, in reference order
```

For images, pass abundance maps with components last to
`concentration_error_maps`. Positive signed errors show overestimation.
Inspect absolute errors for leakage where a component is absent. Relative
errors are signed fractions and are masked where the true abundance is at
or below `min_concentration`. Matching requires the same component and channel
counts and corresponding detector channels. Report missing or extra
components separately when comparing fits of different ranks.

## Example: align a reordered and rescaled fit

```python
import numpy as np
from lumic.evaluation import align_factors, compare_spectra

# These may be spectra from a previous blind fit, not known pure signatures.
reference = np.array([[1., 0., 0.], [0., 0., 1.]])
fitted_spectra = np.array([[0., 0., 4.], [2., 0., 0.]])
fitted_abundances = np.array([[3., 5.], [1., 2.]])

aligned = align_factors(reference, fitted_abundances, fitted_spectra)
print(aligned.component_order)  # [1, 0]: original fitted row indices
print(aligned.abundances)      # [[10., 12.], [4., 4.]]
assert np.allclose(aligned.spectra.sum(axis=1), 1)
assert np.allclose(aligned.abundances @ aligned.spectra,
                   fitted_abundances @ fitted_spectra)
```

Each spectrum now sums to one across the supplied detector channels. Its
removed scale is transferred to the paired abundance column, preserving the
reconstructed image. Abundance is the component's summed contribution in the
data's intensity units; it is not automatically a molecular concentration.
Inputs remain unchanged. A zero fitted spectrum receives zero canonical
abundance because it contributes no signal.

## Example: compare estimates along a sampling trajectory

```python
# All fits should first be aligned to the SAME reference, as above.
# Here one component broadens slightly while the other stays unchanged.
next_spectra = np.array([[0.9, 0.1, 0.], [0., 0., 1.]])
change = compare_spectra(aligned.spectra, next_spectra, match=False)
print(change.shape_distance)      # approximately [0.1106, 0.]
print(change.max_channel_change)  # [0.1, 0.]: 10 percentage points
print(change.max_channel_index)   # [0, 0]: first channel when tied
```

`shape_distance` compares unit-L2 spectral shapes, independent of brightness.
`max_channel_change` compares unit-sum spectra: it reports the largest absolute
change in the fraction of total signal assigned to a channel. These scales
answer different questions and should be labelled separately in plots.
Neither measures changes in abundance; inspect those separately.

The default `match=True` independently pairs rows to the supplied reference.
Use `match=False` once all fits share the same reference ordering, to keep
component IDs consistent rather than allowing pairwise comparisons to switch
them. Similar signatures can still make the original assignment ambiguous.

Both functions require finite, nonnegative spectra with equal numbers of
components and channels. Reference rows used for matching must be nonzero.
Fixed-order comparison permits zero rows: zero versus zero has distance 0,
zero versus a nonzero unit-L2 spectrum has distance 1. This is an absent-signal
convention, not a spectral shape for a vanished component.

For pool-growth or repeated-fit studies, inspect stopping diagnostics alongside
these differences: capped runs may change simply because optimization is
incomplete. Repeated-seed ranges describe sampling sensitivity, not calibrated
confidence intervals. These comparisons remove permutation and scaling
differences; they do not resolve the ambiguity of different factors explaining
the same image.

### Compatibility with existing recovery metrics

`evaluate_factors.max_channel_absolute` retains its existing **unit-L2**
definition. The new `compare_spectra.max_channel_change` uses **unit-sum**
fractions for interpretation across detector channels. These fields are not
interchangeable; existing recovery reports keep their original definitions.

## Check MCR-ALS convergence

Request `return_diagnostics=True` from `lumic.mcr_als()` to get
`(abundances, spectra, history, diagnostics)`. The history starts at the
initial fit and records relative L2 lack of fit on the **supplied fitting
pixels** after each complete update. The returned factors correspond to
`diagnostics.best_iteration` and `diagnostics.best_relative_l2`. Use
`diagnostics.iterations_completed` and `diagnostics.stop_reason` to tell
whether the fit met the optional convergence tolerance or reached the
iteration limit. `tol_percent` is the relative **percentage improvement**
between consecutive fit errors.

For fits trained on different pixel pools, evaluate their predictions on the
same image or held-out pixels for a direct fit comparison. Use reference
factors or simulation truth to assess recovery accuracy.

## Signed residual PCA: spectral patterns and spatial maps

Use these diagnostics on real images or simulations to inspect variation left
unexplained by a reconstruction. Ground truth is not required. The calculation
does not plot, save files, fit another component or decide how many species
exist. Plot the returned loadings against wavelengths and scores as images.

```python
from lumic.evaluation import (
    residual_pca, residual_pca_from_residuals, residual_pca_from_factors,
)

# Channel-last images, matching the reconstruction-error map API:
result = residual_pca(observed, reconstructed, n_components=3)

# An existing SIGNED residual cube (not absolute errors or pixel RMS):
result = residual_pca_from_residuals(residuals, n_components=3)

# Large CZI-style channel-first data, without building C @ S for the whole image:
result = residual_pca_from_factors(
    mixData, abundances, spectra, spectral_axis=0, n_components=6,
    batch_size=65536,
)
pc1_image = result.scores[..., 0]
pc1_spectral_pattern = result.loadings[0]
mean_spectral_bias = result.mean_residual
```

Default `spectral_axis=-1` supports (..., channels), including flattened
(pixels, channels) data. Set zero explicitly for (channels, y, x). All other
axes retain their order; scores always add a final component axis. Factor
spectra have shape (components, channels); abundances may be flattened or
spatial with components last. Inputs must be finite real numbers; signed data
and factors are supported because diagnostics need not assume nonnegative
unmixing. Subtraction occurs in float64, preserving negative residuals from
unsigned image data. Inputs are not modified.

| Returned field | Meaning and use |
| --- | --- |
| `mean_residual` | Mean signed error per channel; inspect persistent bias removed by centred PCA |
| `loadings` | Orthonormal signed spectral directions, one per returned PC |
| `scores` | Signed spatial pattern amplitudes in detector-intensity units, not abundances |
| `explained_variance` | Selected sample-covariance eigenvalues (denominator pixels−1) |
| `explained_variance_ratio` | Fraction of total **centred residual** variance, not original-image variance |
| `eigenvalues`, `covariance` | All channel eigenvalues and covariance, for further diagnostics |
| `spatial_shape`, `n_samples` | Spatial mapping and number of observations |

Choose `n_components` between one and min(channels, pixels−1); at least two
pixels are required. Zero/constant residuals return zero variance, ratios and
scores, with an arbitrary orthonormal basis rather than meaningful directions.
Loading signs use a positive largest absolute entry for reproducible display,
but sign is a convention. Tied eigenvalues can rotate directions. Independently
fitted results need not have matching PC numbers; for direct comparisons,
project both datasets with one fixed mean and basis.

Covariance uses stable centred batch-moment merging rather than subtracting
large raw moments. Reconstruction/residual cubes are not allocated by the
factor path. Scores require roughly pixels × PCs × 8 bytes plus input/layout
and batch working memory; `return_scores=False` avoids score allocation and
the second projection pass. Reshaping unusual input layouts may copy data.

### Interpretation limits

No brightness normalization, shot/read-noise whitening, permutation test or
automatic rank selection is provided. Patterns can reflect omitted components,
inaccurate known spectra, detector artifacts or brightness-dependent and
fitting-induced noise correlations. PCA loadings are signed distortions, not
pure missing signatures. More PCs do not mean more fluorophores. A visually
flat residual does not establish accurate or unique factors, and centred PCA
can hide constant spectral error if the returned mean is ignored.

Residual structure can guide **experimental** sampling/inspection decisions;
scientific significance needs an appropriate noise model and independent
checks. The user-guided notebook keeps those decisions outside this API.

## Select pixels from residual-PC regions

`select_residual_pixels` turns user-chosen residual score maps into traceable
**additional pixel indices**. Use it to investigate coherent unexplained
patterns by expanding a fitting pool; it does not increase the component
count, fit the model, save files or decide that a missing species exists.

```python
import numpy as np
from lumic.utils import select_analysis_pixels
from lumic.evaluation import residual_pca_from_factors, select_residual_pixels

retained = select_analysis_pixels(mixData, median_quantile=0.20)
pca = residual_pca_from_factors(
    mixData, abundances, spectra, spectral_axis=0, n_components=6,
)
selection = select_residual_pixels(
    pca.scores, pcs=[1, 2],              # One-based PCs chosen after inspection
    eligible=retained.mask,
    existing_indices=pool_indices,
    per_sign_budget=512, score_quantile=0.90,
    opening_radius=1, seed=2026,
)
expanded_indices = np.union1d(pool_indices, selection.indices)
# Fit original measured spectra at expanded_indices, not PCA scores/residuals.
```

Currently score maps must have shape (y, x, PCs); this is a 2D image selector,
not a volume selector. Original spatial indices use C-order flattening, matching
the analysis-pixel selector. Selecting two PCs does not mean adding two species;
the workflow sets rank separately. The user-guided notebook explicitly calls
this public selector before its prototype pool-combination/save wrapper.

For each selected PC, process positive then negative scores. On eligible pixels
excluding the existing pool, calculate the quantile of `max(sign * score, 0)`.
**Opposite-sign pixels contribute zeros**; this is not a percentile calculated
only among that sign's nonzero pixels. Retain threshold ties and positive values.
Opening acts on the full mask before intersecting eligibility, so excluded pool
pixels can provide support without being sampled. Radius one uses a 3x3 cross;
zero disables opening. Outside-field support is false-padded.

Uniform sampling draws up to `per_sign_budget` distinct pixels per group.
Already selected pixels are excluded from later groups. Requested PC order
therefore affects overlaps; larger regions carry more sampling weight. Empty
and exhausted groups report shortfalls; no unrelated random replacement is used.

| Result field | Purpose |
| --- | --- |
| `indices` | Sorted distinct additions only; caller combines them with the pool |
| `existing_indices`, `eligible_mask` | Record the excluded pool and permitted addition locations |
| `threshold_masks`, `opened_masks` | Full masks before eligibility intersection, for inspecting support |
| `candidate_masks` | Opened masks intersected with permitted non-pool pixels; groups may overlap |
| `groups` | Thresholds, candidate/available/selected counts, reasons and exact sampled indices |
| `settings` | Requested PCs, budgets, opening/quantile settings, seed and actual entropy |

A fixed seed makes repeated choices reproducible. `seed=None` uses fresh
entropy; reuse `selection.settings['seed_entropy']` as the seed to replay it.
Arrays and input masks are not modified; the function performs no file I/O.

### What the selector does not establish

Raw PC amplitudes are not noise-normalized significance scores. Structured
regions may follow brightness-dependent noise, detector effects or errors in
existing signatures. Opening can erase real thin features and preserve chance
clusters. More selected pixels or a smaller same-image residual does not prove
accurate factors or identify a missing fluorophore. Choose PCs from the current
PCA, since directions can reorder between fits. Callers manage stale diagnostics,
pool history, controls and component-count decisions outside this API.


## Spatial and spectral residual correlation

Both functions accept **signed observed-minus-reconstructed residuals** and
work on real or simulated data without ground truth. They perform no plotting
or file I/O and do not modify inputs.

```python
from lumic.evaluation import (reconstruction_error_maps,
    residual_neighbor_correlation, residual_spectral_correlation)

# Channel-last measured and reconstructed images, shape (y, x, channels).
errors = reconstruction_error_maps(observed, reconstructed)
spatial_score = residual_neighbor_correlation(errors.residuals)
spectral_map = residual_spectral_correlation(errors.residuals)
# For (channels, y, x) residual images, pass spectral_axis=0 instead.
```

| Diagnostic | Output and purpose | Limits |
| --- | --- | --- |
| `residual_neighbor_correlation` | One scalar pooling horizontal/vertical signed residual pairs; compare spatial persistence across fits | Requires a 2D image; channels with larger residual variance have more weight; global score may hide localized changes |
| `residual_spectral_correlation` | Per-pixel Pearson correlation of consecutive spectral channels; locate smooth or alternating spectral errors | Depends on channel ordering and gaps; individual values from short spectra are noisy |

Both range from -1 to 1 and center the compared vectors separately. The
spatial function centers each channel across endpoint populations; the
spectral function centers each pixel's two consecutive-channel vectors.
Neither is a correlation of RMS maps. Constant offsets disappear through
centering: inspect signed mean residuals and error magnitudes too. Zero
endpoint variance gives NaN, not zero; fewer than three channels also give
NaN for spectral correlation. A single spatial pixel has no neighbor pairs.

Use `batch_size` (default 65536) to limit temporary pair/pixel arrays. Spatial
pairs never wrap around image boundaries or include diagonals. Spectral maps
also accept flattened (pixels, channels) data. All inputs must be finite.

These are descriptive diagnostics, **not significance tests**, photon-noise
estimates, measures of species identity, or proof of correct concentrations.
Fitting, PSF, detector processing and noise can create correlations. A flatter
residual map may reflect redistributed errors. Compare maps with RMS, signed
means and residual PCA; use ground-truth factor errors separately when they
exist. Correlations are scale-free, so tiny numerical residuals can still
correlate strongly. Callers set any practical magnitude floor explicitly;
the API applies no automatic noise threshold or noise normalization.
