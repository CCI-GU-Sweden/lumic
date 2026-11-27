# Unmixing and spectral selection

Use `positive_least_squares()` when the component spectra are known. It
fits nonnegative abundances while holding those spectra fixed. Use
`mcr_als()` when you have starting spectra and want to refine them along
with the abundances. Neither function identifies fluorophores by name.

## Array shapes and scale

| Array | Shape | Ordering |
| --- | --- | --- |
| Observations | `(N, C)` | One row per pixel or other observation |
| Component spectra | `(K, C)` | One row per component, with the same detector channels and order as the observations |
| Abundances | `(N, K)` | Same observation order as the input and same component order as the spectra |

`image.unfold().data` provides the observation matrix. For a
`SpectrumCollection`, use `references.values` as the component matrix after
checking that the detector grid matches the image. Save the exact basis and
its scale with the abundances if you want to reconstruct the fit later. See
[Image containers](image-containers-api.md) and
[Save images and spectra](save-results.md).

## Fixed references and MCR-ALS

Normalize the references explicitly when you want to reuse or save the
effective basis. For example, normalize each reference to unit L1 norm,
then fit with `do_norm=False` to use those rows as supplied.

```python
from lumic import normalize_components, positive_least_squares

unfolded = image.unfold()
basis = normalize_components(references.values, do_norm="L1")
abundances = positive_least_squares(unfolded.data, basis, do_norm=False)
maps = unfolded.fold_abundances(abundances, spectra=basis, normalization="L1")
```

With MCR-ALS, the returned abundances and spectra come from the same best
iteration. If you refine on a subset of pixels, fit the resulting spectra
against the full set of observations before folding the abundances.
See [Unmix a ternary image](unmix-ternary-image.md) and
[Estimate unknown spectra](estimate-unknown-spectra.md).

::: lumic.unmixing.positive_least_squares
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.unmixing.mcr_als
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.unmixing.MCRALSDiagnostics
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [stop_reason, iterations_completed, best_iteration, best_relative_l2]

::: lumic.unmixing.normalize_components
    options:
      show_root_heading: true
      show_root_full_path: false

## Select pixels and starting spectra

Use `select_analysis_pixels()` on a loaded 3D spectral plane to track the
pixels included in the analysis. Its indices map back to the original
spatial positions. To find starting spectra, pass the selected observations
to `essential_spectral_pixels()`, then transpose the candidate matrix
before passing it to `unique_profiles()`.
For the geometry, limitations and examples with and without noise, see
[Why use a convex hull?](convex-hull.md#why-use-a-convex-hull).

```python
from lumic import essential_spectral_pixels, select_analysis_pixels, unique_profiles

plane = image.slice()
pixels = plane.unfold().data
selection = select_analysis_pixels(plane.data, spectral_axis=plane.axes.index("C"))
candidates, candidate_indices = essential_spectral_pixels(
    pixels[selection.indices], npc=3,
)
starts, _ = unique_profiles(candidates.T, nr=3, f=1.0)
```

`candidate_indices` indexes the selected rows. Use
`selection.indices[candidate_indices]` to get the corresponding original
pixel indices. The old spelling `essential_spectral_pixles` remains an alias
for existing code.

::: lumic.utils.select_analysis_pixels
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.utils.AnalysisPixelSelection
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [indices, mask, image_shape, spectral_axis, median_quantile, median_threshold, exclusion_masks, exclusion_counts]

::: lumic.uniqueness.essential_spectral_pixels
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.uniqueness.unique_profiles
    options:
      show_root_heading: true
      show_root_full_path: false

## Inspect spectral distinctness

`spectral_uniqueness_map()` scores how far each usable pixel lies from the
center of a fitted spectral projection. It does not select a component
basis or assign fluorophore identities. `spectral_uniqueness_rgb()` turns
those scores and directions into a display image and requires the
`visualization` extra. See [Spectral uniqueness maps](spectral-uniqueness-maps.md).

::: lumic.uniqueness.spectral_uniqueness_map
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.uniqueness.SpectralUniquenessResult
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [scores, valid_mask, total_intensity, coordinates, center_coordinates, projection_components]

::: lumic.uniqueness.spectral_uniqueness_rgb
    options:
      show_root_heading: true
      show_root_full_path: false
