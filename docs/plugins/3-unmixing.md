# Unmixing

After [selecting references](2-spectra-selection.md), calculate component
abundances for the selected image plane.

1. Set **Worker processes** for the abundance fit.
2. Click **Optimize spectra and calculate abundance maps**. With automatic or JSON
   references, the plugin optimizes the spectra before fitting the maps. With
   manual references, the button reads **Calculate abundance maps** and fits
   using the selected spectra unchanged.
3. Review the spectra, abundance layers and signed residual, then continue to
   [Evaluation](4-evaluation.md).

The plugin also creates **Optimized distance from spectral center** using the
fitted reference spectra as the centre. It has the same interpretation as the
[initial distance map](2-spectra-selection.md#distance-from-the-spectra-centre);
it does not measure the residual or fitting error.

## Display controls

- **Optimized spectra display** switches between unnormalized and L1-normalized
  plots; it does not change fitted values or exports.
- **Show spectra** selects all components, individual components, and/or the
  mean signed residual in the plot.
- **Layer visibility** selects abundance maps and the residual. **Deselect all
  layers** hides them. **Additive** combines selected layers; **Opaque** shows
  at most one at a time. Select and show the residual layer and use the bottom
  wavelength slider to inspect its individual channels.

## Save results

- **Save optimised spectra...** writes all fitted components and their metadata to one JSON
  file, preserving their fitted scale regardless of the plot display setting.
  Available source-image pixel sizes and their micrometre unit are included
  in each component's provenance.
  Use **Load JSON** in Spectra selection to reuse these spectra on a compatible
  image. The button's tooltip describes the JSON export.
- **Save abundance maps as OME-TIFF...** writes quantitative maps with their
  fitted spectra for reconstruction. Components are anonymous and abundance
  values have no assumed physical units.
