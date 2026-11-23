# Spectra selection

After [loading an image](1-input.md), obtain one initial reference spectrum for
each component. Select **Automatic**, **Manual** or **Load JSON**
from **References**.

## Automatic references

1. Set **Number of components** and **Median Intensity Quantile**. The quantile
   excludes the lowest-intensity pixels from reference selection; fitting later
   still uses every pixel.
2. Click **Calculate initial reference spectra**. The plugin finds distinctive
   pixel spectra and displays the selected points, spectra and distance map.
3. Review the results and continue to [Unmixing](3-unmixing.md).

## Manual references

1. Select and show the raw-data layer, and click a
   pixel. Its spectrum appears in the live plot.
2. Click **Set clicked spectrum** to add it as a reference. Repeat for as many
   components as needed; no component count is required in advance.
   **Clear manual spectra** clears the selection so
   you can start again.
3. Review the initial spectra and click **Confirm manual references** to
   calculate the initial distance layer and enable unmixing.
   The number of accepted spectra determines the number of components.
   Browsing more pixels leaves confirmed references unchanged; accepting another
   spectrum adds it to the selection and requires confirmation again.

## Load JSON references

Choose **Load JSON**, set **Median Intensity Quantile**, and click
**Load spectra JSON...**. Select either a single spectrum saved by
`Spectrum.save_json`, or a collection saved by the plugin. The number of
components is taken from the file. Loaded references initialize ALS
optimization using the same retained hull pixel pool as automatic references.

Channel counts and wavelength order must match the image. If wavelength
coordinates are absent, both image and references must use channel order.
Detection bands stored in the reference provenance must also match. Values
are never interpolated, reordered or normalized on import; these checks do
not establish compatibility between different detector configurations.

## Distance from the spectra centre

After references are **loaded**, **calculated**, or **confirmed**, the plugin
creates an image layer named **Initial distance from spectral center**.
It measures each pixel's spectral-shape distance from the mean position of the
reference spectra in a shared, normalized spectral space. **Low values** mean
the pixel's spectrum lies near that centre; **high values** mean it lies farther
away and has a more distinct spectral shape relative to the centre.

This is not an abundance map, a measure of fitting error, or a guarantee that
a pixel contains a pure component: noise, background and artifacts can also
produce high values. If the spectral projection cannot be computed, the plugin
reports that the distance layer is unavailable.

## Spectrum colours and saving

Colours follow the spectra's wavelength characteristics where possible. If a
later component would receive a colour very similar to one already assigned,
the plugin gives it a different colour that is distinct from the earlier ones.

In every mode, **Save initial reference spectra...** saves all components in
one JSON file, retaining their values, numeric dtypes, optional wavelengths,
names and provenance. When the current image contains spatial calibration,
each component's provenance also includes available X/Y/Z `pixel_sizes_um`
and `pixel_unit` set to `micrometer`; sizes are expressed in micrometres.
Missing calibration is left absent. These describe the source image and do
not restrict reuse on images with a different pixel size.
Reuse the saved file with **Load JSON**; it can contain
one or several components.
