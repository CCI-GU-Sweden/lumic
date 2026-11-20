# Spectra selection

After [loading an image](1-input.md), choose the number of components and obtain
one initial reference spectrum for each. Select **Automatic** or **Manual**
from **References**.

## Automatic references

1. Set **Number of components** and **Median Intensity Quantile**. The quantile
   excludes the lowest-intensity pixels from reference selection; fitting later
   still uses every pixel.
2. Click **Calculate initial reference spectra**. The plugin finds distinctive
   pixel spectra and displays the selected points, spectra and distance map.
3. Review the results and continue to [Unmixing](3-unmixing.md).

## Manual references

1. Set **Number of components**, select and show the raw-data layer, and click a
   pixel. Its spectrum appears in the live plot.
2. Click **Set clicked spectrum** to add it as a reference. Repeat until all
   components are selected. **Clear manual spectra** clears the selection so
   you can start again.
3. Review the initial spectra and click **Confirm manual references** to
   calculate the initial distance layer and enable unmixing.

In either mode, **Save initial reference spectra as CSVs...** saves one CSV per
component. Each contains intensity and wavelength, or detector channel index
when wavelength metadata is unavailable.
