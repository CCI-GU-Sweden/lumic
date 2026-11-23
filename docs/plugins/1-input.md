# Input

Load a spectral image before selecting references. The plugin processes one time
point and Z plane at a time.

The plugin remembers the last successfully loaded image folder across sessions.
Image and spectrum file dialogs start there; **Reset All** keeps this preference.

1. Click **Browse image...** (tooltip: **Import your desired image**) and select
   a CZI or OME-TIFF file. The image
   summary displays its axes and available metadata.
2. If present, use the **Time** and **Z plane** sliders to select a plane.
   Positions are zero-based. Changing the plane clears the current reference
   and fit results.
3. In napari's **Layers** panel on the left, click **Raw spectral image** so
   the layer is selected/highlighted, and make sure it is visible. The bottom
   **Wavelength** slider controls the raw image only while that layer is
   selected. Use it to inspect detector channels; when wavelength metadata is
   missing, channels are identified by index.
4. Continue to [Spectra selection](2-spectra-selection.md).

To use example data instead, choose **Low**, **Medium** or **High** in
**COS-7 example SNR** and click **Load example data**. This downloads the
mixed acquisition from BioImage Archive
accession S-BIAD4448 (about 145 MB per condition). Downloads run in the
background and completed files are cached for reuse. Single-stain references
are not downloaded. The example then opens through the usual image loader.
The image browser and example-data controls are separated by a divider.

**Reset All**, at the top of the plugin, clears all viewer layers and restores
the startup state.
