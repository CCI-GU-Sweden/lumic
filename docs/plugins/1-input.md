# Input

Load a spectral image before selecting references. The plugin processes one time
point and Z plane at a time.

1. Click **Browse image...** and select a CZI or OME-TIFF file. The image
   summary displays its axes and available metadata.
2. If present, use the **Time** and **Z plane** sliders to select a plane.
   Positions are zero-based. Changing the plane clears the current reference
   and fit results.
3. Select and show the raw-data layer in napari, then use the bottom
   **Wavelength** slider to inspect detector channels. When wavelength metadata
   is missing, channels are identified by index.
4. Continue to [Spectra selection](2-spectra-selection.md).

**Reset All**, at the top of the plugin, clears all viewer layers and restores
the startup state.
