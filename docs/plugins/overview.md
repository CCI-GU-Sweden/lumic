# Quickstart

The napari-lumic-cci plugin provides a graphical workflow for loading spectral
microscopy images, selecting reference spectra, unmixing the image, and examining
residuals with the Lumic backend.

## Installation

In a fresh Python 3.12 environment, install the plugin from PyPI, then install
the Qt binding needed to launch napari:

```bash
python -m pip install napari-lumic-cci
python -m pip install "napari[pyqt6]"
```

The plugin installation also installs the Lumic backend and napari. The second
command supplies the PyQt6 binding for napari's graphical interface.

## Usage

Launch napari with `python -m napari`, then open **Plugins → LUMIC CCI-GU →
Spectral Unmixing**.

For an introduction to the viewer, see the [napari documentation](https://napari.org/stable/index.html) and [napari getting started guide](https://napari.org/stable/getting_started/start_index.html).

## Workflow details

1. [Input](1-input.md): browse a CZI or OME-TIFF image, or load example data,
   and select a time point and Z plane.
2. [Spectra selection](2-spectra-selection.md): calculate automatic references,
   accept and confirm any number of manual reference pixels, or use **Load JSON**
   to reuse saved spectra. **Save initial reference spectra...** exports the
   initial references in one JSON file.
3. [Unmixing](3-unmixing.md): optimize automatic or JSON references and fit
   abundance maps. Manual references remain fixed at their selected values.
4. [Evaluation](4-evaluation.md): inspect residual score maps and loading curves,
   with PC 1 explaining the largest amount of residual variance.

## Try the example data

You can test the plugin yourself with the project's COS-7 example data, also
used in the repository's [COS-7 example notebook](../backend/cos7-workflow.md).
No local image file is required:

1. In **Input**, choose **Low**, **Medium** or **High** from **COS-7 example SNR**,
   then click **Load example data**. The plugin downloads the selected mixed
   acquisition from BioImage Archive and opens it; previously downloaded files
   are reused. For your own data, click **Browse image...** instead.
2. In **Spectra selection**, choose **Automatic** under **References**, set
   **Number of components**, then click **Calculate initial reference spectra**.
   Inspect the reference curves, selected pixels and initial distance map.
3. In **Unmixing**, click **Optimize spectra and calculate abundance maps**.
   Inspect the abundance maps and signed residual. Use **Save optimised spectra...**
   to save every fitted component in one JSON file, or
   **Save abundance maps as OME-TIFF...** to save quantitative maps.
4. In **Evaluation**, click **Calculate residual PCA** to explore the variation
   left unexplained by the fit.
