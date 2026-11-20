# napari-lumic-cci: A spectral unmixing plugin

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


## Workflow

Follow the tabs in order:

1. [Input](1-input.md): load an OME-TIFF or CZI image and select a plane.
2. [Spectra selection](2-spectra-selection.md): find distinctive spectra automatically or select reference pixels manually.
3. [Unmixing](3-unmixing.md): fit abundance maps and save the results.
4. [Evaluation](4-evaluation.md): examine residual variation and spatial patterns.
