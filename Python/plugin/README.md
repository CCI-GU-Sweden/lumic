# napari-lumic-cci: A spectral unmixing plugin

A napari plugin for spectral unmixing with the LUMIC backend.

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

1. **Input:** Load an OME-TIFF or CZI spectral image and select a plane, or load a COS-7 example with low, medium or high SNR from BioImage Archive.
2. **Spectra selection:** Choose automatic or manual references, or load a Lumic spectrum JSON file. Automatic selection finds distinctive spectra.
3. **Unmixing:** Optimize automatic or JSON references and calculate abundance maps; manual references remain fixed. Save all spectra together in one JSON file, and save maps as OME-TIFF.
4. **Evaluation:** Examine residual variation and its spatial patterns using principal component analysis.

See the [plugin Quickstart](../../docs/plugins/overview.md) to try the project’s
COS-7 example using **Load example data**, and follow the linked pages for the
controls, JSON spectra workflow and residual evaluation in greater detail.

## Run plugin tests

With the backend development environment active, install both local packages
and the Qt binding with
`python -m pip install -e ./Python/lumic -e ./Python/plugin "napari[pyqt6]"`.
Restart napari after updating these packages so it uses the current backend,
including the `Spectrum.save_json` and `Spectrum.load_json` methods.
From the `Python` directory, run `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_plugin_*.py`.
