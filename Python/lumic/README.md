# Lumic backend

Lumic is a Python toolkit for spectral unmixing in fluorescence microscopy.
It reads CZI and OME-TIFF, fits component contributions from known reference
spectra or refines spectra with MCR-ALS, and exports quantitative results.
The backend runs without a graphical interface.

## Install

Use Python 3.12. This guide uses the `0.1.0rc3` prerelease.

```sh
python -m pip install "lumic==0.1.0rc3"
```

To install the source version from the repository root:

```sh
python -m pip install ./Python/lumic
```

The default install includes the bundled FPbase emission spectra. Optional
extras enable Matplotlib previews (`visualization`), online spectrum retrieval
(`fpbase`), notebook and plotting tools (`examples`), or pytest (`test`).
Install all four with `python -m pip install './Python/lumic[all]'`.

## Unmix a simulated pixel offline

Sample two bundled spectra through the example 32-channel detector, then
estimate how much of each component was mixed into the pixel. The fit uses
known references, so it works without microscopy files or internet access.

```python
import numpy as np

from lumic import positive_least_squares
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum

detector = SpectralDetector.from_profile("980_32ch")
references = np.array([
    detector.sample(*load_offline_emission_spectrum(name))
    for name in ("EGFP", "mCherry")
])
references /= references.sum(axis=1, keepdims=True)

# One pixel with contributions of 70 and 30 in total detected counts.
mixed_pixel = np.array([[70.0, 30.0]]) @ references
estimated = positive_least_squares(mixed_pixel, references, solver="auto")
print(np.round(estimated[0], 6))  # [70. 30.]
```

## Download and open a real spectral image

Download the high-SNR mixed COS-7 acquisition (about 145 MB) from
BioImage Archive. Later runs reuse the local file.
[Dataset S-BIAD4448 (CC0)](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4448)

```python
from lumic.data import download_cos7_data
from lumic.io import load_image

path, = download_cos7_data(progress=True)
image = load_image(path)
print(image.axes, image.data.shape)  # CYX (32, 1504, 1504)
```

Use `snr="medium"`, `snr="low"`, or `snr="all"` to select other acquisitions.
Set `include_references=True` to also fetch the single-stain reference files.

## Explore the examples

The examples live in the repository’s
[`Python/examples/`](https://github.com/CCI-GU-Sweden/lumic/tree/main/Python/examples)
directory. From the repository root, install the examples extra, switch to
`Python/`, and run the full workflow:

```sh
python -m pip install './Python/lumic[examples]'
cd Python
python -m examples.spectral_image_workflow
```

This walkthrough creates a three-protein mixture with shot noise, estimates
the component maps using known spectra, and plots the residual. It exports the
measurement, fitted reconstruction and abundance maps as OME-TIFF, then
reconstructs the fitted signal from the reloaded maps and their saved spectra.
See the [comparison figure](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/data/visual-check-spectral-image-workflow.png)
for the expected output.

| What you want to try | Command from `Python/` |
| --- | --- |
| Inspect a CZI or OME-TIFF header before loading pixels. Set `LOAD_PIXELS = True` in the script to read the selected plane. | `python -m examples.load_spectral_image` |
| Render the COS-7 CZI in RGB and copy a plane. Set the input path to try a Z stack. | `python -m examples.render_spectral_rgb` |
| Simulate overlapping fluorescent structures and compare the recovered spectra and maps with the known inputs. | `python -m examples.visualize_overlapping_circles` |
| See how lower photon counts affect the relative noise in a spectral mixture. | `python -m examples.visualize_poisson_noise_levels` |
| Inspect spectral distinctness in an image and in the projected spectral space. | `python -m examples.visualize_spectral_uniqueness` |

These examples use bundled spectra or sample images and run offline once the
environment is installed. The separate `examples.simulate_fpbase_mixtures`
script fetches live spectra and needs internet access and the `fpbase` extra.

## Go deeper with the backend guides

- **Work with microscope images.** Inspect CZI or OME-TIFF metadata before
  loading pixels, select a time point or Z plane, and keep track of the image
  axes as you unfold and refold the data. Save abundance maps as OME-TIFF with
  the exact fitted spectra so you can reconstruct the fitted signal later.
  [Image I/O guide](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/IO_README.md)
- **Build a controlled simulation.** Sample reference spectra through a
  detector model, set the component contributions, and render spatial
  shapes with optical blur and noise. Keep the original spectra and
  abundance maps as ground truth for comparing fits.
  [Simulation guide](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/simulation/README.md)
- **Evaluate a fit.** Align components before comparing spectra and
  abundance maps across runs. When ground truth is unavailable, use signed
  residual PCA and spatial or spectral correlations to look for patterns
  the fit leaves unexplained.
  [Evaluation guide](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/evaluation/README.md)

## License and data attribution

Lumic is distributed under the MIT license. The bundled spectra carry their
FPbase source and attribution notice alongside the CSV files.
