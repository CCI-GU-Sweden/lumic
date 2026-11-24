# Examples

Find a Python script or Jupyter notebook by task. Source links open the files on
GitHub. Walkthrough links explain the same or a related workflow in the docs.

## Run an example

From the repository root, install Lumic and its example dependencies in a
Python 3.12 environment. Then run the scripts as modules from `Python/`.

```bash
python -m pip install -e './Python/lumic[examples]'
cd Python
python -m examples.spectral_image_workflow
```

Open the notebooks from `Python/examples/notebooks/` in Jupyter and select
the kernel for the same environment. If you're new to Jupyter, see
[Run your first notebook](backend/cos7-workflow.md#run-your-first-notebook).
Examples marked **COS-7 download** fetch the
public high-SNR image on first use (about 145 MB) and reuse the local copy
afterward. Examples marked **live FPbase** need internet access and the
`fpbase` extra. To install both extras from the repository root, use
`python -m pip install -e './Python/lumic[examples,fpbase]'`.

## Load and inspect real data

| Example | What it shows | Walkthrough |
| --- | --- | --- |
| [Download COS-7 data (.py)][download] | Download selected acquisitions or reference images from BioImage Archive. | [Load spectral images](backend/load-images.md#download-the-public-cos-7-example) |
| [Inspect and load spectral images (.py)][load-image] | Inspect a CZI or OME-TIFF header before loading pixels. **COS-7 download** by default. | [Load spectral images](backend/load-images.md) |
| [Load a CZI (.py)][load-czi] | Read pixels, inspect metadata, and select a plane. **COS-7 download** by default. | [Load spectral images](backend/load-images.md) |
| [Load an OME-TIFF (.py)][load-tiff] | Export the COS-7 CZI to a temporary OME-TIFF and load it. **COS-7 download**. | [Load spectral images](backend/load-images.md) |
| [Render a spectral image in RGB (.py)][rgb] | Preview a CZI using shared scaling and copy a plane from a Z stack. **COS-7 download** by default. | [Load spectral images](backend/load-images.md) |
| [Compare image loading (.ipynb)][nb-load] | Compare the detector pixels loaded from a CZI and its OME-TIFF export. **COS-7 download**. | [Load spectral images](backend/load-images.md) |
| [Inspect image metadata (.ipynb)][nb-metadata] | Compare the metadata Lumic reads from a CZI and its OME-TIFF export. **COS-7 download**. | [What metadata does Lumic return?](backend/load-images.md#what-metadata-does-lumic-return) |
| [Import a ZEN spectrum (.ipynb)][nb-zen] | Read a bundled ZEN XML spectrum into a Lumic `Spectrum`. | |

## Simulate images

| Example | What it shows | Walkthrough |
| --- | --- | --- |
| [Simulate a binary mixture (.py)][binary] | Make a small, noiseless EGFP and mCherry image. **Live FPbase**. | [Simulate a spectral image](backend/simulate-image.md) |
| [Render a mixture of three fluorophores (.py)][three] | Make an RGB preview of a 20 × 20 mixture. **Live FPbase**. | [Simulate a spectral image](backend/simulate-image.md) |
| [Render a ternary mixture (.py)][ternary] | Render mixtures inside a triangle with one pure fluorophore at each vertex. **Live FPbase**. | [Unmix a ternary image](backend/unmix-ternary-image.md) |
| [Render a spatial scene (.py)][spatial] | Compare the same fluorescent shapes with and without approximate optical blur. | [Simulate a spectral image](backend/simulate-image.md) |
| [Compare noise in images (.py)][noise] | View the same scene at several noise levels. | [Add shot noise](backend/simulate-image.md#add-shot-noise) |
| [Compare Poisson noise in spectra (.py)][poisson] | Plot how relative shot noise increases as the expected counts fall. | [Add shot noise](backend/simulate-image.md#add-shot-noise) |

## Unmix, evaluate, and save results

| Example | What it shows | Walkthrough |
| --- | --- | --- |
| [Unmix a COS-7 image (.ipynb)][nb-cos7] | Unmix a real spectral acquisition and inspect the fitted results. **COS-7 download**. | [Unmix a real experimental image](backend/cos7-workflow.md) |
| [Explore the convex hull (.ipynb)][nb-hull] | See how the hull finds distinct spectra in a synthetic mixture. | [Estimate unknown spectra](backend/estimate-unknown-spectra.md) |
| [Map spectral uniqueness (.py)][uniqueness] | Compare spectral distinctness in image coordinates and in the hull projection. | [Spectral uniqueness maps](backend/spectral-uniqueness-maps.md) |
| [Unmix overlapping circles (.py)][circles] | Estimate spectra from a simulated image, then compare the fitted spectra and maps with the known inputs. | [Evaluate simulation recovery](backend/evaluate-simulation-recovery.md) |
| [Run a spectral image workflow (.py)][workflow] | Unfold and unmix a simulation, restore the image layout, then compare and export the fitted results. | [Save images and spectra](backend/save-results.md) |
| [Export spectral and RGB images (.py)][export] | Save a simulation as OME-TIFF, verify its round trip, and export an RGB preview. | [Save images and spectra](backend/save-results.md) |

## Extend Lumic

| Example | What it shows | Walkthrough |
| --- | --- | --- |
| [Image reader template (.py)][reader] | Implement the reader interface for an additional image format. | [Adding an image reader](contributing/adding-an-image-reader.md) |

[download]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/download_cos7_data.py
[load-image]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/load_spectral_image.py
[load-czi]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/load_spectral_czi.py
[load-tiff]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/load_spectral_ome_tiff.py
[rgb]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/render_spectral_rgb.py
[nb-load]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/notebooks/minimal_load.ipynb
[nb-metadata]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/notebooks/minimal_metadata.ipynb
[nb-zen]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/notebooks/xmlZEN_load.ipynb
[binary]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/simulate_fpbase_mixtures.py
[three]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_three_fluorophore_mixture.py
[ternary]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_ternary_mixture.py
[spatial]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_spatial_scene.py
[noise]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_noise_levels.py
[poisson]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_poisson_noise_levels.py
[nb-cos7]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/notebooks/minimal_example.ipynb
[nb-hull]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/notebooks/convexhull_minimal_example.ipynb
[uniqueness]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_spectral_uniqueness.py
[circles]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/visualize_overlapping_circles.py
[workflow]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/spectral_image_workflow.py
[export]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/export_spectral_ome_tiff.py
[reader]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/reader_template.py
