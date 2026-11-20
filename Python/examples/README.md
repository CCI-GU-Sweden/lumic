# Lumic examples

The [documentation site](../../docs/index.md) groups the walkthroughs by
experimental and simulated data. Use this guide to find the runnable scripts
and notebooks in the checkout.

## Run an example

Use Python 3.12. From the repository root, install the backend and the example
dependencies into your active environment:

```bash
python -m pip install -e './Python/lumic[examples]'
cd Python
python -m examples.spectral_image_workflow
```

The rest of the scripts use the same `python -m examples.<script_name>`
pattern from `Python/`. Open notebooks from `examples/notebooks/` in Jupyter
and select the environment where you installed Lumic.

The experimental image examples download the same high-SNR COS-7 CZI from
BioImage Archive on first use (about 145 MB) and reuse the cache afterward.
The archive contains single-plane acquisitions. The CZI and RGB scripts
include an optional bundled Z stack to demonstrate plane selection.

## Choose a starting point

| Task | Example | Input |
| --- | --- | --- |
| Inspect a file before loading pixels | [load_spectral_image.py](load_spectral_image.py) | Public COS-7 CZI, or your own CZI or OME-TIFF |
| Compare detector pixels and metadata across formats | [minimal_load](notebooks/minimal_load.ipynb), [minimal_metadata](notebooks/minimal_metadata.ipynb) | Public COS-7 CZI and a temporary OME-TIFF export |
| Preview a spectral image in RGB | [render_spectral_rgb.py](render_spectral_rgb.py) | Public COS-7 CZI, with an optional bundled Z stack |
| Unmix an experimental image | [minimal_example](notebooks/minimal_example.ipynb) | Public high-SNR COS-7 CZI, downloaded on first run and cached locally |
| Import a ZEN reference spectrum | [xmlZEN_load](notebooks/xmlZEN_load.ipynb) | Bundled ZEN XML export |
| Discover distinct spectra with a convex hull | [convexhull_minimal_example](notebooks/convexhull_minimal_example.ipynb) | Synthetic spectra |
| Simulate, unmix, and export a ternary image | [spectral_image_workflow.py](spectral_image_workflow.py), [export_spectral_ome_tiff.py](export_spectral_ome_tiff.py) | Offline FPbase emission curves bundled with Lumic |
| Explore spatial mixing and blur | [visualize_spatial_scene.py](visualize_spatial_scene.py), [visualize_overlapping_circles.py](visualize_overlapping_circles.py) | Synthetic scenes using offline spectra |
| Inspect noise and spectral uniqueness | [visualize_noise_levels.py](visualize_noise_levels.py), [visualize_poisson_noise_levels.py](visualize_poisson_noise_levels.py), [visualize_spectral_uniqueness.py](visualize_spectral_uniqueness.py) | Simulated detector images using offline spectra |

For format-specific reading, see [load_spectral_czi.py](load_spectral_czi.py)
or [load_spectral_ome_tiff.py](load_spectral_ome_tiff.py), which creates a
temporary export from the public CZI and reloads it. To download COS-7
images separately, run [download_cos7_data.py](download_cos7_data.py) as a
module with `--help` to see the available SNR levels and reference images.

[simulate_fpbase_mixtures.py](simulate_fpbase_mixtures.py),
[visualize_three_fluorophore_mixture.py](visualize_three_fluorophore_mixture.py),
and [visualize_ternary_mixture.py](visualize_ternary_mixture.py) query live
FPbase. Install `.[examples,fpbase]` from `Python/lumic/` and use an internet
connection for these scripts.

Generated exports under `examples/output/` are ignored by Git. Some
visualization scripts write PNGs to `Python/data/`; check the output setting
before rerunning if you want to keep an existing preview.

For the reader template, see [Adding an image reader](../../docs/contributing/adding-an-image-reader.md).
For larger experiments and saved reference results, see the
[evaluation guide](../evaluations/README.md).
