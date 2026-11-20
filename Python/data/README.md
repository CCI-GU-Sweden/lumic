# Data supplied with LUMIC

This directory contains small real-image validation fixtures, saved emission
spectra and synthetic visual examples. This index describes the shareable
repository data; it does not imply that every folder on a developer's machine
is included on GitHub.

## Start here

| Data | What it is for | Details |
| --- | --- | --- |
| [io-data/](io-data/README.md) | Real CZI files and paired OME-TIFF exports for checking pixels, dimensions and metadata | LSM 880/980 spectral-detector images, LSM 900 sequential window scans from a CD7 system, and four LSM 980 ZEN exports |
| [Public COS-7 acquisitions](../../docs/backend/load-images.md#download-the-public-cos-7-example) | Experimental spectral images at three SNR conditions | Download from BioImage Archive with `lumic.data` |
| [FPbase spectra](../lumic/simulation/fpbase_data/README.md) | Fixed offline emission curves for simulations | EBFP2, EGFP, mCherry and mNeptune CSVs, with source IDs and attribution |
| [Argo_405.xml](Argo_405.xml) | Example ZEN linear-unmixing spectrum export | [xmlZEN_load.ipynb](../examples/notebooks/xmlZEN_load.ipynb) loads it into a Lumic `Spectrum` for inspection. Instrument response calibration requires separate data. |
| `visual-check-*.png` | Synthetic figures for human inspection | See the preview index below; these are display images, not quantitative spectral cubes |

For simulation building blocks and runnable code, see
[lumic/simulation/README.md](../lumic/simulation/README.md). For fitted-factor
and residual diagnostics, see [lumic/evaluation/README.md](../lumic/evaluation/README.md).

## Real microscope validation files

The I/O collection is organized by acquisition configuration:

```text
io-data/
├── README.md
├── manifest.json
├── 880/          LSM 880 spectral-detector acquisitions
├── 980/          LSM 980 acquisitions and the original ZEN reference files
└── cd7-900/      LSM 900 sequential window scans, acquired on CD7
```

There are 30 CZI files, approximately 45 MiB in total, with single planes,
small Z-stacks, time series and mosaics. Complete pixel regression checks now cover all 30 files, including full T/Z
arrays and composed single-scene mosaics. These checks do not establish
plane selection or stitching support. The manifest records source checksums,
dimensions, detection bounds, physical sampling and selected decoded pixels;
it distinguishes CZI-derived regression snapshots from independently recorded
ZEN references.

The original `io-example-lsm980/` collection has moved into `io-data/980/`.
Its two CZIs, label screenshots/text and pixel-value reference file retain
their original filenames and contents. Example and test paths follow the new
location.

**LSM 900 spectral scans need a different interpretation.** This strategy is
a property of the LSM 900 scan head, not exclusive to CD7 integration. Our
fixtures come from a CD7 platform, which the directory name records. The image is
acquired repeatedly with changing, overlapping detection windows; the filename
step is not a conventional spectral channel width. Preserve detection-band
bounds and acquisition order. A list of channel midpoints alone is not a
validated spectral axis for these measurements.

Read the [fixture guide](io-data/README.md) before using these files. They
validate I/O and metadata behavior, not the ability to recover pure spectra
from each sample. OME-TIFF interchange and broader instrument/configuration
support remain separate work.

## Saved fluorophore spectra

The four [FPbase CSVs](../lumic/simulation/fpbase_data/README.md) contain source wavelength samples and
relative emission values. They have not been integrated into detector channels
or normalized to a count scale. Simulations choose the detector, normalization
and expected component counts separately. The folder README records retrieval
details and attribution; retain those references when reusing the curves.

## Synthetic preview index

| Figure | Shows | Generator under `Python/examples/` |
| --- | --- | --- |
| [Three-component mixture](visual-check-bfp-egfp-mcherry.png) | Mixture-grid spectral RGB | [visualize_three_fluorophore_mixture.py](../examples/visualize_three_fluorophore_mixture.py) |
| [Ternary mixture](visual-check-ternary-bfp-egfp-mcherry.png) | Composition triangle and spectral RGB | [visualize_ternary_mixture.py](../examples/visualize_ternary_mixture.py) |
| [Spatial scene](visual-check-spatial-scene.png) | Shape contributions before and after blur | [visualize_spatial_scene.py](../examples/visualize_spatial_scene.py) |
| [Overlapping circles](visual-check-overlapping-circles.png) | Three objects, blur and spectral RGB | [visualize_overlapping_circles.py](../examples/visualize_overlapping_circles.py) |
| [Circle recovery](visual-check-overlapping-circles-recovery.png) | Blind-recovery comparison with known simulation factors | [visualize_overlapping_circles.py](../examples/visualize_overlapping_circles.py) |
| [Noise levels](visual-check-noise-levels.png) | Image appearance as expected counts decrease | [visualize_noise_levels.py](../examples/visualize_noise_levels.py) |
| [Poisson noise curves](visual-check-poisson-noise-curves.png) | Expected and noisy spectral measurements | [visualize_poisson_noise_levels.py](../examples/visualize_poisson_noise_levels.py) |
| [Spectral uniqueness](visual-check-spectral-uniqueness.png) | Distinctness maps and direction colors in projected spectral space | [visualize_spectral_uniqueness.py](../examples/visualize_spectral_uniqueness.py) |
| [SpectralImage workflow](visual-check-spectral-image-workflow.png) | Offline ternary simulation, exact pixel restoration, fitted component maps and reconstruction residuals with shot noise | [spectral_image_workflow.py](../examples/spectral_image_workflow.py) |
| [RGB Z-stack preview](visual-check-rgb-stack.png) | Shared RGB scaling across three Z planes from a bundled CZI fixture; display only | [render_spectral_rgb.py](../examples/render_spectral_rgb.py) |

Run example modules from `Python/`, for example:

```bash
python -m examples.visualize_spatial_scene
```

The two older mixture-grid/ternary generators query live FPbase and use
`mTagBFP2`; the bundled blue spectrum is `EBFP2`. Other examples use the saved
offline curves. Consult each script's imports/settings before assuming that
a preview can be regenerated offline or uses exactly the same source spectra.
Display normalization and RGB conversion are for viewing, not quantitative
unmixing inputs. Generating an example may replace its default PNG.

## Evaluation results and local files

Larger synthetic recovery and noise studies are indexed in
[evaluations/README.md](../evaluations/README.md). Running the evaluations
creates figures and data in the ignored `Python/evaluations/results/`
directory.

When adding a dataset, document its origin, intended use, axis/wavelength/count
conventions and reference information. Keep real measurements separate from
simulation truth and display previews, and update the relevant data index.
