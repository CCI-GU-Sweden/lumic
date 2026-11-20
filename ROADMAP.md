# Lumic roadmap

Lumic is a Python toolkit for spectral unmixing in fluorescence microscopy.
It helps researchers separate overlapping signals, inspect what a fitted
model leaves unexplained, and save results they can reuse. Its longer-term
goal is to make unmixing methods comparable on shared microscopy data,
simulations, and evaluation metrics.

## What you can do today

- Load ZEISS CZI and OME-TIFF images, inspect their detector metadata, and
  select time points or Z planes using explicit axis labels.
- Estimate abundance maps from known spectra, or select starting spectra
  from the data and refine them with MCR-ALS.
- Inspect reconstruction errors with residual maps, residual PCA, and
  spatial and spectral correlations.
- Simulate spectral mixtures with known component maps, detector sampling,
  blur, and noise. Use the known inputs to measure recovery accuracy.
- Export quantitative images and fitted abundance maps as OME-TIFF. Save
  the fitted spectra alongside the maps so the reconstruction can be
  regenerated after loading.

Start with the [documentation](docs/index.md) or try the
[real-data COS-7 workflow](docs/backend/cos7-workflow.md).

## Near-term priorities

- **Make the workflow easier to adopt.** Expand practical examples,
  document the public API and supported image layouts, and guide napari
  users through a complete unmixing run.
- **Improve large-image fitting.** Measure runtime and memory on real
  acquisitions and evaluate when batching, optional workers, or reading
  smaller image regions helps.
- **Improve interpretation of weak signals.** Test how noise, spectral
  overlap, and background affect recovery. Use controlled simulations and
  public experimental data to document when residual diagnostics are
  informative.

## Longer-term research direction

We want researchers to use the results of an unmixing run to decide
which additional measurements are worth taking. A spatial pattern in
the residuals can suggest where to look. Different excitation settings
or a reference acquisition may help distinguish competing explanations.
The researcher chooses which measurements to pursue. Future work will
explore how to connect this workflow to microscope acquisition.
