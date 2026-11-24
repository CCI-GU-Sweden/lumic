# Lumic backend changelog

## 0.1.0rc4

- Add `SpectrumCollection` for ordered component spectra on a shared sample
  axis, with optional detector bands and provenance for each component and the
  collection as a whole.
- Save and load collections as versioned JSON while preserving sample order
  and numeric dtypes. Store the shared wavelength coordinates and detector
  bands once per file.
- Add a backend guide and offline example that prepare bundled FPbase spectra
  on the LSM 980 detector grid, save and reload the collection, and use it for
  unmixing.
- Publish the backend API reference and link to the hosted guides from PyPI.
- Expose `essential_spectral_pixels` with the corrected spelling. Existing
  callers can still use `essential_spectral_pixles`.

## 0.1.0rc3

- Verify release packaging and PyPI publishing from the new Lumic repository.
- Update repository links in the package metadata and README.

## 0.1.0rc2

- Add `lumic.data.download_cos7_data()` to retrieve the public COS-7 spectral
  acquisitions from BioImage Archive accession S-BIAD4448. Fetch the high-SNR
  mixed image by default or select other SNR conditions and single-stain
  reference acquisitions.
- Reuse complete local downloads and optionally show per-file progress.
- Update the repository's minimal workflow notebook to use the public
  acquisition by default.

## 0.1.0rc1

First release candidate for the Lumic backend. The package targets Python 3.12,
with wheel installation checks in CI on Linux and Windows.

- Read CZI and OME-TIFF with explicit axis labels and preserve available
  detector metadata and spatial calibration.
- Fit component contributions from known spectra or refine spectra with
  MCR-ALS. Inspect the fit with residual and spectral diagnostics.
- Build offline simulations using the bundled FPbase spectra, detector
  profiles, spatial scenes, blur and noise.
- Save abundance maps and their fitted spectra in lossless OME-TIFF and
  reconstruct the fitted detector signal after reload.

See the backend README for installation instructions and an offline quickstart.
