# Backend API overview

Expand a topic to find the main functions and classes for that task.
Each topic page covers arguments, array layouts, and examples in more detail.

<details markdown="1">
<summary>Image I/O · Open or load a spectral image</summary>

- [Image I/O reference](image-io-api.md)
- [`open_image()`](image-io-api.md#lumic.io.open_image) inspects metadata and lets you read selected time points or Z planes.
- [`load_image()`](image-io-api.md#lumic.io.load_image) reads pixels into a `SpectralImage`.
- [`SpectralImageReader`](image-io-api.md#lumic.readers.SpectralImageReader) describes the interface shared by CZI and OME-TIFF readers.
- [`download_cos7_data()`](image-io-api.md#lumic.data.download_cos7_data) fetches the example acquisitions.

</details>

<details markdown="1">
<summary>Image containers · Slice, display, reshape, and save images</summary>

- [Image containers reference](image-containers-api.md)
- [`SpectralImage`](image-containers-api.md#lumic.image.SpectralImage) keeps a loaded image, its axes, and detector and spatial metadata together.
    - [`slice()`](image-containers-api.md#lumic.image.SpectralImage.slice) selects a plane from a loaded image.
    - [`to_rgb()`](image-containers-api.md#lumic.image.SpectralImage.to_rgb) creates an RGB preview.
    - [`unfold()`](image-containers-api.md#lumic.image.SpectralImage.unfold) prepares a pixel-by-channel matrix.
    - [`save_ome_tiff()`](image-containers-api.md#lumic.image.SpectralImage.save_ome_tiff) exports an image with its metadata.
- [`UnfoldedImage`](image-containers-api.md#lumic.image.UnfoldedImage) restores fitted spectra or component maps to the source image layout.

</details>

<details markdown="1">
<summary>Spectra · Prepare and reuse reference signatures</summary>

- [Spectra reference](spectra-api.md)
- [`Spectrum`](spectra-api.md#lumic.spectra.Spectrum) holds a single signature with optional wavelength coordinates.
- [`SpectrumCollection`](spectra-api.md#lumic.spectra.SpectrumCollection) holds several component spectra on a shared grid.
    - [`from_spectra()`](spectra-api.md#lumic.spectra.SpectrumCollection.from_spectra) groups aligned signatures.
    - [`save_json()`](spectra-api.md#lumic.spectra.SpectrumCollection.save_json) and [`load_json()`](spectra-api.md#lumic.spectra.SpectrumCollection.load_json) exchange reusable reference sets.

</details>

<details markdown="1">
<summary>Unmixing and spectral selection · Fit components and choose starting spectra</summary>

- [Unmixing and spectral selection reference](unmixing-api.md)
- [`positive_least_squares()`](unmixing-api.md#lumic.unmixing.positive_least_squares) fits abundances with fixed reference spectra.
- [`mcr_als()`](unmixing-api.md#lumic.unmixing.mcr_als) refines abundances and spectra together.
- [`select_analysis_pixels()`](unmixing-api.md#lumic.utils.select_analysis_pixels) builds a pool of usable pixels while retaining their original indices.
- [`essential_spectral_pixels()`](unmixing-api.md#lumic.uniqueness.essential_spectral_pixels) and [`unique_profiles()`](unmixing-api.md#lumic.uniqueness.unique_profiles) help select starting spectra.
- [`spectral_uniqueness_map()`](unmixing-api.md#lumic.uniqueness.spectral_uniqueness_map) maps spectral distinctness across an image.

</details>

<details markdown="1">
<summary>Simulation · Generate detector images with known components</summary>

- [Simulation reference](simulation-api.md)
- [`load_offline_emission_spectrum()`](simulation-api.md#lumic.simulation.load_offline_emission_spectrum) loads a bundled FPbase curve.
- [`SpectralDetector`](simulation-api.md#lumic.simulation.SpectralDetector) integrates curves over detector bands.
- [`simulate_linear_mixture()`](simulation-api.md#lumic.simulation.simulate_linear_mixture) mixes aligned component spectra into a clean image.
- [`simulate_noisy_image()`](simulation-api.md#lumic.simulation.simulate_noisy_image) samples emission curves, mixes components, and adds count noise.

</details>

<details markdown="1">
<summary>Evaluation · Inspect fit errors and measure recovery</summary>

- [Evaluation reference](evaluation-api.md)
- **Without ground truth**
    - [`reconstruction_error_maps()`](evaluation-api.md#lumic.evaluation.reconstruction_error_maps) shows where the prediction differs from the measured image.
    - [`residual_pca_from_residuals()`](evaluation-api.md#lumic.evaluation.residual_pca_from_residuals) gives spatial scores and spectral loadings for patterns in the residuals.
    - [`compare_spectra()`](evaluation-api.md#lumic.evaluation.compare_spectra) compares spectral shapes between fits.
- **With known ground truth**
    - [`evaluate_factors()`](evaluation-api.md#lumic.evaluation.evaluate_factors) summarizes how well the fit recovers the known spectra and abundances.
    - [`concentration_error_maps()`](evaluation-api.md#lumic.evaluation.concentration_error_maps) maps the error for each recovered component.

</details>

## Source code for contributors

<details markdown="1">
<summary>Browse implementation modules on GitHub</summary>

- Image readers and format adapters: [`readers.py`][readers-source] and [`io.py`][io-source].
- Spatial simulation: [scene geometry][scene-source], [sampling and rendering][render-source], and [point spread functions][psf-source].
- Backend evaluation: [evaluation modules][evaluation-source].

</details>

[readers-source]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/readers.py
[io-source]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/io.py
[scene-source]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/simulation/scene.py
[render-source]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/simulation/render.py
[psf-source]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/simulation/psf.py
[evaluation-source]: https://github.com/CCI-GU-Sweden/lumic/tree/main/Python/lumic/evaluation
