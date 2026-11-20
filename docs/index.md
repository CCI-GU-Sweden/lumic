# Lumic

Lumic separates overlapping fluorescence signals in spectral microscopy
images. The Python backend reads ZEISS CZI and OME-TIFF, estimates the
contribution of each fluorescent component, and exports fitted results for
further analysis.

[Start with the backend quickstart](backend/quickstart.md) to install Lumic and
unmix a simulated pixel using bundled reference spectra. You can run the
example with the published package and no microscope files.

[Load spectral images](backend/load-images.md) to inspect a CZI or OME-TIFF,
select a Z plane, and understand the metadata Lumic reads.

[Unmix a real experimental image (COS-7)](backend/cos7-workflow.md) to run the complete
workflow with a public BioImage Archive acquisition.

[Simulate a spectral image](backend/simulate-image.md) to build a mixture with
known component maps and see how shot noise changes the detector signal.

[Save images and spectra](backend/save-results.md) to reload abundance maps,
reconstruct the fitted image, and reuse optimized spectra.

[Browse the examples](examples.md) to find runnable Python scripts and Jupyter
notebooks for experimental and simulated data.
