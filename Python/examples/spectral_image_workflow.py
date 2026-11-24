"""Create, unfold, unmix and restore a spectral image using offline FPbase data.

Edit the settings below, then run from Python/:
    python -m examples.spectral_image_workflow
"""
from pathlib import Path

import numpy as np

from lumic import SpectralImage, SpectrumCollection, positive_least_squares
from lumic.io import load_image
from lumic.evaluation import reconstruction_error_maps
from lumic.simulation import (
    SpectralDetector, add_confocal_noise, load_offline_emission_spectrum,
    simulate_linear_mixture,
)
from examples._plot_spectral_image_workflow import save_workflow_figure


# 1. USER SETTINGS
# Choose three proteins from the offline set: EBFP2, EGFP, mCherry, mNeptune.
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
IMAGE_SIZE = 101
TOTAL_COUNTS = 10000.0  # Expected sum across detector channels inside the triangle.
ADD_SHOT_NOISE = True  # Set False to see an almost exact fitted reconstruction.
SEED = 2026  # Reproduce the same noise when rerunning this example.
OUTPUT_PATH = (Path(__file__).resolve().parents[1] / "data" /
               "visual-check-spectral-image-workflow.png")


# Saved files are ignored by Git; reruns replace this example's own outputs.
EXPORT_DIR = Path(__file__).resolve().parent / "output" / "spectral-image-workflow"
COMPRESSION = "deflate"  # Lossless; alternatives: "lzw" or None.


# 2. CREATE A NOISELESS NUMPY IMAGE
# Sample real spectra through our approximate 32-channel LSM 980 detector.
# Normalize each spectrum to sum to one, so abundances are detected counts.
detector = SpectralDetector.from_profile("980_32ch")
spectra = np.array([
    detector.sample(*load_offline_emission_spectrum(name))
    for name in FLUOROPHORES
])
spectra = spectra / spectra.sum(axis=1, keepdims=True)

# A triangle covers all ternary compositions: first protein at the top,
# second at bottom-left, third at bottom-right. Outside pixels are zero.
y, x = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=float) / (IMAGE_SIZE - 1)
first = (0.90 - y) / 0.85
third = (x - 0.05 - 0.45 * first) / 0.90
second = 1.0 - first - third
fractions = np.stack([first, second, third], axis=-1)
inside = np.all(fractions >= -1e-12, axis=-1)
true_abundances = np.where(inside[..., None], np.clip(fractions, 0, 1), 0) * TOTAL_COUNTS

simulation = simulate_linear_mixture(
    true_abundances, spectra, detector.channel_centers,
    fluorophore_names=FLUOROPHORES,
)
numpy_image = simulation.image  # Ordinary NumPy array: (Y, X, channels).


# 3. ADD A SMALL AMOUNT OF SHOT NOISE (OPTIONAL)
# Expected counts control Poisson noise: more counts give smaller relative noise.
# Keep the noiseless simulation above as our ground truth.
if ADD_SHOT_NOISE:
    numpy_image = add_confocal_noise(numpy_image, rng=np.random.default_rng(SEED))


# 4. CREATE A SPECTRALIMAGE FROM THE NUMPY ARRAY
# Axes describe the actual array order. Wavelengths and calibration are optional.
image = SpectralImage(
    data=numpy_image,
    axes="YXC",
    wavelengths_nm=detector.channel_centers,
    detection_bands_nm=np.column_stack([detector.bin_edges[:-1], detector.bin_edges[1:]]),
    pixel_sizes_um={"X": 0.1, "Y": 0.1},
    provenance={"source": "offline FPbase simulation", "shot_noise": ADD_SHOT_NOISE},
)
print(f"Image: axes={image.axes}, shape={image.data.shape}, kind={image.kind}")


# 5. UNFOLD, THEN RESTORE THE SAME PIXELS
# .data is the observations × channels matrix passed to numerical algorithms.
# The unfolded object remembers the original layout for later folding.
unfolded = image.unfold()
restored = unfolded.fold_spectra()
np.testing.assert_array_equal(restored.data, numpy_image)
print(f"Unfolded: {unfolded.data.shape}; restored: {restored.axes}, {restored.data.shape}")
print("Restoration preserves every pixel exactly, including its noise.")


# 6. UNMIX WITH KNOWN SPECTRA, THEN RESTORE COMPONENT MAPS
# This example uses fixed references, not blind discovery or ALS optimization.
# Spectra already sum to one, so keep their scaling with do_norm=False.
abundances = positive_least_squares(unfolded.data, spectra, do_norm=False, solver="auto")
# Save the exact basis paired with these coefficients, without rescaling it.
# The labels describe its existing scale; they do not change any values.
maps = unfolded.fold_abundances(
    abundances, component_names=FLUOROPHORES, spectra=spectra,
    normalization="L1", abundance_units="estimated detected counts",
)
print(f"Abundance maps: axes={maps.axes}, shape={maps.data.shape}, kind={maps.kind}")


# 7. RECONSTRUCT THE DETECTOR SIGNAL AND INSPECT THE RESIDUAL
# Component contributions × spectra predicts all 32 channels at each pixel.
# Both results are SpectralImage objects, with different kinds and axis meanings.
predicted_signal = abundances @ spectra
reconstruction = unfolded.fold_spectra(predicted_signal)
errors = reconstruction_error_maps(restored.data, reconstruction.data)
print(f"Reconstruction: axes={reconstruction.axes}, shape={reconstruction.data.shape}, kind={reconstruction.kind}")
print(f"Largest residual RMS: {errors.rms.max():.3g} counts/channel")
print(f"Largest abundance error vs simulation: {np.abs(maps.data - true_abundances).max():.3g} counts")
# Reconstruction is a model prediction; it does not reproduce every noise
# fluctuation. A small residual does not establish unique blind factor recovery.


# 8. SAVE THE COMPARISON FIGURE
# Plotting lives in an example helper so the image-handling steps remain clear.
save_workflow_figure(
    image, restored, maps, reconstruction, errors,
    inside=inside, expected_counts=TOTAL_COUNTS, shot_noise=ADD_SHOT_NOISE,
    output_path=OUTPUT_PATH,
)
print(f"Saved figure to {OUTPUT_PATH}")


# 9. EXPORT AND RELOAD THE IMAGES AND REFERENCE SPECTRA
# On disk, OME uses C for both detector channels and component maps. Lumic's
# annotation restores K for maps and keeps their optional component spectra.
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
measured_path = image.save_ome_tiff(
    EXPORT_DIR / "measured.ome.tif", compression=COMPRESSION, overwrite=True)
reconstructed_path = reconstruction.save_ome_tiff(
    EXPORT_DIR / "reconstructed.ome.tif", compression=COMPRESSION, overwrite=True)
abundance_path = maps.save_ome_tiff(
    EXPORT_DIR / "abundances.ome.tif", compression=COMPRESSION, overwrite=True)
references_path = EXPORT_DIR / "reference-spectra.json"
SpectrumCollection(
    spectra, image.wavelengths_nm, detection_bands_nm=image.detection_bands_nm,
    names=FLUOROPHORES,
    provenance={"source": "offline FPbase", "detector_profile": detector.name,
                "normalization": "L1"},
).save_json(references_path)

measured_again = load_image(measured_path, t=None, z=None)
reconstructed_again = load_image(reconstructed_path, t=None, z=None)
maps_again = load_image(abundance_path, t=None, z=None)
for original, reloaded in ((image, measured_again),
                           (reconstruction, reconstructed_again), (maps, maps_again)):
    assert reloaded.kind == original.kind and reloaded.axes == original.axes
    np.testing.assert_array_equal(reloaded.data, original.data)
    print(f"Reloaded {reloaded.kind}: {reloaded.axes}, {reloaded.data.shape}")

# For a new image, load the references and check its channel grid in the same
# order before passing references.values to positive_least_squares with do_norm=False.
references = SpectrumCollection.load_json(references_path)
np.testing.assert_array_equal(references.wavelengths_nm, measured_again.wavelengths_nm)
np.testing.assert_array_equal(references.detection_bands_nm, measured_again.detection_bands_nm)
np.testing.assert_array_equal(references.values, spectra)
assert references.names == FLUOROPHORES
print(f"Saved and reloaded reference spectra at {references_path}")

# Only the reloaded maps and saved basis are needed for this prediction.
# It reproduces the fitted signal, not the measured image's random noise.
prediction_from_saved_maps = maps_again.reconstruct()
np.testing.assert_allclose(prediction_from_saved_maps.data, reconstruction.data,
                           rtol=1e-14, atol=1e-12)
print(f"Saved images to {EXPORT_DIR}")
print("Reloaded abundance maps and spectra reproduce the fitted reconstruction.")
