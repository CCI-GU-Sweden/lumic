"""Save and reload an offline FPbase simulation without changing its pixels.

From Python/: python -m examples.export_spectral_ome_tiff
Edit the settings below. The OME-TIFF and RGB PNG are generated locally,
not bundled in Git.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from lumic import SpectralImage
from lumic.io import load_ome_tiff
from lumic.simulation import (
    SpectralDetector, load_offline_emission_spectrum, make_mixture_gradient,
    simulate_linear_mixture,
)


# 1. USER SETTINGS
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
IMAGE_SHAPE = (20, 20)
COUNTS_PER_PIXEL = 5000.0  # Sum over all detector channels.
PIXEL_SIZE_UM = 0.1
COMPRESSION = "deflate"  # Lossless: "deflate", "lzw", or None (uncompressed).
OUTPUT_PATH = Path(__file__).resolve().parent / "output/offline-mixture.ome.tiff"
RGB_OUTPUT_PATH = OUTPUT_PATH.parent / "offline-mixture-rgb.png"
OVERWRITE = True  # Explicitly replace the previous output when rerunning.


# 2. CREATE A NOISELESS NUMPY IMAGE USING REAL OFFLINE SPECTRA
# These names identify components; each detector channel contains their mixture.
detector = SpectralDetector.from_profile("980_32ch")
spectra = np.array([
    detector.sample(*load_offline_emission_spectrum(name))
    for name in FLUOROPHORES
])
spectra /= spectra.sum(axis=1, keepdims=True)
abundances = make_mixture_gradient(FLUOROPHORES, IMAGE_SHAPE, COUNTS_PER_PIXEL)
simulation = simulate_linear_mixture(
    abundances, spectra, detector.channel_centers, fluorophore_names=FLUOROPHORES,
)


# 3. ATTACH EXPLICIT AXES AND METADATA
# Keep fractional expected counts: export does not convert these floats to uint16.
image = SpectralImage(
    simulation.image, "YXC",
    wavelengths_nm=detector.channel_centers,
    detection_bands_nm=np.column_stack([detector.bin_edges[:-1], detector.bin_edges[1:]]),
    pixel_sizes_um={"X": PIXEL_SIZE_UM, "Y": PIXEL_SIZE_UM},
    provenance={
        "source": "offline FPbase simulation",
        "fluorophores": FLUOROPHORES,
        "detector_profile": "980_32ch",
        "counts_per_pixel": COUNTS_PER_PIXEL,
        "channel_names": [f"Detector band {i + 1}" for i in range(len(detector.channel_centers))],
    },
)


# 4. WRITE A LOSSLESS OME-TIFF, THEN LOAD IT WITH THE SAME IMAGE API
# Standard OME fields describe calibration and channel names. Lumic-specific
# fields use a labeled OME annotation; other tools can still read the pixels.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save_ome_tiff(OUTPUT_PATH, compression=COMPRESSION, overwrite=OVERWRITE)
restored = load_ome_tiff(OUTPUT_PATH, t=None, z=None)


# 5. VERIFY THAT EXPORT WAS A STORAGE OPERATION, NOT AN IMAGE TRANSFORMATION
assert restored.axes == image.axes and restored.data.dtype == image.data.dtype
np.testing.assert_array_equal(restored.data, image.data)
np.testing.assert_array_equal(restored.wavelengths_nm, image.wavelengths_nm)
np.testing.assert_array_equal(restored.detection_bands_nm, image.detection_bands_nm)
assert restored.pixel_sizes_um == image.pixel_sizes_um
print(f"Saved: {OUTPUT_PATH}")
print(f"Restored: axes={restored.axes}, shape={restored.data.shape}, dtype={restored.data.dtype}")
print(f"Fluorophores: {restored.provenance['source_provenance']['fluorophores']}")
print(f"Compression: {COMPRESSION}; file size: {OUTPUT_PATH.stat().st_size:,} bytes")
print("All pixels, detector coordinates and calibration survived unchanged.")


# 6. SAVE A DISPLAY PREVIEW OF THE RELOADED IMAGE
# Shared RGB scaling keeps the channel balance. This 8-bit PNG is for viewing;
# the OME-TIFF above retains the quantitative spectral pixels and metadata.
rendering = restored.to_rgb()
Image.fromarray(rendering.data).save(RGB_OUTPUT_PATH)
print(f"Saved RGB preview: {RGB_OUTPUT_PATH}")
