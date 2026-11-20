"""Load a CZI image with explicit axes and inspect its metadata.

Edit the settings below, then run from Python/:
    python -m examples.load_spectral_czi
"""
from pathlib import Path

import numpy as np

from lumic.data import download_cos7_data
from lumic.io import open_czi


# 1. User settings: choose your file and the plane to load.
FILE_PATH = None  # Default to the public COS-7 CZI, or supply your own path.
# To try a small Z stack:
# FILE_PATH = Path("data/io-data/980/ex405-range411_694-res8p8nm_zstack3.czi")
TIME_POINT = 0
Z_SLICE = 0
# Zero also works when the file has no T/Z axis.
# Set Z_SLICE = None to load all Z slices.
# Set both selections to None to load the whole acquisition.

if FILE_PATH is None:
    image_path, = download_cos7_data(snr="high", progress=True)
else:
    image_path = Path(FILE_PATH).expanduser()


# 2. Inspect the available dimensions, then read the selected pixels.
with open_czi(image_path) as reader:
    print(f"Available image: axes={reader.axes}, shape={reader.shape}")
    image = reader.read(t=TIME_POINT, z=Z_SLICE)

    # Load every Z slice at the chosen time point if the file has a stack.
    # A single-plane acquisition or z=None already supplies all Z positions.
    all_z = reader.read(t=TIME_POINT, z=None) if "Z" in reader.axes and Z_SLICE is not None else image


# 3. The SpectralImage remains usable after the file is closed.
print(f"Loaded image: axes={image.axes}, shape={image.data.shape}")
print(f"All Z positions: axes={all_z.axes}, shape={all_z.data.shape}")
print(f"Pixel sizes (µm): {image.pixel_sizes_um}")
print(f"Channel wavelengths (nm): {image.wavelengths_nm}")
# Recorded detection windows are also available as image.detection_bands_nm.


# 4. Unfold either a plane or a full stack into observations × channels.
# The object remembers the layout; numerical algorithms receive only .data.
unfolded = all_z.unfold()
print(f"Unfolded observations: {unfolded.data.shape}")
# Fitting this stack as one matrix would assume shared spectra across Z slices.
# For large acquisitions, the user can instead load and unfold one plane at a time.


# 5. Refold the unchanged matrix to verify the original pixels and axes return.
# No argument restores the stored matrix. This is a layout check, not unmixing.
restored = unfolded.fold_spectra()
np.testing.assert_array_equal(restored.data, all_z.data)
print(f"Restored layout: axes={restored.axes}, shape={restored.data.shape}")

# After an algorithm returns full-image matrices in the original row order:
# maps = unfolded.fold_abundances(abundances, component_names=names)
# reconstruction = unfolded.fold_spectra(abundances @ spectra)
# Both are SpectralImage objects: maps.kind == "abundance" (K axis), whereas
# reconstruction.kind == "reconstructed" (C axis and original channel metadata).
