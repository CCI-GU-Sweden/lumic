"""Export a public COS-7 image to OME-TIFF, then inspect and reload it.

Run from Python/: python -m examples.load_spectral_ome_tiff
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lumic.data import download_cos7_data
from lumic.io import load_czi, open_ome_tiff


# 1. DOWNLOAD THE CZI OR USE YOUR OWN ACQUISITION
CZI_PATH = None
TIME_POINT = 0
Z_SLICE = 0

if CZI_PATH is None:
    image_path, = download_cos7_data(snr="high", progress=True)
else:
    image_path = Path(CZI_PATH).expanduser()

original = load_czi(image_path, t=TIME_POINT, z=Z_SLICE)


# 2. EXPORT AND READ THE SAME IMAGE THROUGH THE OME-TIFF READER
with TemporaryDirectory() as output_dir:
    ome_path = Path(output_dir) / "cos7.ome.tiff"
    original.save_ome_tiff(ome_path)

    with open_ome_tiff(ome_path) as reader:
        print(reader.summary())
        print(f"Wavelength source: {reader.wavelength_source}")
        image = reader.read(t=None, z=None)


# 3. VERIFY THE ROUND TRIP AFTER BOTH FILE HANDLES HAVE CLOSED
assert image.axes == original.axes
assert image.data.dtype == original.data.dtype
np.testing.assert_array_equal(image.data, original.data)
np.testing.assert_array_equal(image.wavelengths_nm, original.wavelengths_nm)
np.testing.assert_array_equal(image.detection_bands_nm, original.detection_bands_nm)
assert image.pixel_sizes_um.keys() == original.pixel_sizes_um.keys()
for axis, spacing in original.pixel_sizes_um.items():
    np.testing.assert_allclose(image.pixel_sizes_um[axis], spacing)

print("The round trip preserved pixels, dtype, axes, calibration, and detector coordinates.")
print(f"Analysis matrix: {image.unfold().data.shape}")
