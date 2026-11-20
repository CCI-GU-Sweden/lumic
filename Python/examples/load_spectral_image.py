"""Inspect a CZI/OME-TIFF header, then explicitly choose whether to load pixels.

Edit the settings below, then run from Python/:
    python -m examples.load_spectral_image
"""
from pathlib import Path

from lumic.data import download_cos7_data
from lumic.io import open_image


# 1. USER SETTINGS
FILE_PATH = None  # Default to the public COS-7 CZI, or supply a CZI or OME-TIFF path.
LOAD_PIXELS = False  # Inspect first; set True when you want to read the image.
TIME_POINT = 0
Z_SLICE = 0  # None requests all Z positions.

if FILE_PATH is None:
    image_path, = download_cos7_data(snr="high", progress=True)
else:
    image_path = Path(FILE_PATH).expanduser()


# 2. INSPECT SUPPORTED FILE METADATA WITHOUT DECODING PIXELS
# open_image returns the existing format-specific reader. Unsupported files
# raise a clear error here. Channel count does not infer fluorophore count.
with open_image(image_path) as reader:
    print(reader.summary())

    # 3. LOAD ONLY WHEN EXPLICITLY REQUESTED
    if LOAD_PIXELS:
        image = reader.read(t=TIME_POINT, z=Z_SLICE)
        print(f"Loaded: axes={image.axes}, shape={image.data.shape}, dtype={image.data.dtype}")
        # The reader can now close; image owns its pixels and metadata.
        unfolded = image.unfold()
        print(f"Analysis matrix: {unfolded.data.shape}")
