"""Working reader template for contributors; NOT registered in open_image().

This teaching adapter uses a .npy memory map to demonstrate the reader contract
without introducing a format-specific dependency. NPY has no channel-axis
metadata, so callers MUST supply axes explicitly. Replace its backend/header
mapping and read/close operations when adapting a real microscopy format.

See docs/contributing/adding-an-image-reader.md in the repository.
No pixels are decoded on construction.
"""
from pathlib import Path

import numpy as np

from lumic import SpectralImage
from lumic.readers import format_reader_summary


class NpyReaderTemplate:
    """Inspect an explicitly labeled .npy array, then copy selected T/Z pixels.

    This is an executable contributor example, not a newly supported dispatcher
    format. It accepts .npy numeric arrays with C/Y/X and optional T/Z axes in
    any order; axes are required because shape alone cannot identify channels.
    No wavelengths/calibration are inferred. Returned pixels are copies, so
    closing the mapping or editing a result does not alter the source file.
    """
    format_name = "NPY (contributor template, explicit axes required)"
    kind = "measured"

    def __init__(self, path, *, axes):
        self.path = Path(path)
        self._array = None
        if self.path.suffix.lower() != ".npy":
            raise ValueError("The contributor template expects a .npy file")
        if not self.path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        try:
            # Mapping reads the array header; it does not materialize pixel data.
            # allow_pickle=False avoids loading Python objects from data files.
            self._array = np.load(self.path, mmap_mode="r", allow_pickle=False)
            header = SpectralImage(self._array, axes)
            self.axes = header.axes
            self.shape = self._array.shape
            self.wavelengths_nm = None
            self.detection_bands_nm = None
            self.pixel_sizes_um = {}
        except Exception:
            self.close()
            raise

    def summary(self):
        """Return common header information; do not print or copy pixels."""
        return format_reader_summary(self)

    def read(self, *, t=0, z=0):
        """Copy the selected array region; 0 selects a plane, None retains an axis."""
        if self._array is None:
            raise ValueError("Reader is closed")
        # SpectralImage validation/wrapping does not copy pixels. slice() checks
        # T/Z indices before copying the requested region of this memory map.
        image = SpectralImage(
            self._array, self.axes,
            provenance={"format": "NPY template", "path": str(self.path)},
        )
        return image.slice(t=t, z=z)

    def close(self):
        """Release this backend's memory mapping; safe to call repeatedly."""
        if self._array is not None:
            array, self._array = self._array, None
            # Backend-specific cleanup: none of our returned images retain
            # views into this mapping, so it is safe to release it here.
            array._mmap.close()

    def __enter__(self):
        if self._array is None:
            raise ValueError("Reader is closed")
        return self

    def __exit__(self, *exc):
        self.close()
