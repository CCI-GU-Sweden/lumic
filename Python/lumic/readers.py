"""Small structural contract and shared presentation for image-file readers.

Readers need not inherit from a Lumic base class. Format adapters inspect
headers on opening and return ordinary SpectralImage objects on explicit read.
See docs/contributing/adding-an-image-reader.md in the repository and the
template at Python/examples/reader_template.py.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, Self, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .image import SpectralImage


class SpectralImageReader(Protocol):
    """Interface for metadata-first readers, independent of their backend.

    Construction validates file/header/layout without decoding image pixels.
    `axes` labels the available array dimensions; `shape` follows that order.
    Readers expose C/Y/X (detector data) or K/Y/X (abundance results), optional
    T/Z, and optional channel/calibration
    metadata. Missing metadata uses None/empty mapping rather than guesses.

    read() loads the first T/Z plane by default. None requests every position
    along that dimension. A request selects before decoding when the backend
    allows it; adapters must disclose any less selective behavior. Results own
    their pixels and remain usable after close(). No unmixing is performed.

    This Protocol documents structural compatibility for contributors/type
    checkers, not scientific validation. It is not a superclass requirement,
    plugin registry, or runtime substitute for validating the actual file.
    """
    path: Path
    format_name: str
    kind: str
    axes: str
    shape: tuple[int, ...]
    wavelengths_nm: np.ndarray | None
    detection_bands_nm: np.ndarray | None
    pixel_sizes_um: Mapping[str, float]

    def summary(self) -> str:
        """Return shared human-readable header information without pixel I/O."""
        ...

    def read(self, *, t: int | None = 0, z: int | None = 0) -> "SpectralImage":
        """Load requested pixels into an independent SpectralImage."""
        ...

    def close(self) -> None:
        """Release resources; repeated calls must be safe."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(self, *exc) -> None: ...


def format_reader_summary(reader: SpectralImageReader) -> str:
    """Format validated reader metadata consistently without printing or I/O.

    Reports detector channels for C-axis images and fitted component count for
    recognized K-axis abundance maps, not an inferred fluorophore count. Header
    support does not guarantee that every pixel block is uncorrupted or that
    channels form an appropriate linear spectral measurement. Spectral meaning
    still depends on the acquisition (for example, overlapping LSM 900 windows).
    The summary is a header snapshot; read() can still report decoding errors.
    """
    sizes = dict(zip(reader.axes, reader.shape))
    components = "K" in sizes
    calibration = ", ".join(f"{a}={reader.pixel_sizes_um[a]:g}" for a in "XYZ"
                            if a in reader.pixel_sizes_um) or "not available"
    return "\n".join([
        f"File: {reader.path}",
        f"Format: {reader.format_name}",
        "Supported for loading: yes (header/layout checks passed)",
        f"Axes: {reader.axes}",
        f"Shape: {reader.shape}",
        f"Components: {sizes['K']}" if components else f"Detector channels: {sizes['C']}",
        f"Spatial size (Y x X): {sizes['Y']} x {sizes['X']} pixels",
        f"Z planes: {sizes.get('Z', 1)}; time points: {sizes.get('T', 1)}",
        "Wavelength coordinates: " + ("available" if reader.wavelengths_nm is not None else "not available"),
        "Detection bands: " + ("available" if reader.detection_bands_nm is not None else "not available"),
        f"Pixel sizes (µm): {calibration}",
    ])
