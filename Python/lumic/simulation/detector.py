"""Ideal spectral detectors with contiguous wavelength channels."""

import numpy as np
from scipy.integrate import trapezoid


# Integer channel labels recorded with the bundled LSM 980 CZI example.
# Their approximately 8.9 nm spacing is used to infer ideal, box-shaped bins;
# these edges are not a measured transmission curve for the instrument.
_LSM980_32CH_CENTERS = (
    415, 424, 433, 442, 450, 459, 468, 477,
    486, 495, 503, 512, 521, 530, 539, 548,
    556, 565, 574, 583, 592, 601, 610, 618,
    627, 636, 645, 654, 663, 671, 680, 689,
)


class SpectralDetector:
    """Integrate spectral intensity within contiguous detector channels.

    ``bin_edges`` has one more value than the number of channels. The
    ``sample`` method treats the supplied intensity as a piecewise-linear
    spectral density and returns its integral in each channel. The spectrum
    is taken to be zero outside its measured wavelength range.
    """

    def __init__(self, bin_edges, *, channel_centers=None, name=None,
                 nominal_resolution_nm=None):
        edges = np.asarray(bin_edges, dtype=float)
        if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("bin_edges must be finite and strictly increasing")

        if channel_centers is None:
            centers = (edges[:-1] + edges[1:]) / 2
        else:
            centers = np.asarray(channel_centers, dtype=float)
            if (centers.ndim != 1 or centers.size != edges.size - 1
                    or not np.isfinite(centers).all()
                    or np.any(centers <= edges[:-1])
                    or np.any(centers >= edges[1:])):
                raise ValueError("channel_centers must lie inside their bins")

        if nominal_resolution_nm is not None and (
            not np.isfinite(nominal_resolution_nm) or nominal_resolution_nm <= 0
        ):
            raise ValueError("nominal_resolution_nm must be positive and finite")

        self._bin_edges = edges.copy()
        self._channel_centers = centers.copy()
        self._bin_edges.flags.writeable = False
        self._channel_centers.flags.writeable = False
        self.name = name
        self.nominal_resolution_nm = nominal_resolution_nm

    @property
    def bin_edges(self):
        """Wavelength boundaries of the channels, in nm."""
        return self._bin_edges.copy()

    @property
    def channel_centers(self):
        """Nominal wavelength labels of the channels, in nm."""
        return self._channel_centers.copy()

    @property
    def n_channels(self):
        return self._bin_edges.size - 1

    @classmethod
    def from_centers(cls, channel_centers, *, name=None,
                     nominal_resolution_nm=None):
        """Infer contiguous bin edges halfway between adjacent center labels.

        The two outer edges are extrapolated by half the nearest center
        spacing. At least two centers are needed to infer a width.
        """
        centers = np.asarray(channel_centers, dtype=float)
        if (centers.ndim != 1 or centers.size < 2
                or not np.isfinite(centers).all()
                or np.any(np.diff(centers) <= 0)):
            raise ValueError("channel_centers must be finite and strictly increasing")

        edges = np.empty(centers.size + 1, dtype=float)
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2
        edges[0] = centers[0] - (centers[1] - centers[0]) / 2
        edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
        return cls(edges, channel_centers=centers, name=name,
                   nominal_resolution_nm=nominal_resolution_nm)

    @classmethod
    def from_profile(cls, name):
        """Build a named ideal detector profile."""
        if name != "980_32ch":
            raise ValueError(f"unknown spectral detector profile: {name!r}")
        return cls.from_centers(
            _LSM980_32CH_CENTERS,
            name=name,
            nominal_resolution_nm=8.9,
        )

    def sample(self, wavelengths, intensities):
        """Return the integrated signal in each spectral channel.

        Source samples are connected by straight lines. Input intensity is
        not normalized; output values have intensity-times-nm units. This is
        deterministic binning and does not add measurement noise.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        if (wavelengths.ndim != 1 or intensities.ndim != 1
                or wavelengths.size != intensities.size or wavelengths.size < 2
                or not np.isfinite(wavelengths).all()
                or not np.isfinite(intensities).all()
                or np.any(np.diff(wavelengths) <= 0)
                or np.any(intensities < 0)):
            raise ValueError(
                "wavelengths and intensities must be matching finite 1D arrays "
                "with increasing wavelengths and non-negative intensities"
            )

        sampled = np.zeros(self.n_channels, dtype=float)
        for index, (left, right) in enumerate(zip(self._bin_edges[:-1],
                                                   self._bin_edges[1:])):
            # Only integrate where the source curve is actually known. This
            # avoids inventing signal beyond its measured wavelength limits.
            start = max(left, wavelengths[0])
            stop = min(right, wavelengths[-1])
            if start >= stop:
                continue
            interior = wavelengths[(wavelengths > start) & (wavelengths < stop)]
            grid = np.concatenate(([start], interior, [stop]))
            values = np.interp(grid, wavelengths, intensities)
            sampled[index] = trapezoid(values, grid)
        return sampled
