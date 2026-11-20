"""An optional container for a single spectral signature."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Keep a spectrum's values, wavelength labels and source information together.

    `values` is a nonempty, finite, one-dimensional array of real intensities.
    Optional `wavelengths_nm` contains one finite, positive coordinate per
    value, in the same order. Use `name` for the label and `provenance` for
    descriptive information about acquisition or processing.

    Arrays are copied at construction. Values retain their dtype, order and
    scale. Wavelength labels alone do not establish compatibility with an
    image's detector channels. Pass ordinary NumPy arrays to the solvers.
    """

    values: np.ndarray
    wavelengths_nm: np.ndarray | None = None
    name: str | None = None
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "uif":
            raise ValueError("values must be a nonempty one-dimensional array of real numbers")
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        object.__setattr__(self, "values", values.copy())

        if self.wavelengths_nm is not None:
            wavelengths = np.asarray(self.wavelengths_nm)
            if (wavelengths.shape != values.shape or wavelengths.dtype.kind not in "uif"
                    or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0)):
                raise ValueError("wavelengths_nm must contain one finite positive number per value")
            object.__setattr__(self, "wavelengths_nm", wavelengths.copy())

        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        object.__setattr__(self, "provenance", dict(self.provenance))
