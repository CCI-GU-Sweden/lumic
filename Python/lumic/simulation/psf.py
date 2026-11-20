"""A simple lateral Gaussian approximation to confocal optical blur.

The approximation combines excitation and emission Gaussian widths for an
ideal small pinhole. It does not model a measured objective, pinhole diameter,
axial blur, or aberrations. A measured FWHM can be supplied directly instead.

Gaussian width reference: https://doi.org/10.1364/AO.46.001819
Confocal PSF and pinhole: https://www.zeiss.com/microscopy/en/c/lsc/24/confocal/basics-of-confocal-laser-scanning-microscopy-white-paper.html
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def _positive_number(value, name, minimum=0):
    requirement = "positive and finite" if minimum == 0 else f"finite and at least {minimum}"
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be {requirement}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {requirement}") from exc
    if not np.isfinite(number) or number < minimum or number == 0:
        raise ValueError(f"{name} must be {requirement}")
    return number


@dataclass(frozen=True)
class GaussianPSF:
    """Lateral FWHM in micrometres for each fluorophore."""

    fwhm_um: Mapping[str, float]

    def __post_init__(self):
        if not isinstance(self.fwhm_um, Mapping) or not self.fwhm_um:
            raise ValueError("fwhm_um must map fluorophore names to widths")
        widths = {}
        for name, width in self.fwhm_um.items():
            if not isinstance(name, str) or not name:
                raise ValueError("PSF fluorophore names must be non-empty strings")
            widths[name] = _positive_number(width, "PSF width")
        object.__setattr__(self, "fwhm_um", widths)

    @classmethod
    def approximate_confocal(cls, numerical_aperture, excitation_nm,
                             emission_nm, *, width_scale=1.0):
        """Estimate a Gaussian PSF for each fluorophore.

        Excitation and emission peak wavelengths are mappings keyed by
        fluorophore name. Each separate Gaussian has sigma ~= 0.21*lambda/NA;
        the ideal confocal effective PSF is their product. ``width_scale``
        can broaden that ideal width, but is not a calibrated pinhole model.
        """
        numerical_aperture = _positive_number(numerical_aperture, "numerical_aperture")
        width_scale = _positive_number(width_scale, "width_scale", minimum=1)
        if (not isinstance(excitation_nm, Mapping) or
                not isinstance(emission_nm, Mapping) or
                not excitation_nm or set(excitation_nm) != set(emission_nm)):
            raise ValueError("excitation_nm and emission_nm must have the same fluorophores")

        widths = {}
        for name, excitation in excitation_nm.items():
            excitation = _positive_number(excitation, "excitation wavelength")
            emission = _positive_number(emission_nm[name], "emission wavelength")
            sigma_exc_um = 0.21 * (excitation / 1000) / numerical_aperture
            sigma_em_um = 0.21 * (emission / 1000) / numerical_aperture
            sigma_effective_um = 1 / np.sqrt(
                1 / sigma_exc_um**2 + 1 / sigma_em_um**2
            )
            widths[name] = float(_FWHM_PER_SIGMA * sigma_effective_um * width_scale)
        return cls(widths)
