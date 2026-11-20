"""Sample a continuous scene on a confocal scan grid, with optional blur."""

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .psf import GaussianPSF
from .scene import Scene2D


_FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))


@dataclass(frozen=True)
class SamplingGrid:
    """Scan positions on a 2D grid, with pixel centres in micrometres.

    ``origin_um`` gives the (x, y) position of the upper-left image edge.
    ``shape`` is (rows, columns). The default pixel spacing is 100 nm.
    """

    shape: Tuple[int, int]
    pixel_size_um: float = 0.1
    origin_um: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if isinstance(self.pixel_size_um, (bool, np.bool_)):
            raise ValueError("pixel_size_um must be positive and finite")
        try:
            shape = tuple(self.shape)
            origin = tuple(float(value) for value in self.origin_um)
            pixel_size = float(self.pixel_size_um)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid sampling grid dimensions or coordinates") from exc
        if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape)):
            raise ValueError("shape must contain two positive integer dimensions")
        if len(origin) != 2 or not np.isfinite(origin).all():
            raise ValueError("origin_um must be a finite (x, y) pair")
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("pixel_size_um must be positive and finite")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "origin_um", origin)
        object.__setattr__(self, "pixel_size_um", pixel_size)


@dataclass(frozen=True)
class SpatialRender:
    """Sampled counts before and after optical blur, in scene component order.

    Both arrays have shape (y, x, fluorophore). ``blurred_counts`` contains the
    expected per-component counts to pass to the spectral detector. It equals
    ``source_counts`` when no PSF was requested.
    """

    source_counts: np.ndarray
    blurred_counts: np.ndarray
    fluorophore_names: Tuple[str, ...]
    grid: SamplingGrid
    psf: GaussianPSF = None


def _evaluate_scene(scene, x_positions, y_positions):
    x_um, y_um = np.meshgrid(x_positions, y_positions)
    maps = np.zeros((*x_um.shape, len(scene.fluorophore_names)), dtype=float)
    component_indices = {name: index for index, name in enumerate(scene.fluorophore_names)}
    for shape in scene.shapes:
        inside = shape.contains(x_um, y_um)
        for name, count_level in shape.counts.items():
            maps[..., component_indices[name]] += count_level * inside
    return maps


def render_scene(scene, grid, psf=None, *, oversample=7):
    """Render shape count levels at scan positions, with optional Gaussian PSF.

    With a PSF, shapes are evaluated on a finer temporary grid, blurred in
    physical units, and sampled at the requested scan positions. Odd
    ``oversample`` values align a fine-grid position with each output pixel
    centre. Increase it if a shape is thinner than the temporary grid spacing.
    A padded field allows objects outside the image to contribute near its
    edges. This is a 2D lateral model; it does not simulate axial light.
    """
    if not isinstance(scene, Scene2D):
        raise TypeError("scene must be a Scene2D")
    if not isinstance(grid, SamplingGrid):
        raise TypeError("grid must be a SamplingGrid")
    if psf is not None and not isinstance(psf, GaussianPSF):
        raise TypeError("psf must be a GaussianPSF or None")
    if isinstance(oversample, bool) or not isinstance(oversample, (int, np.integer)) or oversample < 1 or oversample % 2 == 0:
        raise ValueError("oversample must be a positive odd integer")

    n_y, n_x = grid.shape
    pitch = grid.pixel_size_um
    x_positions = grid.origin_um[0] + (np.arange(n_x) + 0.5) * pitch
    y_positions = grid.origin_um[1] + (np.arange(n_y) + 0.5) * pitch
    source_counts = _evaluate_scene(scene, x_positions, y_positions)

    if psf is None:
        return SpatialRender(
            source_counts, source_counts.copy(), scene.fluorophore_names, grid
        )

    missing = set(scene.fluorophore_names) - set(psf.fwhm_um)
    if missing:
        raise ValueError(f"PSF is missing fluorophore(s): {sorted(missing)}")
    sigmas_um = [psf.fwhm_um[name] / _FWHM_PER_SIGMA
                 for name in scene.fluorophore_names]
    padding_pixels = ceil(4 * max(sigmas_um) / pitch) + 1
    fine_pitch = pitch / oversample
    fine_shape = ((n_y + 2 * padding_pixels) * oversample,
                  (n_x + 2 * padding_pixels) * oversample)
    fine_x = (grid.origin_um[0] - padding_pixels * pitch +
              (np.arange(fine_shape[1]) + 0.5) * fine_pitch)
    fine_y = (grid.origin_um[1] - padding_pixels * pitch +
              (np.arange(fine_shape[0]) + 0.5) * fine_pitch)
    fine_counts = _evaluate_scene(scene, fine_x, fine_y)

    # Each fluorophore may have a different excitation and emission wavelength,
    # so blur its abundance map before the spectral curves are mixed.
    sample_y = padding_pixels * oversample + oversample // 2 + np.arange(n_y) * oversample
    sample_x = padding_pixels * oversample + oversample // 2 + np.arange(n_x) * oversample
    blurred_counts = np.empty_like(source_counts)
    for index, sigma_um in enumerate(sigmas_um):
        blurred_fine = gaussian_filter(
            fine_counts[..., index], sigma=sigma_um / fine_pitch,
            mode="constant", cval=0.0, truncate=4.0,
        )
        blurred_counts[..., index] = blurred_fine[np.ix_(sample_y, sample_x)]

    return SpatialRender(
        source_counts, blurred_counts, scene.fluorophore_names, grid, psf
    )
