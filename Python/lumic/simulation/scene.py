"""Continuous two-dimensional sample geometry for spatial simulations.

Coordinates and lengths are in micrometres. The count level of a shape is the
expected signal from a broad, uniform interior at a fixed acquisition setting;
it does not change when the scan position spacing changes.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import numpy as np


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite (x, y) pair in micrometres")
    return tuple(point)


def _positive_pair(value, name):
    pair = _point(value, name)
    if any(number <= 0 for number in pair):
        raise ValueError(f"{name} must contain positive lengths")
    return pair


def _counts(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("counts must map fluorophore names to count levels")
    counts = {}
    for name, level in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("fluorophore names in counts must be non-empty strings")
        if isinstance(level, (bool, np.bool_)) or not np.isscalar(level):
            raise ValueError("count levels must be non-negative finite numbers")
        try:
            level = float(level)
        except (TypeError, ValueError) as exc:
            raise ValueError("count levels must be non-negative finite numbers") from exc
        if not np.isfinite(level) or level < 0:
            raise ValueError("count levels must be non-negative finite numbers")
        counts[name] = level
    return counts


def _finite_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return number


def _local_coordinates(x_um, y_um, center_um, angle_deg):
    dx = x_um - center_um[0]
    dy = y_um - center_um[1]
    angle = np.deg2rad(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


@dataclass(frozen=True)
class Ellipse:
    """Filled ellipse; equal radii produce a disk."""

    center_um: Tuple[float, float]
    radii_um: Tuple[float, float]
    counts: Mapping[str, float]
    angle_deg: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "center_um", _point(self.center_um, "center_um"))
        object.__setattr__(self, "radii_um", _positive_pair(self.radii_um, "radii_um"))
        object.__setattr__(self, "counts", _counts(self.counts))
        object.__setattr__(self, "angle_deg", _finite_scalar(self.angle_deg, "angle_deg"))

    def contains(self, x_um, y_um):
        x, y = _local_coordinates(x_um, y_um, self.center_um, self.angle_deg)
        return (x / self.radii_um[0]) ** 2 + (y / self.radii_um[1]) ** 2 <= 1


@dataclass(frozen=True)
class Rectangle:
    """Filled rectangle, optionally rotated around its centre."""

    center_um: Tuple[float, float]
    size_um: Tuple[float, float]
    counts: Mapping[str, float]
    angle_deg: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "center_um", _point(self.center_um, "center_um"))
        object.__setattr__(self, "size_um", _positive_pair(self.size_um, "size_um"))
        object.__setattr__(self, "counts", _counts(self.counts))
        object.__setattr__(self, "angle_deg", _finite_scalar(self.angle_deg, "angle_deg"))

    def contains(self, x_um, y_um):
        x, y = _local_coordinates(x_um, y_um, self.center_um, self.angle_deg)
        return (np.abs(x) <= self.size_um[0] / 2) & (np.abs(y) <= self.size_um[1] / 2)


@dataclass(frozen=True)
class Line:
    """Filled line segment with a finite width and rounded ends."""

    start_um: Tuple[float, float]
    end_um: Tuple[float, float]
    width_um: float
    counts: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "start_um", _point(self.start_um, "start_um"))
        object.__setattr__(self, "end_um", _point(self.end_um, "end_um"))
        object.__setattr__(self, "counts", _counts(self.counts))
        if self.start_um == self.end_um:
            raise ValueError("a line must have distinct endpoints")
        object.__setattr__(self, "width_um", _finite_scalar(
            self.width_um, "width_um", positive=True
        ))

    def contains(self, x_um, y_um):
        dx = self.end_um[0] - self.start_um[0]
        dy = self.end_um[1] - self.start_um[1]
        projection = ((x_um - self.start_um[0]) * dx +
                      (y_um - self.start_um[1]) * dy) / (dx * dx + dy * dy)
        projection = np.clip(projection, 0, 1)
        distance_squared = ((x_um - self.start_um[0] - projection * dx) ** 2 +
                            (y_um - self.start_um[1] - projection * dy) ** 2)
        return distance_squared <= (self.width_um / 2) ** 2


Shape2D = Union[Ellipse, Rectangle, Line]


@dataclass(frozen=True)
class Scene2D:
    """Shapes and fluorophores in physical coordinates, independent of imaging."""

    fluorophore_names: Tuple[str, ...]
    shapes: Tuple[Shape2D, ...]

    def __post_init__(self):
        names = tuple(self.fluorophore_names)
        shapes = tuple(self.shapes)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("fluorophore_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("fluorophore_names must be unique")
        for shape in shapes:
            if not isinstance(shape, (Ellipse, Rectangle, Line)):
                raise TypeError("shapes must contain Ellipse, Rectangle, or Line objects")
            unknown = set(shape.counts) - set(names)
            if unknown:
                raise ValueError(f"shape refers to unknown fluorophore(s): {sorted(unknown)}")
        object.__setattr__(self, "fluorophore_names", names)
        object.__setattr__(self, "shapes", shapes)
