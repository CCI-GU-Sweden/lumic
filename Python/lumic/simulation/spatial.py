"""Construction of controlled fluorophore abundance maps."""

from math import comb

import numpy as np


def make_abundance_maps(fractions, shape, total_abundance=1.0):
    """Scale per-pixel fluorophore fractions into abundance maps.

    ``fractions`` may have shape ``(components,)`` to repeat one mixture over
    the image, or ``(y, x, components)`` to specify a mixture at every pixel.
    Fractions must be non-negative and sum to one at each pixel. Set
    ``total_abundance`` to zero for background pixels.
    """
    fractions = np.asarray(fractions, dtype=float)
    if len(shape) != 2 or any(not isinstance(size, (int, np.integer)) or size <= 0
                              for size in shape):
        raise ValueError("shape must contain two positive integer dimensions")
    if fractions.ndim == 1:
        fractions = np.broadcast_to(fractions, (*shape, fractions.size))
    elif fractions.ndim != 3 or fractions.shape[:2] != tuple(shape):
        raise ValueError("fractions must have shape (components,) or (y, x, components)")
    if fractions.shape[2] == 0 or not np.isfinite(fractions).all():
        raise ValueError("fractions must contain finite component values")
    if np.any(fractions < 0) or not np.allclose(fractions.sum(axis=-1), 1.0):
        raise ValueError("fractions must be non-negative and sum to one per pixel")

    total = np.asarray(total_abundance, dtype=float)
    if total.ndim == 0:
        total = np.full(shape, total.item())
    if total.shape != tuple(shape) or not np.isfinite(total).all() or np.any(total < 0):
        raise ValueError("total_abundance must be a non-negative scalar or (y, x) array")
    return fractions * total[..., np.newaxis]


def make_mixture_gradient(
    fluorophore_names, shape, total_abundance=1.0, n_steps=None
):
    """Create a deterministic map sampling mixtures across all components.

    Pixels are filled in flattened row-major order. With two fluorophores, the
    first pixel is pure in the first fluorophore and the last is pure in the
    second, with linearly spaced mixtures in between. For three or more, the
    map samples every composition on a regular simplex lattice. ``n_steps``
    sets the number of increments between pure components; by default the
    finest lattice that fits the number of pixels is chosen. If there are more
    pixels than lattice compositions, lattice points are repeated at even
    intervals. Explicit ``n_steps`` raises an error if its lattice cannot fit.

    The returned array has shape ``(y, x, fluorophore)`` and contains absolute
    abundances scaled by ``total_abundance``. Fractions sum to one at each
    pixel, so this helper creates no background pixels.
    """
    fluorophore_names = tuple(fluorophore_names)
    if not fluorophore_names or any(
        not isinstance(name, str) or not name for name in fluorophore_names
    ):
        raise ValueError("provide at least one non-empty fluorophore name")
    if len(set(fluorophore_names)) != len(fluorophore_names):
        raise ValueError("fluorophore names must be unique")
    if len(shape) != 2 or any(
        not isinstance(size, (int, np.integer)) or size <= 0 for size in shape
    ):
        raise ValueError("shape must contain two positive integer dimensions")

    n_pixels = int(shape[0]) * int(shape[1])
    n_components = len(fluorophore_names)
    if n_pixels < n_components:
        raise ValueError("shape must have at least one pixel per fluorophore")
    if n_steps is None:
        if n_components == 1:
            n_steps = 1
        elif n_components == 2:
            n_steps = n_pixels - 1
        else:
            n_steps = 1
            while comb(n_steps + n_components, n_components - 1) <= n_pixels:
                n_steps += 1
    elif not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")

    n_compositions = comb(n_steps + n_components - 1, n_components - 1)
    if n_compositions > n_pixels:
        raise ValueError(
            f"shape has {n_pixels} pixels but n_steps={n_steps} requires "
            f"{n_compositions} to cover all mixture compositions"
        )

    # Enumerate integer compositions of n_steps, ordered from the first pure
    # component toward the last, then scale them to fractions summing to one.
    compositions = []

    def add_compositions(prefix, remaining, components_left):
        if components_left == 1:
            compositions.append(prefix + [remaining])
            return
        for value in range(remaining, -1, -1):
            add_compositions(prefix + [value], remaining - value, components_left - 1)

    add_compositions([], n_steps, n_components)
    compositions = np.asarray(compositions, dtype=float) / n_steps
    selected = np.rint(np.linspace(0, len(compositions) - 1, n_pixels)).astype(int)
    fractions = compositions[selected].reshape(*shape, n_components)
    return make_abundance_maps(fractions, shape, total_abundance)
