# Spectral uniqueness maps

From the `Python` directory, score a spectral image using the same coordinate
mapping as the convex-hull helper:

```python
import lumic

# CZI/legacy layout: (channel, y, x). For simulation images use spectral_axis=-1.
result = lumic.spectral_uniqueness_map(
    spectral_image, npc=3, spectral_axis=0, min_total_intensity=50
)
score_image = result.scores
direction_image = lumic.spectral_uniqueness_rgb(result)
```

The map fits an uncentered `TruncatedSVD`, divides each pixel's projected
coordinates by its first coordinate, then measures Euclidean distance from
the mean point in the remaining `npc - 1` dimensions. This shares the
projection code with `essential_spectral_pixles`; the map itself does not
need to compute a hull or select pure spectra. Thresholds are in the input's
intensity units and should be chosen for the acquisition.

`result.coordinates`, `center_coordinates`, and `projection_components`
expose the space behind the score. `valid_mask` identifies retained pixels;
excluded pixels have NaN scores. Zero/low signal, negative or nonfinite
channels, and unusable first SVD coordinates are excluded.

Pass `reference=result` when scoring another image to reuse both the basis
and center. The images must have the same channel definitions and order.
For a fixed basis the normalization cancels multiplicative brightness;
refitting SVD after changing the intensity distribution can change the space.
A high distance indicates spectral distinctness relative to the population,
not proven component purity. A uniform pure-component image has zero scores
against its own center. Noise and background may also be distinct.

Direction colors use the angle of the first two retained coordinates as hue,
and distance in all retained dimensions as brightness and saturation. For `npc=2`, the two
sides of the center are cyan and red. Black marks an undefined direction at
the displayed center; excluded pixels are gray. Higher-dimensional directions
can share a hue. Use the same `max_distance` for comparable color scales;
these are display colors, not emission colors or fluorophore identities.

Spectral selection and map functions live together in `lumic/uniqueness.py`,
with existing `lumic.*` calls preserved.

Run the offline three-circle visual example:

```bash
python -m examples.visualize_spectral_uniqueness
```

The example compares intensity, spectral RGB, distance maps, direction colors, and projected
points for the original scene, a scene four times brighter, and a blurred
scene. It uses a shared projection and center from the original image.
The output is saved to
`Python/data/visual-check-spectral-uniqueness.png`.
