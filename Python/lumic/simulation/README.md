# Synthetic spectral microscopy images

`lumic.simulation` builds controlled images with known component spectra and
contributions. Use it to understand mixing, inspect noise effects, and evaluate
unmixing against ground truth. It provides the building blocks used by our
mixture examples, overlapping circles, radial spokes and dim-line evaluations.

This is a small simulation toolbox, with explicit approximations. It does not
model a complete microscope or infer which fluorophores are in a real sample.

## How the pieces fit together

```text
component count maps                 source emission curves
        |                                     |
optional spatial rendering/PSF        spectral detector integration
        |                                     |
expected component counts            unit-sum detector signatures
        +------------------+------------------+
                           |
                 linear mixing: counts @ spectra
                           |
                 clean expected detector image
                           |
                 Poisson + optional Gaussian noise
                           |
                   noisy measured image
```

Keep the clean image and true factors when producing the noisy measurement.
The analysis should receive only the measured image; use truth afterward to
check recovered signatures and contributions.

## Building blocks

| Tool | Purpose | Main convention or limit |
| --- | --- | --- |
| `load_offline_emission_spectrum` | Load bundled FPbase emission curves | Returns `(wavelengths_nm, intensities)`; no network request |
| `resample_spectrum` | Interpolate a curve at specified wavelengths | Point sampling, **not detector-bin integration**; no normalization |
| `SpectralDetector` | Integrate curves into contiguous spectral channels | Ideal box-shaped bins; sampling alone adds no noise |
| `make_abundance_maps` | Turn supplied component fractions into count maps | Fractions sum to one; total abundance can vary per pixel |
| `make_mixture_gradient` | Create deterministic binary or multicomponent mixtures | Composition coverage, not a realistic spatial scene |
| `Scene2D`, `Ellipse`, `Rectangle`, `Line` | Define objects in physical coordinates | Count levels may include multiple components; overlapping objects add |
| `SamplingGrid` | Specify scan shape, spacing and origin | Sampling belongs to the renderer, not the scene |
| `GaussianPSF`, `render_scene` | Render component maps with optional lateral blur | Blur each component before spectral mixing |
| `simulate_linear_mixture` | Form a clean image from maps and channel signatures | Does not sample source spectra, normalize signatures or add noise |
| `add_confocal_noise` | Draw a measurement from an expected count image | Poisson noise is always drawn; optional Gaussian term follows |
| `simulate_noisy_image` | Combine detector integration, normalization, mixing and noise | Takes source curves and expected **detected** component counts |

Imports are available directly from `lumic.simulation`; implementations are
separated across [spectra.py](spectra.py), [detector.py](detector.py),
[spatial.py](spatial.py), [scene.py](scene.py), [render.py](render.py),
[psf.py](psf.py), [mixing.py](mixing.py), [noise.py](noise.py),
[simulate.py](simulate.py) and [result.py](result.py).

## Array layouts, units and normalization

| Quantity | Shape / units |
| --- | --- |
| Component maps | `(y, x, components)` |
| Detector signatures | `(components, channels)` |
| Clean or noisy image | `(y, x, channels)` |
| Source wavelengths / detector labels | Nanometres |
| Shape coordinates | `(x, y)` in micrometres |
| Grid shape | `(rows, columns)` = `(y, x)` |
| Pixel spacing / lateral PSF FWHM | Micrometres |

For count-domain simulations, each detector signature sums to one. Then a
component value of `500` means **500 expected detected counts across all
channels at that pixel**. The clean image's channel sum equals the sum of its
component contributions. These are idealized count units, not molecular
concentrations or calibrated raw detector values.

`simulate_noisy_image` normalizes **after detector integration**. Counts refer
to the detector's covered wavelength range, even when some source emission
falls outside it. It does not calculate absolute photon collection efficiency.
The lower-level `simulate_linear_mixture` preserves the signatures supplied;
normalize them yourself when you want this count interpretation.

Spatial objects use a count level per scan pixel at fixed acquisition settings,
not a count density per square micrometre. Changing scan spacing does not
automatically rescale the uniform-interior count level. Blur, boundaries and
object overlap still change local values. Consequently, changing the number
of scan positions is different from summing already recorded pixels into bins.

Many CZI-facing LUMIC functions use `(channels, y, x)`. Convert explicitly:

```python
channel_first = np.moveaxis(image, -1, 0)
# Use spectral_axis=-1 instead when a function supports channel-last input.
```

## Quick start: offline mixtures with shot noise

Activate the repository environment and run Python from the `Python/` directory
so `lumic` is importable. The following example needs no internet connection:

```python
import numpy as np
from lumic.simulation import (
    SpectralDetector, load_offline_emission_spectrum,
    make_mixture_gradient, simulate_noisy_image,
)

names = ("EBFP2", "EGFP", "mCherry")
curves = [load_offline_emission_spectrum(name) for name in names]
detector = SpectralDetector.from_profile("980_32ch")

# Sample a lattice of three-component mixtures, including pure endpoints.
# These are expected contributions, summing to 500 counts at every pixel.
counts = make_mixture_gradient(names, shape=(20, 20), total_abundance=500.)
result = simulate_noisy_image(
    counts, curves, detector, fluorophore_names=names,
    gaussian_sigma=0., rng=np.random.default_rng(2026),
)

expected = result.expected_counts       # (20, 20, 32), clean expectation
measured = result.noisy_image           # (20, 20, 32), Poisson realization
true_counts = result.ideal.abundances   # (20, 20, 3)
true_spectra = result.ideal.spectra     # (3, 32), rows sum to one
wave = result.ideal.wavelengths         # (32,), channel labels in nm

np.testing.assert_allclose(expected.sum(axis=-1), true_counts.sum(axis=-1))
```

The saved offline set contains **EBFP2, EGFP, mCherry and mNeptune**. Attribution
and source information are in [bundled spectra README](fpbase_data/README.md).
The loader returns the same kind of curve pair as
`lumic.fpbase.fetch_emission_spectrum`, so supplied source curves may also come
from FPbase or another measured dataset. Fetching live data is separate from
these offline simulations.

The `980_32ch` profile uses 32 labels from the bundled LSM 980 example, spanning
415–689 nm at approximately 8.9 nm spacing. Bin edges are inferred from adjacent
labels. These are approximate ideal bins, not measured channel transmission or
a claim about the full instrument response. Use `SpectralDetector(bin_edges)`
for explicit edges or `SpectralDetector.from_centers(centers)` to infer them.

To simulate a different detector, specify one more edge than the number of
channels. For example, `SpectralDetector(np.linspace(400., 720., 33))` gives
32 bins of width 10 nm. `detector.sample(wavelengths, intensities)` integrates
the piecewise-linear curve within each bin, treating its intensity as zero
outside the supplied wavelength range. Use the resulting integrated values
to compare how different bin settings sample the same source curve.
`simulate_noisy_image` retains them in `result.sampled_spectra` and uses
unit-sum versions in `result.ideal.spectra` to form the image.

## Control how much the components mix

`make_mixture_gradient` includes pure endpoints. To study a component with
no pure pixels, supply your own fractions to `make_abundance_maps`. Here the
first component has pure pixels and the second contributes at most 35% of
the expected counts at any pixel.

```python
import numpy as np
from lumic.simulation import make_abundance_maps

mixture_shape = (12, 20)
second_fraction = np.broadcast_to(np.linspace(0., 0.35, mixture_shape[1]), mixture_shape)
fractions = np.stack((1. - second_fraction, second_fraction), axis=-1)
mixture_counts = make_abundance_maps(fractions, mixture_shape, total_abundance=500.)

np.testing.assert_allclose(mixture_counts.sum(axis=-1), 500.)
np.testing.assert_allclose(mixture_counts[..., 1].max(), 175.)
```

Raise the lower endpoint above zero to remove the pure regions of the first
component as well. A `(y, x)` array of total counts lets you vary brightness
independently of the mixture fractions. Set the total to zero where you
want an empty field. In a rendered scene, compute fractions from
`spatial.blurred_counts` at pixels with nonzero total counts to measure how
much blur and overlap change the mixtures. If your scene includes a broad
background component, state whether its counts are included in that total.

## Spatial objects, background and blur

Continue from the imports, `names`, `curves` and `detector` above:

```python
from lumic.simulation import (
    Scene2D, Ellipse, Rectangle, Line,
    SamplingGrid, GaussianPSF, render_scene,
)

# Background is a real fourth contribution with its own broad spectrum.
scene_names = (*names, "broad_background")
background_wave = np.arange(400., 721.)
background_curve = (
    background_wave, np.exp(-0.5 * ((background_wave - 500.) / 105.)**2)
)
scene_curves = [*curves, background_curve]

scene = Scene2D(scene_names, (
    Rectangle((3.84, 3.84), (20., 20.), {"broad_background": 24.}),
    Ellipse((2.8, 3.2), (1.4, 1.4), {"EBFP2": 500.}),
    Rectangle((4.1, 3.6), (2., 1.5), {"EGFP": 500.}, angle_deg=20.),
    Line((1., 5.), (6., 2.), width_um=0.18, counts={"mCherry": 500.}),
))
grid = SamplingGrid((64, 64), pixel_size_um=0.12)
psf = GaussianPSF({name: 0.30 for name in scene_names})
spatial = render_scene(scene, grid, psf, oversample=7)

spatial_result = simulate_noisy_image(
    spatial.blurred_counts, scene_curves, detector,
    fluorophore_names=scene_names, rng=np.random.default_rng(2026),
)
```

Shapes can carry several components, for example `{"EGFP": 300., "mCherry": 100.}`.
Their names must appear in the scene's component order, and the source curves
must follow that same order. Equal ellipse radii make a filled circle. Lines
are finite-width segments with rounded ends; polygons and 3D volumes are not
implemented. Shapes do not fetch their own spectra.

`spatial.source_counts` is the scene sampled before blur;
`spatial.blurred_counts` is the **expected contribution map that generated the
detector image**. Use the latter as truth when auditing a blurred simulation.
Blur each component with its own width if desired. Pass `psf=None` for no blur.

The renderer samples scan positions rather than integrating square detector
pixels. With a PSF it uses a finer temporary grid and a padded field. An odd
`oversample` aligns output positions; increase it for thin structures. Without
a PSF, scan positions can miss a narrow object lying between them.

`GaussianPSF.approximate_confocal(NA, excitation_nm, emission_nm)` can estimate
widths from numerical aperture and per-component wavelength dictionaries.
This is an ideal small-pinhole Gaussian approximation, not a calibrated
objective, adjustable pinhole or axial model. A directly specified FWHM is
useful for controlled blur sweeps. No excitation-dependent brightness or
spectral response correction is implied by this approximation.

## Clean images, noise and reproducibility

For a **noiseless** image, integrate and normalize the signatures, then mix:

```python
from lumic.simulation import simulate_linear_mixture, add_confocal_noise

sampled = np.array([detector.sample(*curve) for curve in scene_curves])
spectra = sampled / sampled.sum(axis=1, keepdims=True)
clean = simulate_linear_mixture(
    spatial.blurred_counts, spectra, detector.channel_centers,
    fluorophore_names=scene_names,
)
noisy = add_confocal_noise(clean.image, rng=np.random.default_rng(2026))
np.testing.assert_allclose(clean.image, spatial_result.expected_counts)
```

### Choose the signal level and noise

Lumic draws one noisy measurement for each pixel and detector channel after
spectral mixing. For an expected channel count of `x`, the Poisson draw has
mean `x` and standard deviation `sqrt(x)`. At 2,000 expected counts in a
channel, the shot-noise standard deviation is about 45 counts, or 2.2% of the
signal. At 100 counts, it is 10 counts, or 10%. Expected counts per channel
come from the component maps and unit-sum spectra. A total of 500 expected
counts across the detector can therefore give much lower counts in individual
channels.

Adjust the expected count level through the component maps. If you already
have a clean image in relative units, multiply it by a scale of your choice
to put it in expected count units before calling `add_confocal_noise()`.
For example, multiplying a channel value of `0.5` by 4,000 gives 2,000
expected counts. Choose the scale to match the signal level you want to test.

Use `gaussian_sigma` to add independent zero-mean Gaussian fluctuations, with
the standard deviation specified in the same count units. The default of
zero gives shot noise alone. For example, add a Gaussian standard deviation
of five counts to the clean image above:

```python
noisy_with_gaussian = add_confocal_noise(
    clean.image, gaussian_sigma=5.0, rng=np.random.default_rng(2026),
)
```

Lumic clips negative measurements to zero, which changes the statistics near
zero. Use `clean.image` or `result.expected_counts` for the noiseless
expectation. The Poisson-plus-Gaussian model is useful for controlled tests
across signal levels. Real-detector gain, offset, amplification noise and
saturation require additional calibration and modeling.

Supply a seeded `numpy.random.Generator` for repeatable draws. Construct a
fresh generator with the same seed to replay; reusing an existing generator
advances its state. Keep measurement-noise seeds separate from selection or
initialization seeds in analysis. Record scene, detector, count levels, blur,
seeds, software versions and solver settings for an evaluation.

To inspect a spectral image using the existing display helper:

```python
from matplotlib import pyplot as plt
from lumic.utils import RGB_from_spec

rgb = RGB_from_spec(np.moveaxis(spatial_result.noisy_image, -1, 0), do_norm=True)
plt.imsave("simulation_rgb.png", rgb)
```

This creates `simulation_rgb.png` in the working directory. The RGB helper
groups spectral channels for display; it is not a calibrated color or a
substitute for spectral inspection. Display normalization changes appearance,
not the stored measurement.

## What the result containers retain

| Result | Main contents |
| --- | --- |
| `SpatialRender` from `render_scene` | Before/after blur count maps, component names, grid and PSF |
| `SyntheticDataset` from `simulate_linear_mixture` | Clean image, true contributions, supplied signatures, wavelength labels and names |
| `NoisySimulationResult` from `simulate_noisy_image` | `ideal` dataset, `noisy_image`, detector, source curves, unnormalized integrated `sampled_spectra`, and Gaussian sigma |

`NoisySimulationResult.expected_counts` is its clean `ideal.image`. Shapes,
rendering grid, seeds and analysis settings are not all included in that
container; retain them separately. Frozen dataclasses do not make every array
immutable.

## Examples and evaluations

Run these modules from `Python/`. The first group is for visual understanding;
the evaluations add blind fitting and explicit ground-truth audits.

| Example / evaluation | What to inspect |
| --- | --- |
| [examples.visualize_three_fluorophore_mixture](../../examples/visualize_three_fluorophore_mixture.py) | Three-component mixture grid and spectral RGB |
| [examples.visualize_ternary_mixture](../../examples/visualize_ternary_mixture.py) | A triangular composition diagram; zero signal outside the triangle |
| [examples.visualize_spatial_scene](../../examples/visualize_spatial_scene.py) | Shape maps before and after blur |
| [examples.visualize_overlapping_circles](../../examples/visualize_overlapping_circles.py) | Three overlapping circles, RGB and a separate blind-recovery comparison |
| [examples.visualize_noise_levels](../../examples/visualize_noise_levels.py) | Image previews as expected counts decrease |
| [examples.visualize_poisson_noise_levels](../../examples/visualize_poisson_noise_levels.py) | Spectral curves and noise spread |
| [evaluations.evaluate_hidden_component](../../evaluations/evaluate_hidden_component.py) | Binary mixture with a controlled ceiling on one component's fraction |
| [evaluations.evaluate_radial_spoke_blur](../../evaluations/evaluate_radial_spoke_blur.py) | Noiseless blur, pure-pixel availability and factor recovery |
| [evaluations.evaluate_radial_spoke_noise](../../evaluations/evaluate_radial_spoke_noise.py) | Shot-noise sweep, repeated seeds and noiseless controls |
| [evaluations.evaluate_radial_spoke_binning](../../evaluations/evaluate_radial_spoke_binning.py) | Sum-bin the same low-count pixels for discovery; project spectra onto original pixels |
| [evaluations.evaluate_dim_line_recovery](../../evaluations/evaluate_dim_line_recovery.py) | A dim far-red line and under-specified component count, without noise |

For example: `python -m examples.visualize_spatial_scene`. Visual examples
typically write to `Python/data/`; evaluation arrays and figures default to
ignored subfolders under `Python/evaluations/results/`. Check each script's
output setting; selected reference figures are archived separately.

The binning helper currently lives in evaluation code, **not the public
simulation API**. It sums nonoverlapping measured blocks, without drawing new
noise. This preserves total counts and field of view but changes sampling and
mixes nearby spectra. Learning spectra from those bins does not add photons to
the original pixels when estimating their contributions afterward.

## Interpretation limits

These simulations test explicit hypotheses, not the full behavior of a
microscope. Optical filters, wavelength-dependent efficiency, realistic
autofluorescence variation and photophysics require additional modeling.
The broad Gaussian background in our evaluations is a supplied fourth or
fifth spectrum; it is different from random measurement noise.

A good reconstruction, smooth-looking maps or a solver stopping criterion
does not establish accurate or unique spectra/contributions. Strong noise can
corrupt hull selection before ALS. Binning can help while reducing spatial
detail. Spectral smoothness is a **proposed future prior**, not a current ALS
constraint or simulator feature. Component-count and residual-PC decisions
remain human-guided.

See [lumic.evaluation](../evaluation/README.md) for shared reconstruction,
spectral, contribution and residual diagnostics and [the evaluation guide](../../evaluations/README.md)
for experiment details.

## Save a simulated detector image

Wrap `simulation.image` in a `SpectralImage` with axes `YXC`, optional detector
wavelength/band coordinates and physical pixel sizes. Record component names
under `provenance["fluorophores"]`; they are not detector-channel names.
`image.save_ome_tiff(path)` then writes lossless Deflate OME-TIFF
(the equivalent standalone function is `lumic.io.save_ome_tiff(image, path)`);
`compression="lzw"` or None are alternatives. Reload with `load_ome_tiff` using
`t=None, z=None` for all dimensions. This saves the quantitative detector image,
not a display RGB conversion. To save factor pairs separately, create maps with
`image.unfold().fold_abundances(coefficients, spectra=basis,
component_names=names)` and call `maps.save_ome_tiff(path)`. Use the exact basis
paired with those coefficients; reloaded maps can call `.reconstruct()`.
The [spectral image workflow](../../examples/spectral_image_workflow.py) shows
measured, reconstructed and abundance exports together.
See the [export example](../../examples/export_spectral_ome_tiff.py) and
[metadata contract](../IO_README.md#save-a-quantitative-ome-tiff).
