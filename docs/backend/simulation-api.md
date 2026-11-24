# Simulation API

Use the simulation API to create spectral images with known component
maps and spectra. You can use bundled emission curves offline and choose
whether to work with a clean image or simulate a noisy acquisition.

| If you want to… | Use |
| --- | --- |
| Load a bundled FPbase emission curve | `load_offline_emission_spectrum()` |
| Interpolate a curve at specific wavelengths | `resample_spectrum()` |
| Integrate a curve over detector bands | `SpectralDetector.sample()` |
| Mix spectra that are already on a shared detector grid | `simulate_linear_mixture()` |
| Add count noise to an existing clean detector image | `add_confocal_noise()` |
| Sample emission curves, mix components, and add noise in one call | `simulate_noisy_image()` |

## Emission curves and detector sampling

`load_offline_emission_spectrum()` returns a pair of one-dimensional arrays
containing wavelengths in nanometres and relative emission intensities.
`resample_spectrum()` interpolates intensity at each requested wavelength.
In contrast, `SpectralDetector.sample()` integrates intensity across the
width of each band, producing one value per detector channel in input
intensity times nanometres. Neither operation normalizes the result.

The built-in `980_32ch` profile is an idealized model of one LSM 980 setting
with band edges inferred from channel labels. For a custom detector with
known contiguous bands, supply their edges to `SpectralDetector()`.
`from_centers()` can infer edges when only channel centers are known.
See [Load and prepare spectra](load-spectra.md) for saving sampled curves as
reusable references.

::: lumic.simulation.SpectralDetector
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [bin_edges, channel_centers, n_channels, from_centers, from_profile, sample]

::: lumic.simulation.load_offline_emission_spectrum
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.simulation.resample_spectrum
    options:
      show_root_heading: true
      show_root_full_path: false

## Mix components and simulate noise

Simulation arrays use **channels last**. Component maps have shape
`(Y, X, K)`, the component matrix has shape `(K, C)`, and the output
detector image has shape `(Y, X, C)`. The names and component maps follow
the row order of the component matrix. When wrapping the output in a
`SpectralImage`, pass `axes="YXC"`.

`simulate_linear_mixture()` uses the component spectra at their supplied
scale. If each row sums to one, the component maps can represent expected
counts per component. `simulate_noisy_image()` integrates the input curves
and normalizes the sampled rows for you. It keeps the clean expected counts,
the noisy image, and the basis used to form the clean image.

```python
import numpy as np

from lumic.simulation import (
    SpectralDetector,
    load_offline_emission_spectrum,
    simulate_noisy_image,
)

names = ("EGFP", "mCherry")
detector = SpectralDetector.from_profile("980_32ch")
curves = [load_offline_emission_spectrum(name) for name in names]
component_counts = np.array([[[100.0, 50.0], [50.0, 100.0]]])
result = simulate_noisy_image(
    component_counts,
    curves,
    detector,
    fluorophore_names=names,
    rng=np.random.default_rng(2026),
)

clean = result.expected_counts       # (1, 2, 32)
noisy = result.noisy_image            # (1, 2, 32)
basis = result.ideal.spectra          # (2, 32), unit-sum detector spectra
```

Noise is drawn in count units. The Poisson draw models shot noise.
Set `gaussian_sigma` above zero to add independent Gaussian noise with
that standard deviation. Observations below zero after adding noise are
clipped to zero. Pass a seeded generator to reproduce a draw.

::: lumic.simulation.simulate_linear_mixture
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.simulation.add_confocal_noise
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.simulation.simulate_noisy_image
    options:
      show_root_heading: true
      show_root_full_path: false

## Result containers

`SyntheticDataset` holds the clean image and the ground truth used to form
it. In `NoisySimulationResult`, `ideal` is that clean dataset.
`sampled_spectra` holds the integrated rows before normalization, and
`source_spectra` holds the original wavelength and intensity pairs.

::: lumic.simulation.SyntheticDataset
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [image, abundances, spectra, wavelengths, fluorophore_names]

::: lumic.simulation.NoisySimulationResult
    options:
      show_root_heading: true
      show_root_full_path: false
      members: [ideal, expected_counts, noisy_image, detector, source_spectra, sampled_spectra, gaussian_sigma]

For a complete workflow, see [Simulate a spectral image](simulate-image.md)
and [Evaluate simulation recovery](evaluate-simulation-recovery.md).
