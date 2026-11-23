# Save images and spectra

Save the fitted abundance maps and the exact spectra that produced them so
you can reconstruct the fitted detector signal later. Save the spectra as
a separate CSV when you want to use them to unmix another acquisition.

Run [Estimate unknown spectra](estimate-unknown-spectra.md) through **Refine
the spectra with MCR-ALS** to get `unfolded`, `refined_counts`, and
`refined_spectra`. You can also use these steps with your own fitted data.

## From measured image to fitted results

| Object | Shape in this example | What it holds |
| --- | --- | --- |
| Measured `SpectralImage` | `YXC` | The input detector signal, including noise. `C` is the detector channel axis. |
| `UnfoldedImage` | Pixels × channels | One row per pixel for fitting. It remembers the original axes and shape so results can be folded back into an image. |
| Abundance `SpectralImage` | `YXK` | One map per fitted component. `K` identifies components. |
| Reconstructed `SpectralImage` | `YXC` | The detector signal predicted by the fitted abundance maps and spectra. |
| `RGBRendering` | `YXS` | An 8-bit preview with red, green and blue display samples on axis `S`. |

These axes describe the example's array order. Lumic can also preserve other
orders and retain time and Z axes. `fold_spectra()` restores the original
measured pixels. Passing `refined_counts @ refined_spectra` instead folds
the predicted signal into a reconstructed image. To save the abundance maps
with their basis, pass the exact spectra paired with the fitted counts.

```python
measured = unfolded.fold_spectra()
reconstructed = unfolded.fold_spectra(refined_counts @ refined_spectra)

# Use component numbers as labels for a blind fit.
component_names = tuple(f"Component {i + 1}" for i in range(len(refined_spectra)))
abundance = unfolded.fold_abundances(
    refined_counts,
    component_names=component_names,
    spectra=refined_spectra,
    normalization="L1",
    abundance_units="estimated detected counts",
)
```

In this example, each fitted spectrum sums to one, so the abundances
represent estimated contributions to detector counts. The normalization
and unit labels describe the scale of the supplied factors.

## Save and reload OME-TIFF images

`save_ome_tiff` uses lossless Deflate compression by default and preserves
the pixel values. Run this block in a new output directory, or choose
new filenames when repeating it.

```python
from pathlib import Path

from lumic.io import load_image

output = Path("unmixing-results")
output.mkdir(exist_ok=True)

measured_path = measured.save_ome_tiff(output / "measured.ome.tif")
reconstructed_path = reconstructed.save_ome_tiff(output / "reconstructed.ome.tif")
abundance_path = abundance.save_ome_tiff(output / "abundances.ome.tif")

saved_measured = load_image(measured_path, t=None, z=None)
saved_reconstructed = load_image(reconstructed_path, t=None, z=None)
saved_abundance = load_image(abundance_path, t=None, z=None)

# The abundance file alone contains both factors needed for reconstruction.
prediction = saved_abundance.reconstruct()
np.testing.assert_allclose(prediction.data, saved_reconstructed.data)
```

`t=None, z=None` loads the full saved image. Other OME readers can open the
abundance maps as ordinary channels. Lumic's OME annotation restores the
`K` axis and stores the paired spectra and their detector coordinates.
`saved_abundance.unmixing["spectra"]` gives you the exact fitted matrix;
`saved_abundance.component_names` gives its row labels. Reconstruction
from this file reproduces the fitted signal. To compute residuals, use the
measured image as well.

## Save a display preview

Save an RGB preview separately from the quantitative images. This example
writes a regular RGB TIFF for viewing. Its colors come from groups of
detector channels and use a shared display scale.

```python
from tifffile import imwrite

preview = saved_measured.to_rgb()
imwrite(output / "preview-rgb.tif", preview.data, photometric="rgb")
```

## Save and reuse the optimized spectra

Use one `Spectrum` object per component to keep its name, detector wavelengths
and processing history together. Save the related spectra in a single JSON
collection. The numerical APIs still receive a NumPy matrix, so convert the
objects to an array only where fitting starts.

```python
import numpy as np
import matplotlib.pyplot as plt
from lumic import Spectrum, load_spectra_json, save_spectra_json

fitted_spectrum_objects = [
    Spectrum(
        values=refined_spectra[index],
        wavelengths_nm=unfolded.wavelengths_nm,
        name=component_names[index],
        provenance={"method": "MCR-ALS", "normalization": "L1"},
    )
    for index in range(len(component_names))
]

collection_path = output / "optimized-spectra.json"
save_spectra_json(fitted_spectrum_objects, collection_path)
reloaded_spectra = load_spectra_json(collection_path)

# Convert to components × channels for the solver.
spectra_matrix = np.stack([spectrum.values for spectrum in reloaded_spectra])
```

Plot the values against their attached wavelength labels:

```python
for spectrum in reloaded_spectra:
    plt.plot(spectrum.wavelengths_nm, spectrum.values, label=spectrum.name)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized intensity")
plt.legend()
```

The collection stores each component's values, wavelength labels, name and
provenance together in one file. `Spectrum.save_json()` and
`Spectrum.load_json()` remain available for individual references.

The CSV below is a compact alternative when you only need a table for plotting
or exchanging the numerical values. It does not preserve each spectrum's
provenance metadata.

```python
wavelengths_nm = unfolded.wavelengths_nm
csv_path = output / "optimized-spectra.csv"
np.savetxt(
    csv_path,
    np.column_stack((wavelengths_nm, refined_spectra.T)),
    delimiter=",",
    header="wavelength_nm," + ",".join(component_names),
    comments="",
)
```

Reload the table and transpose the component columns to recover the
components × channels layout. Use the same detector setup, including
channel order and detection bands, for the next acquisition. This CSV
stores the channel centers. The abundance OME-TIFF also retains detection
bands when they were present on the source image.

```python
from pathlib import Path

import numpy as np

from lumic import positive_least_squares
from lumic.io import load_image

output = Path("unmixing-results")
with (output / "optimized-spectra.csv").open() as csv_file:
    _, *component_names = csv_file.readline().strip().split(",")
    table = np.loadtxt(csv_file, delimiter=",", ndmin=2)

loaded_spectra = table[:, 1:].T

# Here we refit the saved measured image to demonstrate the round trip.
# For a new acquisition with matching detector settings, load it here.
target = load_image(output / "measured.ome.tif", t=None, z=None).unfold()
np.testing.assert_allclose(target.wavelengths_nm, table[:, 0], rtol=0, atol=1e-6)
new_counts = positive_least_squares(
    target.data, loaded_spectra, do_norm=False, solver="auto",
)
new_maps = target.fold_abundances(
    new_counts, component_names=component_names, spectra=loaded_spectra,
)
```

This fit holds the saved spectra fixed and estimates new abundances.
If you want to refine those spectra for the new image, pass
`loaded_spectra` as the initial estimates to `mcr_als`.
