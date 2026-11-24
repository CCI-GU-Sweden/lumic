# Spectra API

Use `Spectrum` for a single spectral signature and `SpectrumCollection`
for several signatures sampled on the same grid. A collection stores one
row per component and one column per wavelength sample or detector channel.
It keeps the shared axis and optional detector bands alongside the matrix,
with a name and source information for each row.

## Choose a container

| Task | Use |
| --- | --- |
| Keep or exchange one signature | `Spectrum` with a one-dimensional `values` array |
| Group aligned signatures with their names and provenance | `SpectrumCollection.from_spectra()` |
| Attach a shared grid to an existing component matrix | `SpectrumCollection(values, wavelengths_nm=..., detection_bands_nm=...)` |
| Recover one signature from a collection | `collection.spectrum(index)` |

When grouping spectra, sample counts and wavelength coordinates must match
exactly in the same order, or all spectra must omit wavelength coordinates.
The collection does not resample, sort, or normalize its inputs. If wavelengths
are unavailable, leave `wavelengths_nm` unset and verify channel order from the
acquisition. Matching wavelength centers alone does not establish that two
detector grids are compatible. Check the detector bands when available.

## Save and reuse references

Both containers have `save_json()` and `load_json()` methods. Use the class
that matches the file format: `lumic-spectrum` for a single signature,
`lumic-spectrum-collection` for a shared-axis matrix. The latter stores the
wavelength axis and detector bands once. Provenance must be JSON compatible
when saved.

```python
from lumic import Spectrum, SpectrumCollection

wavelengths = [500, 510, 520]
reference_a = Spectrum([0.2, 0.5, 0.3], wavelengths, name="A")
reference_b = Spectrum([0.5, 0.4, 0.1], wavelengths, name="B")
references = SpectrumCollection.from_spectra([reference_a, reference_b])

references.save_json("references.json")
restored = SpectrumCollection.load_json("references.json")
first_reference = restored.spectrum(0)
component_matrix = restored.values  # (2, 3), components × samples
```

Pass `component_matrix` to the solver when its columns match the detector
channels of the observations. See [Load and prepare spectra](load-spectra.md)
for a detector sampling example and
[Unmixing and spectral selection](unmixing-api.md) for fitting.

## Single spectrum

::: lumic.spectra.Spectrum
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [values, wavelengths_nm, name, provenance, save_json, load_json]

## Spectrum collection

::: lumic.spectra.SpectrumCollection
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members: [values, wavelengths_nm, detection_bands_nm, names, row_provenance, provenance, from_spectra, spectrum, save_json, load_json]
