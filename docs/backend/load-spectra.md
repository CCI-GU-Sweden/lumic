# Load and prepare spectra

Use a `Spectrum` for a single reference and a `SpectrumCollection` for an
ordered set of references that share a sample axis. Install Lumic using the
[backend quickstart](quickstart.md).

## Load saved references

Use the class that matches the format in the JSON file. A single-reference
file has format `lumic-spectrum`. A collection file has format
`lumic-spectrum-collection` and stores the wavelength coordinates and
detector bands once for the whole collection.

```python
from lumic import Spectrum, SpectrumCollection

single = Spectrum.load_json("reference.json")
references = SpectrumCollection.load_json("references.json")
first = references.spectrum(0)
```

To group existing `Spectrum` objects, call
`SpectrumCollection.from_spectra([first, second])`. Sample counts must agree.
If the spectra have wavelength labels, the labels must match in sample order.
Align the references explicitly before grouping them. For image unmixing,
check that each column corresponds to the same detector channel in the
references and the measured data, in the same order.

## Prepare references from bundled FPbase curves

The `980_32ch` profile approximates one LSM 980 detector setting with 32
ideal wavelength bands. `detector.sample()` integrates each emission curve
over those bands. Use the profile for this simulation and use acquisition
metadata to establish the detector bands for experimental data.

```python
import numpy as np

from lumic import SpectrumCollection
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum

names = ("EGFP", "mCherry")
detector = SpectralDetector.from_profile("980_32ch")
values = np.stack([
    detector.sample(*load_offline_emission_spectrum(name))
    for name in names
])
edges = detector.bin_edges
references = SpectrumCollection(
    values,
    wavelengths_nm=detector.channel_centers,
    detection_bands_nm=np.column_stack((edges[:-1], edges[1:])),
    names=names,
    row_provenance=[{"source": "Bundled FPbase emission curve"} for name in names],
    provenance={"detector_profile": detector.name},
)
references.save_json("references.json")
components = SpectrumCollection.load_json("references.json").values
```

`components` is a NumPy array with shape `(2, 32)`, with one reference per
row and one detector channel per column. Pass it as the second argument to
`positive_least_squares(observations, components)` after checking that the
observations use the same channel order. The bundled curves and this workflow
run offline.

The [full runnable example][workflow] saves the reference file at a path you
choose and compares the fits from the original and reloaded matrices. Run it
from `Python/`.

```bash
python -m examples.spectrum_collection_workflow references.json
```

[workflow]: https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/spectrum_collection_workflow.py
