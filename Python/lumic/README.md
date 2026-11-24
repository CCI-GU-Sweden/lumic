# Lumic

Lumic separates overlapping fluorescence signals in spectral microscopy
images. The Python backend reads ZEISS CZI and OME-TIFF, estimates component
contributions from reference spectra or refines spectra with MCR-ALS, and
exports abundance maps and reusable spectral references. It runs without a
graphical interface.

[Documentation](https://cci-gu-sweden.github.io/lumic/) ·
[Backend API reference](https://cci-gu-sweden.github.io/lumic/backend/api-overview/) ·
[GitHub repository](https://github.com/CCI-GU-Sweden/lumic)

## Install

Install Lumic in a Python 3.12 environment.

```sh
python -m pip install lumic
```

For Jupyter notebooks and plotting tools, install
`"lumic[examples]"`. Use the `visualization` extra for spectral
uniqueness RGB rendering or the `fpbase` extra to retrieve spectra online.
Bundled FPbase curves are available in the default install. See the
[example setup guide](https://cci-gu-sweden.github.io/lumic/examples/) for
running scripts from a repository checkout.

## Unmix a simulated pixel offline

Run this example with the installed package. It samples two bundled emission
curves over the idealized 32-channel LSM 980 detector bands and recovers their
known contributions. It needs no microscopy files or network access.

```python
import numpy as np

from lumic import positive_least_squares
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum

detector = SpectralDetector.from_profile("980_32ch")
references = np.array([
    detector.sample(*load_offline_emission_spectrum(name))
    for name in ("EGFP", "mCherry")
])
references /= references.sum(axis=1, keepdims=True)

mixed_pixel = np.array([[70.0, 30.0]]) @ references
estimated = positive_least_squares(mixed_pixel, references, solver="auto")
print(np.round(estimated[0], 6))
```

Expected output: `[70. 30.]` in EGFP, mCherry order.

## Choose your next step

| Task | Guide |
| --- | --- |
| Inspect image metadata and select a time point or Z plane | [Load spectral images](https://cci-gu-sweden.github.io/lumic/backend/load-images/) |
| Unmix an experimental image from the public BioImage Archive | [COS-7 walkthrough and notebook](https://cci-gu-sweden.github.io/lumic/backend/cos7-workflow/) |
| Simulate a mixture with known component maps | [Simulate a spectral image](https://cci-gu-sweden.github.io/lumic/backend/simulate-image/) |
| Inspect residuals or measure recovery against known factors | [Experimental fit diagnostics](https://cci-gu-sweden.github.io/lumic/backend/evaluate-results/) and [simulation recovery](https://cci-gu-sweden.github.io/lumic/backend/evaluate-simulation-recovery/) |
| Export abundance maps or reuse reference spectra | [Save images and spectra](https://cci-gu-sweden.github.io/lumic/backend/save-results/) |

For a graphical workflow, see the separately installed
[napari plugin](https://cci-gu-sweden.github.io/lumic/plugins/overview/).

## License and data attribution

Lumic is distributed under the MIT license. The bundled emission curves
include [FPbase source and attribution information](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/simulation/fpbase_data/README.md).
