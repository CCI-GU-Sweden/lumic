# Backend quickstart

Install the Lumic backend release candidate in a **Python 3.12** environment.

```bash
python -m pip install "lumic==0.1.0rc4"
```

## Unmix a simulated pixel

Copy this example into a Python script and run it in the same environment.
It uses bundled FPbase spectra for EGFP and mCherry and samples them with
Lumic's approximate 32-channel LSM 980 detector model. Once Lumic is installed,
the example runs offline.

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

Expected output:

```text
[70. 30.]
```

Each reference spectrum sums to one across the detector channels. The two
numbers are the estimated contributions to the total detector counts, in
EGFP and mCherry order. We can check the estimate directly because we chose
the original mixture. The fit receives both reference spectra as inputs.

The emission curves come from [FPbase](https://www.fpbase.org/). Source IDs
and attribution are bundled with the spectra in the Lumic package.
