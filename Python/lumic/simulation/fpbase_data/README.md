# FPbase emission spectra

These CSV files are a small, fixed snapshot of default-state **emission** (`EM`)
spectra retrieved from [FPbase](https://www.fpbase.org/) on 2026-09-30 using
the `fpbase` Python client (version 0.1.3) and its GraphQL API. They allow
simulation examples and tests to run without an internet connection. `EBFP2`
is the blue protein selected for this set; it is distinct from `EBFP1.2` and
`mTagBFP2`.

| File | FPbase protein page | Spectrum ID | Samples | Wavelength range |
| --- | --- | ---: | ---: | --- |
| `ebfp2_emission.csv` | [EBFP2](https://www.fpbase.org/protein/ebfp2/) | 14 | 201 | 400–600 nm |
| `egfp_emission.csv` | [EGFP](https://www.fpbase.org/protein/egfp/) | 18 | 238 | 463–700 nm |
| `mcherry_emission.csv` | [mCherry](https://www.fpbase.org/protein/mcherry/) | 80 | 251 | 550–800 nm |
| `mneptune_emission.csv` | [mNeptune](https://www.fpbase.org/protein/mneptune/) | 104 | 222 | 542–763 nm |

Each file has the header `wavelength_nm,relative_emission` and one wavelength
sample per row. The intensity values are dimensionless relative emission
values from FPbase, rounded to six decimal places for readable CSVs. The
curves have **not** been resampled onto detector channels, area-normalized,
or scaled by protein brightness. A simulation must sample them through its
chosen detector and set its own count scale. In particular, the `980_32ch`
detector does not cover the full red tails of mCherry and mNeptune.

Load one curve from the repository's `Python` directory with:

```python
import numpy as np

curve = np.loadtxt("lumic/simulation/fpbase_data/egfp_emission.csv", delimiter=",", skiprows=1)
wavelengths_nm, relative_emission = curve.T
```

## Attribution

FPbase asks users to attribute the original authors and recommends citing
Lambert, TJ (2019), *FPbase: a community-editable fluorescent protein database*,
Nature Methods 16, 277–278, [doi:10.1038/s41592-019-0352-8](https://doi.org/10.1038/s41592-019-0352-8).
See FPbase's [citation guidance](https://www.fpbase.org/cite/) and
[data policy](https://www.fpbase.org/privacy/). The protein pages above list
the original protein publications: Ai et al. (2007) for EBFP2
([doi:10.1021/bi700199g](https://doi.org/10.1021/bi700199g)), Cormack et al.
(1996) for EGFP ([doi:10.1016/0378-1119(95)00685-0](https://doi.org/10.1016/0378-1119(95)00685-0)),
Shaner et al. (2004) for mCherry ([doi:10.1038/nbt1037](https://doi.org/10.1038/nbt1037)),
and Lin et al. (2009) for mNeptune
([doi:10.1016/j.chembiol.2009.10.009](https://doi.org/10.1016/j.chembiol.2009.10.009)).
Those are protein references; the FPbase entries are the source of these
particular CSV curves.

FPbase's data policy makes database data available for commercial and
noncommercial use and asks users to credit the original authors. FPbase lists
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) on its
[license and policy page](https://www.fpbase.org/privacy/). These CSVs retain
that source notice and document the conversion and rounding above.
