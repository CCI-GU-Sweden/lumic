# Load spectral images

Lumic uses the same reader interface for ZEISS CZI and OME-TIFF. Start with
`open_image()` to inspect a file before loading its pixels, or use
`load_image()` when you already know which plane you want. A plain TIFF without
OME-XML is outside the OME-TIFF reader's scope.

## Download the public COS-7 example

Use `lumic.data.download_cos7_data()` to fetch the COS-7 acquisitions with
Lumic `0.1.0rc2` or later. See the [backend quickstart](quickstart.md) for
installation and the [COS-7 notebook](cos7-workflow.md) for a full unmixing run.

By default the function downloads only the high-SNR mixed CZI (about
145 MB). Use the returned path to load the image from a script or notebook.

```python
from lumic.data import download_cos7_data
from lumic.io import load_image

path, = download_cos7_data(progress=True)
image = load_image(path)
```

Set `snr="medium"` or `snr="low"` for another condition, or `snr="all"`
for all three mixed acquisitions. Set `include_references=True` to add the
three single-stain files for each selected condition. All twelve files
together take about 1.74 GB.

Downloads go to `~/.lumic/data/bioimage-archive/S-BIAD4448/` by default.
Pass `data_dir=` to choose a different folder. The downloader keeps
conditions in separate subfolders and reuses files that match the
archive's expected byte size. Set `progress=True` for percentage updates
while each file downloads. The command-line example shows progress by
default and accepts the same SNR and reference choices.

```bash
python Python/examples/download_cos7_data.py --snr all --include-references
```

The images are available under CC0. [BioImage Archive accession S-BIAD4448](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4448)

## Inspect and load a plane

Install Lumic using the [backend quickstart](quickstart.md). To try both
formats, run this example from the repository root. It uses a small CZI Z
stack and its paired OME-TIFF export under `Python/data/io-data/`. To use your
own files, replace the paths with a CZI or OME-TIFF that you have locally.

```python
from pathlib import Path

from lumic.io import open_image

fixtures = Path("Python/data/io-data")
paths = [
    fixtures / "980/ex405-range411_694-res8p8nm_zstack3.czi",
    fixtures / "980-ome-tiff/ex405-range411_694-res8p8nm_zstack3.ome.tiff",
]

for path in paths:
    with open_image(path) as reader:
        print(reader.summary())
        image = reader.read(t=0, z=1)

    print("Loaded plane:", image.axes, image.data.shape)
    print("Pixel sizes (µm):", image.pixel_sizes_um)
    print("Wavelength coordinates (nm):", image.wavelengths_nm)
    print("Detection bands available:", image.detection_bands_nm is not None)
```

The summary describes the available stack before selection. Both fixtures
have 32 detector channels and three Z planes. With zero-based indexing,
`z=1` loads the second plane. Both selected arrays have axes `CYX` and shape
`(32, 32, 32)`. The loaded image remains usable after the reader closes.

For a direct load, use the same selection arguments:

```python
from lumic.io import load_image

# Load all Z planes at the first time point from either file above.
stack = load_image(paths[0], t=0, z=None)
print(stack.axes, stack.data.shape)  # CZYX (32, 3, 32, 32) for this fixture.
```

The default is `t=0, z=0`. Use `None` to keep all positions along that
dimension. An explicit `series` argument selects an OME-TIFF image series
when the file contains more than one. Check `image.axes` before indexing,
since the axis order can differ across files.

## What metadata does Lumic return?

| Attribute | What it tells you |
| --- | --- |
| `image.axes` | The order of dimensions in `image.data`. `C` is detector channel, `Y` and `X` are lateral axes, `Z` is depth, and `T` is time. |
| `image.pixel_sizes_um` | Available physical pixel sizes in micrometres, keyed by spatial axis. Missing calibration leaves the relevant key absent. |
| `image.wavelengths_nm` | An optional wavelength coordinate for each detector channel. |
| `image.detection_bands_nm` | Optional lower and upper detection bounds in nanometres for each channel. |
| `image.provenance` | Source details such as file format, original axes and plane selection. For OME-TIFF it also records the series, stored channel names, and the source of the wavelength coordinates when available. |

**CZI.** Lumic reads recorded detection windows when available. For ordered,
non-overlapping windows with positive width, it uses the midpoint of each
window as the wavelength coordinate. Overlapping acquisition windows can
retain their bounds with `wavelengths_nm=None`.

**OME-TIFF.** Lumic reads spatial calibration from OME-XML and can get
wavelength coordinates from emission wavelength fields or channel names
that are all numeric labels such as `415 nm`. The paired OME-TIFF in this
example retains rounded wavelength labels but lacks detector window bounds.
Lumic's own OME-TIFF exports can preserve available detector bounds and
coordinates in a versioned annotation.

Treat missing spectral metadata as unknown. Detector channels are
measurements that can contain contributions from several fluorophores.
