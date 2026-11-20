# Microscopy I/O fixtures

Small real CZI acquisitions and paired OME-TIFF exports supplied by the microscope operator for testing
image loading, dimensions, channel order and acquisition metadata. These are
I/O fixtures, not demonstrations of reliable spectral recovery or calibrated
detector response. Files from different acquisitions need not contain matching
pixels, even if they cover similar sample regions.

## Provider and data dedication

These fixtures were acquired at Centre for Cellular Imaging (CCI), Gothenburg,
Sweden, with Rafael Camacho as acquisition operator. CCI owns the data and
dedicates its rights in the CZI acquisitions, paired OME-TIFF exports, reference
screenshots, recorded labels/pixel values, and metadata/reference JSON files
under **CC0 1.0 Universal**. See [the data dedication](LICENSE.md) for the exact
scope and terms. Attribution is appreciated but is not required by CC0.

## Collection

| Directory | Acquisition model | CZI files | Currently supported single-plane cases | Approximate size |
| --- | --- | ---: | ---: | ---: |
| `880/` | LSM 880 spectral detector | 11 | 8 | 1.87 MiB |
| `980/` | LSM 980 spectral detector | 15 | 12 | 13.44 MiB |
| `cd7-900/` | LSM 900 sequential window scan, acquired on a CD7 system | 4 | 2 | 29.56 MiB |

The 30 CZIs include 22 single-plane cases and eight stack/time/mosaic cases.
Z/time selections are supported by the explicit CZI reader introduced in PR #44;
single-scene mosaics now load as composed arrays using pylibCZIrw. The
legacy loader still expects one 2D image per channel.

[`980-ome-tiff/`](980-ome-tiff/README.md) adds four ZEN Blue 3.6 exports of the
original 32/18-channel images, a three-plane Z stack and a three-time-point
series. These are paired with the same acquisitions in `980/`, so every pixel
can be compared after accounting for axis order. The OME adapter validates those
quantitative exports and records which channel metadata survived conversion.

`980/` also contains the original `spectral-example-all-channels.czi` and
`spectral-example-subset-channels.czi` (32 and 18 channels), moved unchanged
from `data/io-example-lsm980/`. Their ZEN label screenshots/text and recorded
pixel values are retained beside them. Existing tests still use those
independent references; examples now read wavelength labels from this location.

Most new single-plane files have 8-bit pixels; the two original 980 fixtures
have 16-bit pixels. The 880 scan images are 8×8 pixels, and new 980/CD7 scan
images are 32×32. Values and physical sampling describe the supplied small
acquisitions, not recommended settings for scientific unmixing.

## Filename conventions

For example, `ex405_488-range411_694-res8p8nm.czi` indicates:

- `ex405_488`: excitation wavelengths used in the acquisition, in nm.
- `range411_694`: nominal range recorded by the operator; consult actual
  per-channel detection metadata for numerical analysis.
- `res8p8nm`: approximate spectral setting for the 880/980 acquisitions.
  It is not a measured channel transmission function.
- `zstack3`: three Z positions; `tseries3`: three time points.
- `tile2x2`: tiled acquisition, loaded as one composed array using stored tile positions.

The source filenames intentionally retain their original approximate labels
and suffix separators (`_` for 880/980, `-` for CD7). Do not infer channel count,
precise wavelengths or axis layout solely from a filename. Excitation settings
and detection gaps can change the actual set of stored channels.

## Acquisition software identifiers

The operator confirms that **`AIMApplication` identifies ZEN Black** in the
LSM 880 files. It must not be treated as an unknown producer merely because
its metadata name lacks the word ZEN. The supplied 980 files identify
`ZEN (blue edition)`, while the CD7/LSM 900 files identify `ZEN`. Record the
literal producer and version alongside a recognized family; do not infer an
edition from the plain `ZEN` name alone. Supported software aliases do not
establish support for every dimension layout or measurement model.

## LSM 900 sequential window scans (CD7 fixture system)

The operator confirms that this acquisition strategy belongs to the LSM 900
confocal scan head generally, not specifically to its integration with CD7.
It acquires spectral measurements repeatedly with changing detection windows
rather than using a conventional spectral detector array. CD7 identifies the
Celldiscoverer 7 platform that supplied these fixtures; `cd7-900/` retains that
provenance. Our file-validation evidence covers this supplied system, not
every LSM 900 acquisition configuration.
The nominal 450–650 nm range is the scanned boundary range, and `res66p6nm`
describes an approximate boundary step based on a 200 nm span divided by three.
It must not be interpreted as the bandwidth/resolution of each saved channel.

The stored metadata records paired windows at each boundary. For the
`res66p6nm` files the bounds, in saved channel order, are:

| Boundary position (nm) | First recorded window (nm) | Second recorded window (nm) |
| ---: | --- | --- |
| 450 | 400–450 | 450–650 |
| 516.666… | 400–516.666… | 516.666…–650 |
| 583.333… | 400–583.333… | 583.333…–650 |
| 650 | 400–650 | 650–650 |

Thus the file has eight recorded channels, including endpoint settings. The
`res40nm` file has six boundary positions (450, 490, 530, 570, 610, 650 nm),
giving twelve recorded channels. The final zero-width `650–650` range is
preserved exactly as metadata; it is not a measured response curve or proof
that the recorded values in that channel must be zero.

These overlapping window measurements are **not adjacent wavelength bins**.
Loading their channel arrays is supported, but the legacy
`getCZIchannelWave` midpoint output is not a validated spectral axis for them.
The tests deliberately do not present it as one. Future analysis needs an
explicit acquisition/band-response interpretation and must account for the
sequential nature of acquisition (for example, motion or bleaching). No new
LSM 900 window-scan unmixing or spectral-reconstruction algorithm is added by
these fixtures.

## Manifest and provenance

[manifest.json](manifest.json) records each CZI's:

- Relative path, byte count and SHA-256 checksum.
- Instrument/configuration and measurement model.
- Native axis names, sizes and mosaic flag.
- Detection-band bounds in stored channel order, without rounding.
- Available positive physical pixel sizes, converted to micrometres.
- For single planes: decoded shape, dtype and selected numerical pixels.

The manifest was frozen from the supplied CZI headers/XML and decoded pixels
using `aicspylibczi` on 2026-10-16. It supplies regression snapshots, **not an
independent ZEN measurement or optical calibration**. New scope acquisitions
still benefit from separately recorded ZEN channel/detection labels and pixel
values. The original 980 references provide that independent check for their
two corresponding files.

Array samples use zero-based `[channel_index, y, x]` coordinates. They must not
be compared directly with a ZEN cursor unless its coordinate/channel numbering
convention is known. Detection bounds are nm; physical sizes are µm. Float
band boundaries are kept even though the current spectral midpoint helper
returns integer nm and can lose sub-nm precision.

## Automated checks and remaining work

Run from `Python/`:

```bash
python -m pytest tests/test_czi_io.py tests/test_czi_instrument_fixtures.py
```

The new checks cover all 30 files' integrity/layout and detection metadata,
and exercise the legacy pixel loader for the 22 single-plane files. Spectral
midpoints are compared with recorded bands for the 880/980 cases only. Two
additional cases check the CD7 paired-window metadata against the operator's
description. Tests for the original ZEN labels and pixel samples remain intact.

The selective reader now tests full Z/time stacks and single-scene tile
composition separately. This does not validate new stitching/registration,
all Zeiss configurations or arbitrary open-format conversion. See the
[I/O guide](../../lumic/IO_README.md) for supported layouts and metadata.

## Repository handling

The collection is approximately 45 MiB including the original 980 CZIs;
individual files are below 8 MiB. Keep binary fixtures stable and update the
manifest deliberately when replacing a source. Windows `:Zone.Identifier`
sidecars are ignored; they are not acquisition data. No files have been cropped,
rescaled or stripped of acquisition metadata by the folder reorganization.

### Complete pixel references for the CZI backend migration

[`czi-pixel-reference.json`](czi-pixel-reference.json) freezes SHA256 hashes of
complete decoded arrays, their axes, shapes and dtypes for all 30 acquisitions.
References were recorded with aicspylibczi 4.0.1 before removing it: full T/Z
arrays for 27 non-mosaics and `read_mosaic(scale_factor=1)` for three mosaics.
Tests decode with pylibCZIrw and require exact agreement with those references.
No new image bytes are added. These are decoder regression checks, not independent
optical calibration. The original manifest and ZEN measurements are unchanged.
