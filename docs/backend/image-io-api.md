# Image I/O API

Lumic supports metadata-first inspection and explicit pixel reads for CZI and
OME-TIFF files. Open a reader when you want to inspect axes, dimensions, or
metadata before decoding; use a `load_*` function when you want an in-memory
`SpectralImage` immediately.

## Generic entry points

`open_image` dispatches by filename extension: `.czi` files use the CZI reader
and `.tif`/`.tiff` files use the OME-TIFF reader. TIFF files must have supported
OME metadata; Lumic does not guess the meaning of plain TIFF files. The generic
functions return or load through the existing format-specific readers, whose
signatures remain available below.

```python
from lumic.io import load_image, open_image

with open_image("acquisition.czi") as reader:
    print(reader.summary())  # Header information only; no pixel array loaded.
    image = reader.read(t=0, z=2)

# Convenience form: read the default first T/Z plane and close the reader.
image = load_image("acquisition.ome.tif")
```

Calling `open_image` validates the path and reads the format header and layout.
It does not decode pixel data. `reader.read(...)` decodes the requested pixels
into an independent `SpectralImage`, which remains usable after the reader is
closed. A `with` block closes the reader on exit, including when reading raises
an exception. If you keep a reader open without a context manager, call
`close()` yourself.

By default, `read()` and the `load_*` helpers select `t=0, z=0` when those axes
exist. Pass `t=None` and/or `z=None` to retain all positions along that axis;
the result then keeps the corresponding T/Z axes. Use `summary()`, `axes`, and
`shape` to inspect available dimensions before choosing a selection.

::: lumic.io.open_image
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.io.load_image
    options:
      show_root_heading: true
      show_root_full_path: false

## Format-specific entry points

Use these functions when the format is known or when the format-specific
options are useful. CZI has no `series` option. OME-TIFF supports selecting a
series (default `0`). Each `open_*` function returns its native reader; each
`load_*` function reads selected pixels and closes that reader before returning.

::: lumic.io.open_czi
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.io.load_czi
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.io.open_ome_tiff
    options:
      show_root_heading: true
      show_root_full_path: false

::: lumic.io.load_ome_tiff
    options:
      show_root_heading: true
      show_root_full_path: false

## Reader contract

Both format readers implement the `SpectralImageReader` protocol. Its
`summary()` presents validated header metadata without loading pixels;
`read(t=..., z=...)` returns an independent `SpectralImage`; `close()` releases
the reader's resources and is safe to call repeatedly. Readers are context
managers, so prefer `with open_image(...) as reader:` when reading selectively.
The `axes`, `shape`, and optional calibration and wavelength attributes
describe available metadata. A successful header check does not guarantee
that every pixel block is readable or that the data are scientifically
suitable for spectral unmixing.

::: lumic.readers.SpectralImageReader
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - path
        - format_name
        - kind
        - axes
        - shape
        - wavelengths_nm
        - detection_bands_nm
        - pixel_sizes_um
        - summary
        - read
        - close
        - __enter__
        - __exit__

## COS-7 example data

`download_cos7_data` fetches curated public CZI acquisitions from BioImage
Archive accession S-BIAD4448. By default it downloads the high-SNR mixed
acquisition (about 145 MB). `include_references=True` adds the single-stain
reference acquisitions; `snr="all"` includes high, medium, and low conditions.
Files are stored under `~/.lumic/data/bioimage-archive/S-BIAD4448` unless
`data_dir` is supplied. Existing complete files are reused.

```python
from lumic.data import download_cos7_data
from lumic.io import open_image

path, = download_cos7_data(snr="high", progress=True)
with open_image(path) as reader:
    print(reader.summary())
```

Downloading all conditions with references requires about 1.74 GB.

::: lumic.data.download_cos7_data
    options:
      show_root_heading: true
      show_root_full_path: false
