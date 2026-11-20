# Adding an image reader to Lumic

A format adapter translates file storage into a `SpectralImage`: numerical
pixels with explicit axes and optional metadata. It should not unmix data,
normalize intensities, decide fluorophore count or run a user interaction.

The dispatcher lives in [`lumic/io.py`](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/io.py). Existing CZI and
OME-TIFF readers remain directly usable; `open_image()` returns one of those
objects, not a second wrapper. The small structural interface lives in
[`lumic/readers.py`](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/lumic/readers.py).

## Start with the executable template

[`examples/reader_template.py`](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/examples/reader_template.py) demonstrates a
complete adapter without an additional backend dependency. It opens an NPY
memory map, inspects the header, copies selected pixels and closes its resource.
NPY does not declare spectral axes, so the template requires them explicitly.
It is a teaching adapter and **is not registered in the image dispatcher**.

From `Python/`:

```python
import numpy as np
from examples.reader_template import NpyReaderTemplate

# Known coordinates make time/channel/Z swaps easy to detect.
values = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)
np.save("example.npy", values)
with NpyReaderTemplate("example.npy", axes="TCZYX") as reader:
    print(reader.summary())              # Header information only.
    image = reader.read(t=1, z=0)         # CYX, one selected plane.
    volume = reader.read(t=1, z=None)     # CZYX, all Z at one time.

# Both results remain valid after the reader closes.
print(image.axes, image.data.shape)
```

When implementing another microscopy format, replace the template's backend
opening, header extraction, pixel selection and cleanup. Its use of a NumPy
memory map is an example, not a requirement or recommendation for all formats.
Do not substitute a guessed axis string for actual format metadata.

## Reader contract

`SpectralImageReader` is a typing `Protocol`. Adapters do not need to inherit
from it or any base class. A reader should expose:

| Member | Contract | Purpose |
| --- | --- | --- |
| `path` | Input `Path` | Identify the acquisition |
| `kind` | `"measured"`, `"reconstructed"` or `"abundance"` | State pixel meaning without inferring it from channel names |
| `format_name` | Human-readable format name | Consistent summary output |
| `axes`, `shape` | Validated available array order and matching dimensions | Inspect data before deciding how much to read |
| `wavelengths_nm` | One coordinate per C, or None | Plot/match spectral coordinates when known |
| `detection_bands_nm` | C × 2 recorded window bounds, or None | Retain detector measurement definitions |
| `pixel_sizes_um` | X/Y/Z calibration mapping; empty when unknown | Preserve physical scale |
| `summary()` | Return a string via `format_reader_summary(self)` | Present header information without printing or pixel I/O |
| `read(t=0, z=0)` | Return an independent `SpectralImage` | Explicitly load the requested pixels |
| `close()` | Release resources; safe to repeat | Avoid leaked file handles/mappings |
| `__enter__`, `__exit__` | Support `with`; close on exceptions | Make resource ownership clear |

Format-specific properties can remain available, such as OME series selection.
Keep them out of numerical solvers. If a future adapter needs additional options,
document which formats accept them; never silently ignore an unrelated option.

## Opening, validation and selection

1. Validate the file and inspect its metadata **without decoding pixel arrays**.
   Close acquired resources if construction fails at any stage. If a backend
   cannot inspect without eager decoding, document the limitation and resolve
   it before claiming this common metadata-first contract.
2. Read the actual axis meanings. C identifies measured/reconstructed detector channels;
   Y/X space and optionally Z/T volume/time. Array shape alone cannot distinguish
   them. OME DimensionOrder is storage metadata, not a NumPy axis string. Preserve
   order when practical and always label the actual output array order.
3. Decide supported layouts explicitly: scenes, tiles, RGB samples and unsupported
   dimensions must not be silently collapsed. Existing CZI single-scene mosaics
   compose stored tile positions; this does not perform new registration.
4. Default to the first T/Z plane. Zero is accepted for an absent/singleton axis;
   reject invalid index types, negative indices and out-of-range requests before
   pixel decoding. None requests the whole dimension. Match the adapter's
   reported axes/shape and document singleton handling. Existing CZI and external
   OME files omit singleton T/Z; explicit axes in the template and Lumic exports
   can retain them when None is requested.
5. Apply selections through the backend before decoding whenever possible.
   Report when a TIFF page/chunk must be read in full. Do not present an eager
   full-image read followed by NumPy slicing as selective file I/O.
6. Return pixels in their original quantitative units and supported dtype. Do
   not apply display scaling, clipping, channel normalization or guessed units.
   Results must survive closing the reader and must not be writable aliases into
   the source file. Copy backend views when needed.
7. Preserve known metadata; use None/empty mappings for unknown fields. Detector
   windows, emission peaks and fluorophore names are different concepts. A count
   of stored C channels does not count fluorophores. Record source/format/selection
   provenance, and state interpretation limits such as LSM 900 overlapping windows.

`summary()` reports that header/layout validation succeeded. It cannot prove
all pixel blocks are uncorrupted or that the acquisition meets a linear mixing
model. A later `read()` may still raise a decoding error. It should not prompt,
print automatically, infer an optimal rank or assert biological identities.

## Registering the adapter

1. Add explicit format-specific opening/loading functions, consistent with
   `open_czi`/`load_czi` and `open_ome_tiff`/`load_ome_tiff`. Keep dependencies
   proportionate and use lazy imports where appropriate.
2. Add a documented suffix branch to `open_image` in `lumic/io.py`. The suffix
   selects the adapter; the adapter must validate the actual file contents.
   Ambiguous `.tif`/`.tiff` files currently require OME metadata; arbitrary TIFF
   images are not assumed to be spectral.
3. Keep `load_image` delegating to `open_image` and then `read`. Do not duplicate
   pixel/metadata logic, summary formatting or algorithms in the dispatcher.
4. Add a small shared fixture with known values and attribution. Update the I/O
   guide and example settings. Existing format-specific calls must still work.

This is an explicit dispatcher, not an entry-point/plugin system. Revisit that
architecture only when independently distributed adapters justify it.

## Validation before proposing a PR

Use the existing [`test_image_readers.py`](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/tests/test_image_readers.py) as a
reference. Include tests with independent expected values, not only comparisons
between two entry points that share an implementation:

- Header open and summary decode no pixels and emit no automatic output.
- Axes, shape, channel count and optional metadata match the fixture.
- Coordinate-encoded non-square arrays detect C/T/Z/Y/X swaps; check dtype and
  exact pixels, including singleton axes and requested full dimensions.
- Backend spies confirm selective decoding; invalid selections fail before reads.
- Context/constructor/read failures release resources; close is idempotent and
  returned arrays survive close without modifying the source file.
- Invalid contents, unsupported layouts and wrong format-specific options produce
  clear errors, without silently dropping a dimension or selecting another scene.
- Dispatcher and direct-format entry points agree on metadata/pixels, and existing
  direct-reader tests continue to pass.

Run from `Python/`: `python -m pytest`. Document what is actually supported,
which metadata is absent/approximate and which layouts remain future work.

## Derived abundance files

A recognized derived-image annotation may identify fitted component maps as
`kind="abundance"` with K/Y/X and optional T/Z. OME-TIFF itself stores these as
ordinary C channels; its Lumic version 2 annotation restores K semantics and an
optional exact spectral basis. Keep detector coordinates inside the basis, not
on the component axis. The shared summary reports Components for K images.
Unknown annotations must not cause an adapter to guess abundance kind from
channel names. The teaching NPY adapter remains explicitly measured data.
