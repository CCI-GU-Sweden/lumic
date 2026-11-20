# Spectral images, CZI reading and OME-TIFF I/O

## Choose a reader automatically, inspect first

```python
from lumic.io import open_image, load_image

with open_image("image.czi") as reader:  # Also accepts supported OME-TIFF.
    print(reader.summary())             # No pixels decoded or automatic printing.
    # Only call read() when you choose to load data:
    image = reader.read(t=0, z=0)

# Or load directly when no interactive inspection is needed:
image = load_image("image.ome.tiff", t=0, z=None)
```

`open_image` dispatches .czi to `open_czi` and .tif/.tiff to `open_ome_tiff`;
the latter validates OME-XML and rejects plain TIFF. Unsupported extensions,
file contents and layouts raise clear errors. It returns the existing
format-specific reader directly, with its axes, shape, metadata and read method.
`load_image` uses that reader and closes it after loading or a read error.

All four direct functions (`open_czi`, `load_czi`, `open_ome_tiff`,
`load_ome_tiff`) remain available with unchanged signatures/defaults/results.
Both reader classes now have `summary()`, using shared formatting. The summary
reports stored detector-channel count, layout, calibration and channel metadata
availability. It validates neither fluorophore count nor suitability for a
particular unmixing model; pixel corruption can still be detected later on read.
Summary formatting uses only header information, which remains available after
close. No file-loading function prompts the user or prints automatically.

`open_image(path, series=1)` selects an OME image series. The default None uses
the existing first-series default. An explicit series option is rejected for
CZI (including zero), rather than being ignored or confused with scene selection.
T/Z selection on `load_image` follows its chosen adapter. Summaries describe the
available image/selected OME series, before any read-time T/Z selection.

The [inspection example](../examples/load_spectral_image.py) downloads the
public COS-7 CZI and defaults to `LOAD_PIXELS=False`. Edit that setting to
read pixels after inspecting the summary. Set `FILE_PATH` to inspect your
own CZI or OME-TIFF. The notebooks
[minimal_load](../examples/notebooks/minimal_load.ipynb) downloads the public
COS-7 CZI and compares its pixels and RGB preview with a temporary OME-TIFF
export. [minimal_metadata](../examples/notebooks/minimal_metadata.ipynb)
uses the same public data in a standalone notebook and explains the layout,
calibration, and detector metadata available before and after export.
The main [minimal workflow](../examples/notebooks/minimal_example.ipynb) also
uses the dispatcher with separate header-inspection and pixel-loading cells.
It defaults to the public COS-7 image. Select one CYX plane and review the
manual reference coordinates when changing acquisitions. From `Python/`:

```sh
python -m examples.load_spectral_image
```

For contributors, see the [reader contract and guide](../../docs/contributing/adding-an-image-reader.md)
and [executable template](../examples/reader_template.py). The template is not
registered as another supported format.


`SpectralImage` describes pixels already in memory. Readers populate it; it does
not open files itself. NumPy input, simulations, CZI and OME-TIFF readers use
the same contract. Existing `loadCZISpecData` / `getCZIchannelWave` calls keep
their behavior.

```python
import numpy as np
from lumic import SpectralImage
from lumic.io import load_czi, open_czi

# Provide axes explicitly for your own arrays; no inference from array shape.
image = SpectralImage(np.zeros((32, 10, 20)), axes="CYX")
image = load_czi("single_plane.czi")

# Inspect before decoding large acquisitions; select T/Z in the backend.
with open_czi("acquisition.czi") as reader:
    print(reader.axes, reader.shape, reader.dimension_ranges)
    image = reader.read()  # First CYX plane, with or without T/Z axes

print(image.axes, image.data.shape)
```

`reader.read()` defaults to `t=0, z=0`, returning the first CYX plane. Zero is
also accepted when T/Z is absent; requesting another index on an absent axis
raises IndexError. Indices are zero-based relative to the file's ranges.
Use `reader.read(z=None)` for all Z slices at the first time point, or
`reader.read(t=None, z=None)` for the whole acquisition. Full-dimension loading
must be requested explicitly.
`load_czi(path, t=0, z=2)` provides the same selection with automatic closure.
Arrays remain usable after closing. Use the context manager for deterministic
cleanup, including when an exception occurs. This is selected eager reading,
not a lazy array, chunk scheduler or memory-mapped image.

## Contract and interpretation

- Required: a real numeric NumPy array and unique axes containing C, Y, X with
  optional T/Z in any order for channel data. Component maps use K instead of C.
  No automatic transposition or intensity rescaling.
- C denotes measurement channels. Optional `wavelengths_nm` has one label per
  channel; optional `detection_bands_nm` has shape `(channels, 2)`. Wavelengths
  need not be uniformly spaced. Bounds can overlap or have zero width.
- `pixel_sizes_um` records available X/Y/Z calibration; `provenance` records the
  source path, format, native axes and selections. Missing metadata stays absent.
- Immutable `kind` distinguishes `"measured"` (default, including simulated
  detector data), `"reconstructed"` (predicted channel intensities) and
  `"abundance"` (estimated component contributions). Abundance images may carry
  `component_names`, but cannot carry wavelength/detection-band metadata.
- The object retains the pixel array without copying. Frozen attributes do not
  make the array or metadata dictionaries immutable. No full-array finite-value
  scan is performed; numerical functions remain responsible for their inputs.
- The CZI reader drops singleton T/Z and singleton backend bookkeeping axes,
  including selected axes, but always preserves C/Y/X even when length one.
  Retained axes keep backend order: time-series input may return TCYX, for example.
- OME-XML storage `DimensionOrder` is not a NumPy axes string.

CZI band midpoints are supplied only for positive-width, ordered non-overlapping
windows. Overlapping LSM 900 sequential windows retain their bounds and pixel
values but have `wavelengths_nm=None`. Loading them does **not** reconstruct a
conventional spectrum or validate their use in a linear spectral mixing model.
Bounds are metadata descriptions, not measured detector response curves.

CZI pixels and metadata now use **pylibCZIrw >= 6.1.0** only; Lumic no
longer requires aicspylibczi. Single images, Z stacks, time series and single-scene
tiled images are supported. Tiles form one array using their stored positions
at native resolution. Overlaps follow libCZI's highest-M-index rule; this is
composition, not new registration or blending. Missing regions are filled with
zero by the backend, so zero intensity alone does not identify missing coverage.
`image.provenance["mosaic_composed"]` records composition and
`spatial_origin_xy` retains the array's origin in CZI pixel coordinates, which
may be negative. Multiple independent scenes/views and RGB samples are rejected.

Public calls to `open_czi`, `load_czi`, `loadCZISpecData` and
`getCZIchannelWave` retain their signatures and return types. The array-only
`loadCZISpecData` expects a CYX image; use explicit T/Z selection in `load_czi`
for stacks. `loadCZIChannel` emits FutureWarning and accepts open Lumic or
pylibCZIrw readers, plus existing objects implementing the old `read_image`
interface. That compatibility path does not import aicspylibczi.

For future large-image processing, pylibCZIrw can read composed spatial regions
with `read(roi=(x, y, width, height), plane=...)`. Lumic does not yet expose ROI
loading. Python tile enumeration exposes headers, not uncomposed tile pixel
arrays. The underlying C++ libCZI can read/decode individual subblocks; exposing
that in Python is a possible future extension, not part of this migration.


## Save a quantitative OME-TIFF

```python
from lumic.io import save_ome_tiff, load_ome_tiff

image.save_ome_tiff("simulation.ome.tiff")  # Lossless Deflate by default.
# Equivalent standalone call: save_ome_tiff(image, "simulation.ome.tiff")
save_ome_tiff(image, "plain.ome.tif", compression=None)
save_ome_tiff(image, "lzw.ome.tif", compression="lzw")
restored = load_ome_tiff("simulation.ome.tiff", t=None, z=None)
```

`image.save_ome_tiff(...)` delegates to `lumic.io.save_ome_tiff(image, ...)`;
both share the same exporter, options and validation. The exporter accepts a
measured, reconstructed or abundance `SpectralImage`. It keeps
quantitative pixel values and supported dtypes unchanged: uint8/16/32,
int8/16/32, float32/64. Float simulation counts remain floats; no conversion to
uint16, normalization, clipping, or display scaling occurs. Deflate and LZW
are lossless for both integer and floating data. Noise compresses less well
than smooth data; a smaller file is not evidence of a better image.

| Information | OME-TIFF storage | What loading restores |
| --- | --- | --- |
| Pixels and dtype | Grayscale TIFF pages | Identical numerical values and precision |
| Spatial calibration | Standard OME PhysicalSizeX/Y/Z in µm | `pixel_sizes_um` |
| Detector channel names | Standard OME Channel.Name, when supplied in provenance | `provenance["channel_names"]` |
| Original axes, including permutations | OME storage order plus Lumic annotation | Original axis order; use T/Z=None for all planes/singletons |
| Measured/reconstructed kind | Lumic annotation | `image.kind` |
| Wavelength coordinates and detector windows | Lumic annotation | `wavelengths_nm`, `detection_bands_nm`; missing values stay missing |
| Source provenance, including simulated fluorophore names | Lumic annotation | `provenance["source_provenance"]` |

Lumic-specific fields use an image-linked OME MapAnnotation with namespace
`urn:lumic:spectral-image:1` for measured/reconstructed images, or
`urn:lumic:spectral-image:2` for abundance maps. Both contain a `spectral_image`
JSON entry. Existing version 1 files remain supported. Other
OME tools can read the ordinary pixels/calibration/channel names without knowing
this extension; they must interpret it to recover our extra fields. An unrelated
annotation is ignored. A malformed recognized annotation raises a clear error
before pixel decoding. Complete vendor XML and arbitrary OME metadata are not
copied or claimed to round-trip.

### Save abundance maps with their component spectra

```python
maps = unfolded.fold_abundances(
    abundances, component_names=["EBFP2", "EGFP", "mCherry"],
    spectra=spectra, normalization="L1", abundance_units="estimated detected counts",
)
maps.save_ome_tiff("abundances.ome.tif")
reloaded_maps = load_ome_tiff("abundances.ome.tif", t=None, z=None)
reconstructed = reloaded_maps.reconstruct()
```

**Pass the exact effective spectra used to fit these abundances**, including any
normalization applied by the solver. Neither folding, saving nor reconstructing
normalizes them. `normalization` and `abundance_units` are optional descriptive
strings; they do not enforce a scale. L1 spectra only make coefficients count
estimates if the data and acquisition model support that interpretation.

The TIFF stores standard OME **C** channels with component names (or anonymous
`Component 1`, etc.). K is only a Lumic axis meaning, restored from the version 2
annotation; it never appears in standard OME DimensionOrder. Ordinary readers
can display the maps as named channels without understanding the annotation.
They need to interpret it to recover component semantics and reconstruct.
If the annotation is removed or unrecognized, Lumic treats the file as ordinary
C-axis measured data; component-like names alone do not establish image kind.

The JSON entry records original `axes`, `shape`, `kind`, `provenance`,
`component_names` (list or null) and `unmixing` (object or null). Top-level
`wavelengths_nm`/`detection_bands_nm` are null for abundance images. Inside the
serialized unmixing object, `spectra` is a nested K × C array and `spectra_dtype`
is its NumPy `dtype.str` descriptor (for example, `<f4`); optional fields use null
when absent. Loading validates shape, finite spectra and representable values
before decoding pixels. The in-memory `unmixing` mapping contains these fields:

| Field in `maps.unmixing` | Meaning and use |
| --- | --- |
| `spectra` | Exact K × C basis, in map order; values and numeric dtype preserved |
| `wavelengths_nm`, `detection_bands_nm` | Optional detector C coordinates/windows for the reconstructed signal |
| `channel_names` | Optional detector C names, distinct from the K component names |
| `normalization`, `abundance_units` | Optional labels describing the paired factors' existing scale |

Abundance maps themselves have no detector wavelength/band arrays on K. They
retain spatial calibration and source provenance. Direct construction also
accepts `SpectralImage(data, axes="KYX", kind="abundance", unmixing={...})`.
This is optional metadata around ordinary arrays; solvers still use matrices.

Omit `spectra` to save/load maps alone. Calling `.reconstruct()` on such maps
raises a clear error. With a basis, reconstruction replaces K by C in the same
axis position and computes the fitted signal. It allocates the complete output;
it does not recover measured noise/residuals or establish unique signatures or
biological identity. No clipping or scale conversion occurs. Selecting T/Z with
`.slice()` copies the basis along with the selected image metadata.

The [spectral image workflow](../examples/spectral_image_workflow.py) demonstrates
export/reload of all three kinds and reconstruction using only the saved maps.

Fluorophores and detector channels are different: a simulated 32-channel image
may contain three fluorophores. Record the component list, e.g.
`provenance={"fluorophores": ["EBFP2", "EGFP", "mCherry"]}`, rather than assigning
one fluorophore to each mixed detector channel. Detector coordinate wavelengths
are also not automatically fluorophore emission maxima: we preserve their actual
meaning in our annotation rather than filling OME EmissionWavelength fields.

Source provenance must be JSON-compatible. NumPy arrays/scalars and Paths are
converted to lists/scalars/strings; tuples reload as lists. Unsupported objects
and non-finite metadata fail before creating an output. Pixel NaN/Inf values
are preserved separately as ordinary TIFF floating-point pixels.

Existing destinations raise FileExistsError; pass `overwrite=True` explicitly
to replace one. The parent directory must exist. Writes use a temporary file
in the same directory and publish after success, protecting existing outputs
from failed writes. `bigtiff=None` automatically chooses BigTIFF near the classic
TIFF 4 GiB limit, based conservatively on uncompressed bytes. `bigtiff=True`
requests it explicitly; False rejects arrays above that limit.

Initial scope is one in-memory measured/reconstructed image per file. Abundance
K images and RGB renderings are rejected; streamed/multiseries/pyramid exports
remain future work. No new dependencies are needed. The offline example
[`export_spectral_ome_tiff.py`](../examples/export_spectral_ome_tiff.py) generates
an EBFP2/EGFP/mCherry image on the approximate 980 detector and saves it under
`Python/examples/output/` (ignored by Git), alongside an RGB PNG made from the
reloaded image with `SpectralImage.to_rgb()`. The PNG is an 8-bit display
preview; quantitative spectral pixels remain in the OME-TIFF. Run from `Python/`:

```sh
python -m examples.export_spectral_ome_tiff
```

## OME-TIFF: the same image contract

Use the updated `environment.yml`: pixel I/O uses tifffile and imagecodecs;
ome-types parses and validates OME-XML using the existing lxml installation.
The adapter returns the same SpectralImage used by CZI, simulations and NumPy
input. AICSImageIO was unused and has been removed from the environment because
its dependency constraints kept tifffile at a 2023 version.

```python
from lumic.io import load_ome_tiff, open_ome_tiff

image = load_ome_tiff("image.ome.tiff")  # First CYX plane: t=0, z=0.
with open_ome_tiff("volume.ome.tiff", series=0) as reader:
    print(reader.axes, reader.shape)  # Header/metadata only; no pixels decoded.
    plane = reader.read(t=0, z=2)
    stack = reader.read(t=0, z=None)

unfolded = stack.unfold()
restored = unfolded.fold_spectra()
preview = plane.to_rgb()
```

Indices are zero-based. None keeps all positions; selected T/Z axes are omitted
and C/Y/X are retained. Lumic exports restore their source array order and
explicit singleton T/Z axes when requested with None. Third-party files omit
singleton T/Z axes. OME always declares T/Z extents, so index
zero works on a single-plane image. Only matching channel/time/Z pages are
decoded, also for compressed files; tiled pages are decoded in full, not as
sub-plane ROIs. No full-series read followed by slicing occurs. Returned arrays
own their pixels and survive file closure. Callers still choose block sizes.

Array axes come from tifffile's logical series layout, reconciled with the OME
dimension sizes. OME DimensionOrder is not copied as NumPy axes. The supplied
time-series export is CTYX while its CZI loads as TCYX; transpose by axis names
when comparing. `.read()` uses the logical page sequence incorporating TiffData,
not an assumed raw IFD ordering. `series=0` selects the first OME image; choose
another index explicitly for multi-series input.

### Metadata fidelity and limits

- Available physical X/Y/Z calibration is converted from explicit common length
  units to micrometres. Unsupported units or inconsistent sizes raise errors.
- Complete OME Channel.EmissionWavelength values are converted to nm when
  present. If all are absent, complete channel names consisting of a numeric
  nm label can supply approximate wavelength coordinates. The result records
  `wavelength_source` and original `channel_names` in provenance. Partial fields,
  ambiguous names or absent information leave wavelengths None.
- The supplied ZEN exports preserve rounded names such as `415 nm`, but omit
  the CZI detector-band bounds and explicit emission-wavelength attributes.
  Their label coordinates agree with original band midpoints within 1 nm.
  They are not precise recovered bounds or independent calibration. For these files the
  OME adapter leaves `detection_bands_nm=None`; it does not infer detector response
  from channel spacing, fluorophore names or the simulation profile.
- Valid OME-XML does not guarantee correct spectral interpretation or successful
  unmixing. Referenced spectra must still match the measured channel definitions.

Initial scope is single-file, non-pyramidal OME-TIFF or BigTIFF with scalar YX
pages. Losslessly compressed and tiled pages are tested. Multi-file datasets,
pyramids, RGB/sample images, missing planes and volumetric TIFF pages are rejected.
The API supports [quantitative OME-TIFF export](#save-a-quantitative-ome-tiff).
Generic non-OME TIFF loading, OME-Zarr access, and stitching are outside its
scope. The OME model stays on the reader as `.ome_metadata`.

The [round-trip example](../examples/load_spectral_ome_tiff.py) creates a
temporary OME-TIFF from the public COS-7 CZI and checks pixels and metadata.
See [export provenance](../data/io-data/980-ome-tiff/README.md) for the separate
ZEN fixture exports. From `Python/`:

```sh
python -m examples.load_spectral_ome_tiff
```

## Unfold, fit and restore the layout

For a readable, complete example starting with a simulated NumPy array, see
[spectral_image_workflow.py](../examples/spectral_image_workflow.py). It uses
offline FPbase spectra and a ternary mixture. Optional seeded shot noise makes
the distinction visible: restoration preserves all pixels, whereas a fitted
reconstruction does not reproduce every noise fluctuation. The script exposes
settings at the top, with plotting in a separate example helper. Run from
`Python/` with `python -m examples.spectral_image_workflow`; its saved
[comparison figure](../data/visual-check-spectral-image-workflow.png) includes
component maps, a residual RMS map and one pixel's measured/fitted spectra.
Set `ADD_SHOT_NOISE = False` to inspect the noiseless case, whose exact-reference
reconstruction should agree to numerical precision. This example uses known
spectra to demonstrate array handling; it is not blind component discovery.

The numerical API expects an observations-by-channels matrix. Use
`image.unfold()` for any valid channel-image axis order, including full Z/T
stacks. The result is an `UnfoldedImage`: `.data` is an ordinary NumPy matrix,
while `.source_axes` and `.source_shape` remember where the pixels came from.

```python
unfolded = image.unfold()
restored = unfolded.fold_spectra()  # Original pixels, axes, metadata and kind.
```

Once you have learned spectra (components × channels), unmix all observations
and restore either component maps or a predicted detector image:

```python
from lumic import positive_least_squares

# `spectra` is supplied or learned separately, in the image's channel order.
# Disable normalization here to keep its scaling unchanged for reconstruction.
abundances = positive_least_squares(unfolded.data, spectra, do_norm=False)
maps = unfolded.fold_abundances(abundances)  # kind="abundance"
predicted_signal = abundances @ spectra
reconstruction = unfolded.fold_spectra(predicted_signal)  # kind="reconstructed"
```

For example, a CZYX image with 32 channels and three 32×32 planes unfolds to
`(3072, 32)`. Four fitted components produce a `(3072, 4)` abundance matrix and
maps with axes KZYX and shape `(4, 3, 32, 32)`. The reconstructed detector image
has the original CZYX layout and 32 channels. The difference between measured
and reconstructed data is the residual; abundance maps describe contributions,
not residuals or detector intensities. Component names may be supplied with
`fold_abundances(abundances, component_names=[...])` in matrix column order.

The same `SpectralImage` class holds every image result. C is replaced by K at
its original axis position: YXC produces YXK, TZCYX produces TZKYX. Wavelengths
and detection bounds stay with channel images; spatial calibration and source
provenance are also retained for abundance maps. A no-argument `fold_spectra()`
preserves the source kind; passing a matrix explicitly marks a reconstruction.
Abundance images are outputs, so their `unfold()` is not a detector-input API.

### Ordering, memory and scope

- Rows enumerate the non-C axes in their original relative order, in NumPy C
  order: the last non-channel axis changes fastest. No pixel selection occurs.
- Both folding methods require all original rows in the same order. You cannot
  fold a reduced convex-hull pool into the full image. Learn spectra on that pool,
  then unmix the full loaded block before folding. Shape validation cannot detect
  reordered rows of the same size; keep correspondence intact.
- Views are used when possible, but moving/reshaping some axis orders requires
  a pixel copy. This is not lazy or out-of-core processing; no dtype conversion,
  normalization or full-array finite-value scan occurs in unfolding/folding.
- The user chooses which planes/blocks to read, which candidates to retain and
  whether spectra are shared across Z or T. Neither the reader, unfolding object
  nor convex-hull function automatically chooses a large-image strategy.
- Legacy `unfold_spec_data`, `fold_flat_spec_data` and `fold_abundance_matrix`
  keep their CYX array interfaces and results. Existing solvers still accept
  NumPy matrices; they do not need to know about image objects or axis layouts.

## Copy a plane and create an RGB preview

`image.slice(t=0, z=2)` copies pixels already in memory into a new SpectralImage.
The selected T/Z axes disappear; other axis order, image kind, channel/component
metadata and spatial calibration are preserved. Pixel storage and nested
provenance are independent, so editing the result does not edit the original.
Source provenance is retained with an `image_slice` record of the parent layout
and selection. Repeated slicing keeps the preceding record under `parent_slice`.

As with the reader, default T/Z selections are zero. An absent axis accepts zero
or None; another index is an error. `None` retains the full corresponding axis,
including singletons. `image.slice(t=None, z=None)` copies the whole loaded image.
This is integer T/Z selection, not channel/ROI selection or resampling. Request
planes from the reader first when the whole acquisition should not enter memory.

```python
plane = image.slice(t=0, z=2)
rendering = plane.to_rgb()
print(rendering.axes, rendering.data.shape)  # YXS for a CYX or YXC plane.
```

`to_rgb()` returns an **RGBRendering**, defined alongside the image classes in
`lumic/image.py`. Its uint8 `.data` has RGB last; `.axes` appends S (color samples)
to the non-C source axes. CZYX produces ZYXS; TCZYX produces TZYXS. Spatial axes
retain their relative order, so a source XY layout is not automatically YX.
Use the returned axes to choose and transpose planes for plotting. Creating RGB
data does not choose a Z/time position and does not change the spectral image.

This is a **channel-order preview, not spectral unmixing or calibrated true
color**. The first channel group maps to blue, middle to green, last to red,
assuming shorter-to-longer wavelength order. Each group contributes its mean.
No wavelength metadata is required or interpreted; unordered channels and
overlapping detection windows need not produce meaningful spectral colors.
The recorded `.channel_groups` lists original indices in R/G/B order. There is
no arbitrary grouping option in this first version. Measured and reconstructed
images are supported; abundance coloring requires explicit component colors and
is deliberately outside this method.

### Brightness and contrast stay linked across RGB

One linear transform is applied to all R/G/B values and every loaded Z/T plane.
There is no independent channel or per-plane normalization. Auto limits map
zero to black and the largest group mean in the entire loaded image to 255;
all-zero/nonpositive data render black with reusable limits `(0, 1)`. Constant
positive RGB data render white. Explicit limits use the same group-mean intensity
units as the input. Clipping and 8-bit rounding can alter extreme colors.

```python
stack_preview = image.to_rgb()
same_scale = plane.to_rgb(limits=stack_preview.limits)
fixed_scale = image.to_rgb(limits=(0, 1000))
```

RGBRendering keeps `.limits`, `.channel_groups`, calibration and provenance
(source metadata, source axes and source kind). It has no wavelength/unfolding
API and is not an image of fitted component contributions. Rendering processes
the whole loaded array and allocates RGB and temporary arrays; it is not lazy.
At least three finite channels are needed; signed values below the lower display
limit are clipped. The quantitative source data remains untouched.

The legacy `RGB_from_spec` still returns its original array and assumes CYX.
It now prints that assumption and suggests `SpectralImage.to_rgb()`. Its integer
averaging and `do_norm` behavior are unchanged. The new method averages in float
before display scaling, so weak signals do not disappear through early integer
truncation; display pixels need not exactly match the legacy helper.

The [RGB example](../examples/render_spectral_rgb.py) renders the public
COS-7 CZI by default. Set `FILE_PATH` to the bundled Z stack in the script
to compare three planes with shared limits, as in the
[saved stack preview](../data/visual-check-rgb-stack.png).
From `Python/`, run `python -m examples.render_spectral_rgb`.

See [the runnable CZI example](../examples/load_spectral_czi.py) and
[fixture documentation](../data/io-data/README.md). From `Python/`, run:

```sh
python -m examples.load_spectral_czi
```
