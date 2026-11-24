# Image containers API

`SpectralImage` stores an already loaded NumPy image with its actual axis order
and optional detector and spatial metadata. It does not open files, infer axes,
or run unmixing. `unfold()` turns detector data into an observations-by-channels
matrix while retaining enough layout information to fold results back.

## Axes and metadata

Measured and reconstructed images use `C` for detector channels; abundance
images use `K` for fitted components. All images include `Y` and `X`, with
optional `T` and `Z` axes. `axes` describes the array's actual dimension order,
so for example `CZYX` means channel, depth, row, column. Array values and dtype
are retained; no normalization or resampling is performed.

`wavelengths_nm` and `detection_bands_nm` describe detector channels and must
match the `C` dimension. They are not attached directly to abundance `K` maps.
Abundance images can carry component names and an optional `unmixing` mapping
with the exact `K × C` spectra and their detector metadata. `pixel_sizes_um`
records available spatial calibration. Metadata is copied when a container is
created, though NumPy arrays and mapping values remain mutable.

## Practical flow

```python
from lumic import SpectralImage

# A measured volume with four detector channels and three Z planes.
image = SpectralImage(data, axes="CZYX", wavelengths_nm=wavelengths)
plane = image.slice(z=1)       # CYX; copies the selected plane.
unfolded = plane.unfold()      # (Y * X, C), one observation per pixel.

# After fitting a full matrix in the same row order:
maps = unfolded.fold_abundances(
    coefficients, component_names=["A", "B"], spectra=effective_spectra,
    normalization="L1",
)
restored_signal = maps.reconstruct()
```

When `T` or `Z` remains in the image, unfolding includes those positions as
observations in NumPy C order. If you fit all observations together with fixed
references, the same reference matrix is used for every observation.
Fold methods require every original observation in its original
row order; shape checks cannot detect reordered rows. For reader-based selective
loading and ownership, see the [Image I/O API](image-io-api.md). For full
workflows, see [Unmix a ternary image](unmix-ternary-image.md) and
[Save images and spectra](save-results.md).

## Loaded images

::: lumic.image.SpectralImage
    options:
      show_root_heading: true
      show_root_full_path: false
      merge_init_into_class: true
      separate_signature: true
      members:
        - data
        - axes
        - wavelengths_nm
        - detection_bands_nm
        - pixel_sizes_um
        - provenance
        - kind
        - component_names
        - unmixing
        - slice
        - to_rgb
        - unfold
        - save_ome_tiff
        - reconstruct

## Fold results into images

`UnfoldedImage.data` has shape `(number of observations, C)`. It retains
`source_axes` and `source_shape` so it can restore the original dimension order.
`fold_spectra()` restores detector-channel pixels as a `SpectralImage`;
`fold_abundances()` replaces `C` with `K` and returns an abundance image.
Optional `spectra` must be the effective `K × C` basis used for the supplied
coefficients if later reconstruction is needed.

::: lumic.image.UnfoldedImage
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - data
        - source_axes
        - source_shape
        - wavelengths_nm
        - detection_bands_nm
        - pixel_sizes_um
        - provenance
        - source_kind
        - fold_spectra
        - fold_abundances

## Display rendering

`to_rgb()` returns `RGBRendering`, a display-only preview rather than measured
spectral data. It averages groups of source channels into blue, green, and red
in source order, assuming channels are ordered from shorter to longer emission
wavelength. The method does not inspect wavelength metadata or accept abundance
maps. The returned array is `uint8`, appends an `S` axis of length three, and
records the source groups and shared intensity limits. For example, `CYX`
becomes `YXS`, while `CZYX` becomes `ZYXS`; select a plane explicitly when
displaying a stack. Supply `limits=(low, high)` to use the same scale for
comparable images.

::: lumic.image.RGBRendering
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - data
        - axes
        - channel_groups
        - limits
        - pixel_sizes_um
        - provenance
