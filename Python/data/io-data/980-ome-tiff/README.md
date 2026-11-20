# Paired LSM 980 OME-TIFF exports

The microscope operator exported these files from the matching CZIs in
[`../980/`](../980/) using **ZEN Blue v3.6 → OME TIFF-Export**. The original
operator's export information is recorded here. These are quantitative exports,
not RGB previews; all supplied exports are uncompressed.

The acquisitions and exports were provided by Centre for Cellular Imaging
(CCI), Gothenburg, with Rafael Camacho as acquisition operator. The TIFFs are
included in the fixture collection's [CC0 1.0 data dedication](../LICENSE.md).

| OME-TIFF basename | Logical axes | Shape | Dtype |
| --- | --- | --- | --- |
| `spectral-example-all-channels.ome.tiff` | CYX | 32×32×32 | uint16 |
| `spectral-example-subset-channels.ome.tiff` | CYX | 18×32×32 | uint16 |
| `ex405-range411_694-res8p8nm_zstack3.ome.tiff` | CZYX | 32×3×32×32 | uint8 |
| `ex405-range411_694-res8p8nm_tseries3.ome.tiff` | CTYX | 32×3×32×32 | uint8 |

The collection is approximately 445 KiB. Original TIFF contents are unchanged.

## What conversion preserved

`tests/test_ome_tiff_io.py` compares every pixel against the original CZI,
accounting for logical axis order. Values, bit depth and channel order are
preserved exactly; physical calibration agrees within exporter precision.
The CZI reader returns TCYX for the time series while this OME-TIFF reports
CTYX. Equal content does not require equal array dimension order.

## What metadata is missing or approximate

These exports have channel names such as `415 nm`, but no explicit OME
Channel.EmissionWavelength values and no original CZI DetectionWavelength
bounds. The adapter recognizes complete numeric nm labels and records
`OME Channel.Name (numeric nm label)` as their source. These rounded labels
agree with CZI band midpoints within 1 nm; they are not recovered precise
band centers or independent wavelength calibration. Detector bounds remain
None. No values are imported from the CZI during ordinary OME-TIFF loading.

Missing or ambiguous wavelength metadata is allowed. Do not infer spectral
band widths from channel spacing or treat a fluorophore name as a detector
wavelength. Channel definitions must still match any supplied reference spectra.

## Use

From `Python/` with the updated environment:

```bash
python -m pytest tests/test_ome_tiff_io.py
```

The [standalone OME-TIFF example](../../../examples/load_spectral_ome_tiff.py)
shows how to create and reload an export using the public COS-7 acquisition.

Synthetic temporary files cover combined T/Z layouts, lossless compression,
tiled BigTIFF, singleton axes, multi-series selection and invalid inputs. They
are generated during tests; no large synthetic fixture files are versioned.
Multi-file datasets, pyramids, RGB/sample images and volumetric TIFF pages are
outside this first adapter's supported scope.
