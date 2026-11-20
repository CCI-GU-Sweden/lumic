# Lumic test suite

See [CONTRIBUTING.md](../../CONTRIBUTING.md#check-changes-locally) for environment
setup and commands for running the full suite or just the backend. Run pytest
from `Python/`, where [pytest.ini](../pytest.ini) directs discovery to `tests/`.

## Find and run a focused check

| Area | Start here |
| --- | --- |
| Unmixing, solver fallback, and convergence | [test_unmixing.py](test_unmixing.py), [test_batched_nnls.py](test_batched_nnls.py), [test_mcr_als.py](test_mcr_als.py) |
| Spectral selection and uniqueness | [test_convex_hull.py](test_convex_hull.py), [test_unique_profiles.py](test_unique_profiles.py), [test_uniqueness.py](test_uniqueness.py) |
| Image axes, metadata, and I/O | [test_spectral_image.py](test_spectral_image.py), [test_image_readers.py](test_image_readers.py), [test_czi_instrument_fixtures.py](test_czi_instrument_fixtures.py), [test_ome_tiff_export.py](test_ome_tiff_export.py) |
| Simulation and evaluation | [test_simulation.py](test_simulation.py), [test_spatial_scene.py](test_spatial_scene.py), [test_evaluation_metrics.py](test_evaluation_metrics.py), [test_residual_pca.py](test_residual_pca.py) |

```bash
# From Python/
python -m pytest tests/test_mcr_als.py
python -m pytest tests/test_czi_io.py tests/test_ome_tiff_export.py
```

## Fixtures and reproducibility

- Use small arrays with known answers for numerical checks. Set an explicit
  seed for randomized simulations and use tolerances suited to the quantity
  being compared.
- Use the [bundled CZI and OME-TIFF fixtures](../data/io-data/README.md) to
  check vendor layouts and metadata. Generate temporary exports with pytest's
  `tmp_path` fixture for round-trip tests.
- The suite runs offline. [FPbase lookups](test_fpbase.py) and
  [BioImage Archive downloads](test_example_data.py) use mocked responses.
  Keep new tests runnable with bundled fixtures or synthetic data.
- Run larger [evaluations and benchmarks](../evaluations/README.md) separately
  when measuring runtime or exploring sensitivity across cases.
