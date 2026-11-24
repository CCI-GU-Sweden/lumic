# Unmix a real experimental image (COS-7)

Use this notebook to unmix a spectral confocal image from the public COS-7
dataset. First fit the image with spectra selected from individual pixels.
Then estimate starting spectra from the convex hull, refine them with
MCR-ALS, and inspect the resulting abundance maps and reconstruction error.

[Download the notebook](minimal_example.ipynb) and open it in Jupyter with
Python 3.12. Install the backend candidate and its notebook dependencies in the
same environment as the Jupyter kernel.

```bash
python -m pip install "lumic[examples]==0.1.0rc4"
```

Run the notebook from top to bottom in a folder where you want the output
TIFFs. The first run downloads the high-SNR mixed CZI (about 145 MB), and
later runs reuse it. To use your own acquisition, change `IMAGE_PATH` and
choose reference pixels for that image.
[BioImage Archive S-BIAD4448](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4448)

![COS-7 cells in a detector-channel RGB preview, a fitted component overlay, and a grayscale reconstruction error map.](../assets/cos7-workflow.png)

The detector preview groups the measured channels into RGB. The fitted overlay
shows three selected abundance maps from the four-component blind fit.
The error map shows the mean squared residual per pixel. Its display scale
ends at the 99th percentile so the brightest pixels do not obscure smaller
errors. See [Evaluate experimental data](evaluate-results.md) for ways to
investigate these residuals.
