# Evaluation

After [unmixing](3-unmixing.md), examine the signed residual (observed signal
minus the fitted signal) for unexplained variation and spatial structure.

1. Set **Residual principal components** to the number of components to inspect.
2. Click **Calculate residual PCA** to calculate principal components of the
   residual and show a score map and loading curves.
3. Use the **Residual PC** slider or the **◀** and **▶** buttons to move between
   score maps. Look for spatial patterns in each map. The plotted loadings show
   the spectral shape of the corresponding residual variation.
4. Use **Plot PC loadings** to choose curves, and **Reset line emphasis** to
   show plotted curves with equal opacity. Moving the PC slider highlights its
   loading again; curve selection and score-map selection are independent.

The signed scores and loadings describe unexplained variation, not abundance.
PCA helps assess the relative importance and structure of residual variation;
the plugin does not provide a statistical significance test here.
