# Data-driven recovery evaluation

For the simulation API, array layouts, count conventions and runnable offline
examples, see the [simulation building-block guide](../lumic/simulation/README.md).
This page describes the evaluation experiments built from those tools.

## Standalone runs and provenance

Run any evaluation directly from `Python/` with
`python -m evaluations.<script_name>`. Each script accepts `--output-dir PATH`.
The fitting-pool robustness study and the three real-image follow-ups generate
their prerequisite fits/reports automatically; no prior evaluation command is
required. Their default results directories are separate, so running one script
does not overwrite another script's default results. Preparing a full real-image
chain takes several minutes and writes several GB of local arrays.

By default, prerequisites are rebuilt for each invocation using the current
configuration. To deliberately reuse a completed run, pass
`--reuse-prerequisites --output-dir PATH`. Reuse checks the required artifacts
and the prerequisite's completion manifest, and rejects a different `--input`.
It does not certify that saved settings match every current code/configuration
change: choose reuse only for the saved experiment you intend to inspect.
Older outputs without completion manifests must be regenerated once.
Use different output directories for concurrent runs of the same evaluation.

Each script writes `run_<script_name>.json` alongside its results, recording
the Git commit and working-tree status (when running from a checkout), Python
and dependency versions, arguments, serializable top-level settings, elapsed
time, and whether execution completed, failed, or was interrupted. Prerequisite
stages have separate manifests in the same results tree. Existing experiment
settings, scientific metrics, seeds, and solver stopping diagnostics remain in
their usual files. `completed` means the script finished, not that every fit
converged or recovered the true factors. The noise and binning sweeps also print
completed/failed case counts and iteration-cap counts in the terminal; expected
per-case SVD failures remain recorded results and do not change the exit code.

Run the noiseless binary titration from the `Python` directory:

```bash
conda activate unmixing-env
python -m evaluations.evaluate_hidden_component
```

Edit the settings near the top of `evaluate_hidden_component.py` to choose any
two of the four saved FPbase spectra (`EBFP2`, `EGFP`, `mCherry`, `mNeptune`) and
the maximum fraction of the second component in each dataset. The first
component always has a pure pixel. A maximum fraction of `1.0` is the control
with both pure pixels; every smaller value hides the second pure spectrum.
Each 20 × 20 image has the same total expected counts at every pixel and no
noise. The saved curves are binned with the approximate `980_32ch` detector.

The script gives only the generated spectral image to convex hull selection,
then unique profiles, then MCR-ALS. Ground-truth fractions and spectra are used
after fitting to match component order and calculate errors. Results are saved
under `results/binary_hidden_component/`:

- `summary.csv`: one row per maximum hidden fraction, with candidate selection,
  spectral errors, concentration RMSE, and reconstruction error.
- `component_metrics.csv`: one row per component and titration level, comparing
  selected-start and fitted spectral error, concentration RMSE in counts, and
  concentration error relative to that component's true signal. It also records
  the matched component indices and spectral matching costs.
- `summary.md`: a compact table of the main results for reading in an editor.
- `overview.png`: errors across the titration. The global spectral error is the
  relative L2 distance between recovered and true spectra after each is
  normalized to unit L2 length; the largest channel error is the greatest
  absolute difference between those normalized spectra. These match the two
  measures used in the MCR-ALS recovery test.
- `case_*.png`: true and recovered spectra, true and recovered concentration
  maps, and signed concentration error maps for each component. The spectra
  panels also show the convex-hull/unique-profiles starting spectra before
  MCR-ALS.
- `optimization_history.png`: relative lack of fit at initialization and after
  each completed MCR-ALS iteration, up to 500. The run stops early when the
  relative fit improvement is at most 0.1% or reaches numerical precision.
  Values near machine precision can vary from round-off.
- `component_recovery.png`: spectral and concentration errors for every
  component before and after MCR-ALS across the titration.

The shared metrics in `lumic.evaluation` match estimated components to truth
by spectral shape, then apply the same order to their concentration maps.
Spectra are scaled to unit sum, and the paired concentrations are rescaled so
the image reconstruction is unchanged. Spectral relative L2 compares unit-L2
shapes. Concentration RMSE uses expected counts; concentration relative L2
divides each component's error by that component's true count signal. When a
true component is absent, its relative concentration error is undefined and
recorded as `NaN`, while its count RMSE remains available.
The summaries record the stopping reason and completed iteration count.
`converged` means the fit met the stopping rule; it does not establish that
the recovered spectra are correct. The solver returns its best paired factors,
which can come from an earlier iteration than the final history point.

Run the script to generate the case figures, overview, optimization history,
and CSV summaries.

With no pure pixel of the second component, its exact spectrum may not be
identifiable from these mixtures. More than one nonnegative pair of spectra
and concentrations can reconstruct the same noiseless image. A small fit
residual therefore does not by itself establish recovery of the true spectra
or concentrations. The results directory is ignored by Git so exploratory
outputs do not accumulate in commits.

## Three-component evaluation with EGFP hidden

Run the EBFP2/EGFP/mCherry evaluation from `Python`:

```bash
conda activate unmixing-env
python -m evaluations.evaluate_ternary_hidden_egfp
```

This uses the same noiseless 20 × 20 image, detector profile, candidate
selection workflow, and MCR-ALS limit of 500 iterations with a 0.1% fit-change
tolerance. Along the image's bottom edge, EBFP2-to-mCherry mixtures include
pure pixels of both.
Moving up increases EGFP linearly to the configured maximum while preserving
the EBFP2:mCherry ratio. Therefore, below the 100% control, the data contain
no pure EGFP pixel. Results are written to
`results/ternary_hidden_egfp/`, separate from the binary evaluation.

The output set mirrors the binary evaluation: a CSV and readable summary,
one figure per EGFP titration level, an overview, and an MCR-ALS history plot.
Each case plots true, selected-start, and post-optimization spectra where
available, plus true and estimated component-count maps and their signed
errors. If MCR-ALS fails during an NNLS update, the evaluator records the error
and shows the selected-start fit instead; post-optimization metrics are left
blank rather than being reported as recovered results.


## Four-component radial-spoke blur experiment

Run the noiseless spatial benchmark from `Python`:

```bash
conda activate unmixing-env
python -m evaluations.evaluate_radial_spoke_blur
```

During fitting, a progress bar shows the current PSF FWHM and completed
MCR-ALS iterations out of the configured maximum. If the solver converges
early, the bar stops below 100% and shows the stop reason.

Twelve narrow spokes radiate from the center. A fixed seed assigns each spoke
one of the saved EBFP2, EGFP, and mCherry spectra, ensuring that all three are
represented. A weak, spatially uniform fourth component has a synthetic broad
Gaussian-like spectrum centered at 500 nm. All four spectra are integrated by
the approximate `980_32ch` detector. The same sample is rendered at zero and
several progressively larger Gaussian PSF widths; no noise is added. These
widths are controlled stress-test settings, not a calibration of an objective.

The script passes the full convex-hull candidate pool to `unique_profiles`,
which selects four starting spectra. MCR-ALS then uses a fixed seeded sample
of detector pixels plus those selected pixels; it does not fit every hull
candidate, since the size of that pool changes with blur. The workflow never
receives the true spectra. Ground truth is used afterward for component
matching and separate reconstruction, spectral, and abundance error reports.
The per-case figures show the true, selected-start, and fitted spectra beside
true and recovered maps. `overview.png`, `summary.csv`, and `summary.md` show
how fluorophore purity and recovery change with blur. Outputs go to
`results/radial_spoke_blur/`.

Because the background is present everywhere, a spoke pixel is not exactly
pure when background counts as a fourth component. The report counts pixels
where a *fluorophore* supplies at least 90%, 95%, or 99% of all counts,
including background in the denominator. It also reports fractions among only
the three fluorophores. Background-only pixels do not count as pure
fluorophore pixels. The foreground statistic requires fluorophore signal at
least as large as the background count level, so tiny blur tails are not
reported as pure. Good reconstruction alone does not establish accurate
spectra or concentrations.

The reference run allows up to 2,000 MCR-ALS iterations. The zero and 0.30 µm
cases meet the stopping criterion; the stronger-blur cases reach the limit.
Compared with a 500-iteration run, the extra iterations change the true
spectral and abundance errors very little, despite small improvements in
reconstruction error. Convergence therefore does not establish factor accuracy.

## MCR-ALS solver performance

Run a short comparison of the original per-observation NNLS solver and the
automatic batch solver:

```bash
conda activate unmixing-env
python -m evaluations.benchmark_mcr_als
```

The benchmark uses the 0.65 µm spoke scene with the same selected spectra and
seeded fitting subset as the blur evaluation. Both solvers run 30 ALS updates
without early stopping; wall time covers the optimizer only. It checks that
concentrations, spectra, and fit histories agree, then writes timing and
accuracy tables to `results/mcr_als_performance/`. Timings depend on
hardware and library versions.

`mcr_als` now defaults to `solver="auto"`. For up to six well-conditioned
components it solves the possible nonzero component subsets in batches,
minimizing the same nonnegative least-squares objective as the original
solver. Larger or degenerate systems retain the original NNLS/BVLS path.
For more than six components, ALS prints an informational message once per
run explaining that the cutoff limits the exponential number of subsets
and can be revisited with benchmarks.
Use `solver="nnls"` to compare against the original implementation. The
public `positive_least_squares` helper keeps its original default and allows
the batch path through `solver="auto"`.

## Weak-background recovery diagnosis

Run the noiseless diagnostic from `Python`:

```bash
python -m evaluations.evaluate_background_recovery --compare-random-subset
```

It reuses the radial-spoke scene at zero and 0.65 µm blur and compares fixed
true-spectrum abundance fitting, ALS initialized with true spectra, and the
existing blind selection/refinement workflow. The default ALS fitting pool
is every convex-hull candidate, matching the intended image workflow. Both
ALS paths use identical hull pixels. The optional `--compare-random-subset`
reruns the previous 512-random-pixel-plus-starts variant as a labelled control
and creates `pool_comparison.csv`, `.md`, and `.png`. Ground truth enters only the two diagnostic controls and
subsequent evaluation, never the blind selection.

User settings at the top cap prototype runs at 500 iterations, with restarted
checkpoints at 0, 50, 200, and 500. Each checkpoint returns the best factors
within that run; early stopping and executed iteration counts remain visible.
Figures separate full-image recovery from the fitting-subset residual history.
Tables record per-component spectral errors, maximum channel error, count
RMSE, and foreground/background-region errors. The complementary background
region may include faint blur tails. Undefined relative errors are NaN.

Outputs go to `results/background_recovery/`.
Use `--output-dir PATH` to choose another destination. Exact fitting indices,
scene settings, and true detector spectra are saved in `settings.json`.
This diagnoses background distortion without new simulator features, optimizer
changes, weighting, or constraints. Truth-start stability does not guarantee
blind recovery, and low residuals do not establish unique factors.

The initial hull comparison retains one background-region pixel in each blur
condition, versus 456 and 215 in the random-subset controls. The comparison
report records this change alongside recovery metrics. It is a sampling
hypothesis to investigate, not proof that weighting will resolve ambiguity.

The diagnostic also saves `residuals_*.png`: observable reconstruction residual
RMS per channel beside background concentration errors that require truth.
The raw residuals are displayed on a shared logarithmic scale and are not
noise significance scores. Adaptive residual-guided sampling, noise scaling,
and superpixel grouping remain follow-up ideas in the working plan; this
script does not implement them.

## Shared image error maps

`lumic.evaluation.reconstruction_error_maps(observed, reconstructed)` works
on real and simulated data without truth. Arrays have channels on the last
axis, for example `(height, width, channels)` or `(pixels, channels)`.
It returns signed channel residuals (measured minus reconstructed), per-pixel
RMS in input intensity units, and dimensionless relative L2 residuals.
Use channel residuals to inspect wavelength-specific structure, RMS for absolute
mismatch, and relative residuals to compare brightness levels. `min_signal`
masks relative results where measured spectral L2 norm is too small: excluded
values are NaN, with an explicit `relative_valid` mask.

```python
from lumic.evaluation import reconstruction_error_maps, concentration_error_maps

# C has shape (pixels, components); S has shape (components, channels).
fit_maps = reconstruction_error_maps(image, (C @ S).reshape(image.shape),
                                     min_signal=1.0)
```

`concentration_error_maps(true_C, true_S, C, S)` requires known ground truth,
normally from simulations. Components are matched and both factor pairs put
on a unit-sum spectral scale, following `evaluate_factors`. Abundances have
components on the last axis and may retain spatial dimensions. The result
contains signed errors (recovered minus true), absolute errors, and signed
fractional errors, all in true-component order. Counts have physical meaning
only when the image uses count units. `min_concentration` masks relative errors
at or below the true-concentration threshold, while absolute errors remain
available to reveal false positives in absent components.

```python
truth_maps = concentration_error_maps(true_C, true_S, C, S,
                                      min_concentration=1.0)
percent_error = 100 * truth_maps.relative
```

Small reconstruction residuals do not establish correct factor recovery.
Concentration errors cannot be calculated on ordinary real images without
independent truth. Neither function returns uncertainty or noise significance;
shot-noise scaling follows in the noisy simulation work. Their docstrings
describe formulas, signs, units, masks, uses, and limitations.

The background, radial-spoke, binary/ternary hidden-component, and overlapping-
circle figures use the shared concentration maps. The minimal and spatial
notebooks use shared RMS squared to retain their existing MSE maps (squared
intensity units). These notebook source changes preserve saved outputs;
rerunning the full real-data notebooks still requires their input images.

## Convex-hull fitting-pool expansion

From `Python/`, run:

```bash
python -m evaluations.evaluate_fitting_pool_expansion
```

This reuses the noiseless four-component spoke scene at zero and 0.65 µm
blur. It compares hull-only fitting, hull plus 64/256/512 additional non-hull
pixels, and full-image fitting. Every pool retains all hull pixels; random
additions are nested within each of three fixed seeds. Starts and solver
settings are identical within each blur condition (L1, tolerance 0.1%, up to
1000 updates, solver auto). The smaller pool counts refer to added pixels,
not the total pool size. Truth is used only to evaluate recovery and label
foreground/background representation, never to select pixels or initialize ALS.

Outputs go to `results/fitting_pool_expansion/`, or choose `--output-dir PATH`.
Figures show all component recovery curves and background spectra/error maps
beside observable residual maps. Curves summarize median and range across
sampling seeds, not statistical confidence intervals. Preview maps always use
the first declared seed. CSV files contain individual runs, initial metrics,
regional errors, timing, and stopping diagnostics; JSON records exact pool
indices, starts, histories, scene settings, and software versions.

This is a sampling experiment, not a new solver or adaptive pixel-selection
API. More pixels may improve representation without identifying the true
factors. Compare concentration and spectral recovery separately from fit;
the full-image reference is subject to the same iteration cap. Failed fits
remain visible. Noise scaling, weighting, residual-guided selection, and
superpixels are separate future work.

### Iteration cap and repeated sampling

The reference folder retains the original 500-cap run in `iterations_500/`;
the primary report uses a 1000-update cap. Run the robustness study directly:

```bash
python -m evaluations.evaluate_fitting_pool_robustness
```

The script first generates a fresh expansion in
`results/fitting_pool_robustness/`, then writes its robustness report under
`repeated_sampling/` within that directory. Use `--output-dir PATH` to choose
another root. To reuse an existing expansion, use
`--reuse-prerequisites --output-dir PATH`; `--base-dir` remains an alias for
`--output-dir`. Ten seeds (2026–2035)
at the +512 budget measure sensitivity to which additional pixels are used.
The first three successful fits are reused from `fitted_factors.npz`; seven
additional seeds are fitted per blur condition. The hull, blind starts, data,
and solver settings remain fixed. Sampling is without replacement, not a
bootstrap. It measures conditional sampling sensitivity, not uncertainty in
the whole pipeline or acquisition noise.

`repeated_sampling/` contains per-run metrics and stopping diagnostics,
per-channel spectral bands, concentration SD maps (sample SD, ddof=1), and
aligned factors in NPZ for later inspection. Ground truth is used for alignment
and accuracy evaluation only. Failed seeds remain in the run table and are
excluded from bands. Stable fits can still be inaccurate; the seed range is
not a confidence interval, and capped runs may remain unconverged.

When the archived 500-cap run is available, the script also produces
`iteration_comparison.csv`, `.md`, and `.png`, checking that scene settings,
starts, and exact fitting indices agree before comparing caps. Outputs from
both caps restart from the same starts; they are not continuation runs.

### Spectral drift without ground truth

Reusable factor alignment and spectral comparison are provided by
`lumic.evaluation.align_factors` and `compare_spectra`. See the
[usage guide](../lumic/evaluation/README.md) for real-data and simulation
examples, metric scales, and interpretation limits. The trajectory script
uses these functions; shared recovery metrics and concentration maps also
use the same factor alignment.

```bash
python -m evaluations.evaluate_spectral_pool_stability
```

This pilot keeps one nested sampling sequence (seed 2026), adding
64/256/512/1024/2048 pixels to the hull, with a full-image endpoint and a
500-update cap. Analysis receives only observed spectra and wavelengths.
Components are matched to fixed blind starts and reported as anonymous IDs.
Plots and CSVs track unit-L2 spectral shape changes and the largest change
in a channel of the unit-sum spectra. The full-image endpoint is a reference,
not truth. Results go to `data_driven_stability/` beneath the expansion folder.

In the saved pilot, component 4 changes much more than the other three in
both blur conditions. This flags dependence on pixel coverage, without
establishing which recovered spectrum is correct. Most runs reach the cap,
so incomplete optimization remains a possible contributor. Additional
sampling sequences and convergence checks are needed before interpreting
these trajectories as a general robustness result.

Related literature distinguishes sampling sensitivity, noise sensitivity,
and accuracy. Ahmad, Vitale, Cocchi and Ruckebusch's
[2023 weighted MCR-ALS study](https://doi.org/10.1002/cem.3478) compares
hull selection, full-data fitting and convex-peeling weights, and evaluates
spectral dispersion using noise replicates and residual bootstrap analysis.
Our nested-pool trajectory changes pixel coverage rather than resampling
noise, and does not implement their weighting method. Their results also
show that precise recovered spectra can be biased. Vitale and Ruckebusch's
[black-hole-effect study](https://doi.org/10.1002/cem.3442) explains why
adding observations can dilute the influence of essential spectra.

## Spatially supported residual-pool additions

From `Python/`, run:

```bash
python -m evaluations.evaluate_residual_pooling
```

This is a bounded four-component pilot on the existing noiseless spoke scenes
at zero and 0.65 µm blur. It makes one addition/refit round, with a requested
256 added pixels, three fixed seeds and a 500-update cap. It compares:

| Strategy | Candidate pixels |
| --- | --- |
| Hull only | Original convex-hull pool; no additions |
| Uniform | All eligible non-hull pixels |
| High residual | Pixels above the hull-only residual-RMS 80th percentile |
| Spatial residual | High-residual mask after opening with a radius-one disk |

Scikit-image performs [opening](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening)
on the Boolean mask, with false padding outside the field of view. This removes
isolated detections and imposes local support; it can also erase true small or
thin objects. Connected-region labels use four-neighbor connectivity. Sampling
is uniform over surviving pixels, not balanced over regions or spectral types.
All hull points remain; opening is applied before excluding hull points from
the addition candidates. There is no second convex hull or spatial penalty
in ALS in this first experiment.

Available strategies use the same actual addition budget, reduced if a
nonempty shortlist is too small. Empty lists are reported as unavailable,
not replaced by another method. If the baseline global relative residual is
at/below 1e-10, guidance is skipped rather than ranking tiny arithmetic
differences. This is an explicit diagnostic convention, not a noise threshold.
Threshold masks, exact indices, starts and solver settings are recorded.

`analyze_image(image, wavelengths, output_dir)` accepts a channel-last image
and uses no true spectra or counts. Metrics report raw RMS, relative fit,
adjacent signed-residual correlation, RMS/brightness association, anonymous
spectral drift and stopping diagnostics. Spatial correlation is descriptive;
it is not a randomness test or a certificate of correct factor recovery.
The same image informs selection and reporting, so these are not independent
validation scores. Constant channel offsets are removed by correlation
centering and should still be inspected through residual magnitude.

The entry point generates the scene, runs observed-data analysis, then makes
a separate simulation truth audit using the shared `evaluate_factors` API.
The audit cannot be performed on ordinary real images without references.
Truth snapshots/config are kept separate from analysis settings and factors.

Outputs go to `results/residual_pooling/`; choose `--output-dir PATH` to change
it. Results include selection masks, sampled points, linear/log residual maps, spectral
seed ranges (not confidence intervals), CSV metrics and NPZ factors. Plot
scales are shared across strategies and preview maps use the first seed.

In this pilot, unblurred hull data already fit below the floor despite wrong
factors in the truth audit. In the blurred scene, high-residual additions
improve both fit and recovery relative to uniform additions, while opening
adds very little. Raw residuals are far below one expected count and all
blurred fits reach the cap. This does not establish usefulness under noise or
at convergence, and does not justify ranking real-image factor accuracy from
residuals alone. Stop at this comparison before further method/parameter sweeps.
Next, assess shot/read-noise scaling with explicit count units or detector
calibration; raw spatial heterogeneity can follow brightness-dependent noise,
PSF/resampling correlations, or model mismatch. Remaining selection ideas
(region balancing, a hull within the shortlist, relevance weighting) are later
comparisons, not requirements for this PR.

### Real-image inspection using public COS-7 data

```bash
python -m evaluations.evaluate_real_residual_pooling
```

The script downloads the high-SNR mixed COS-7 CZI used by the minimal
notebook from [BioImage Archive S-BIAD4448](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4448).
The first run downloads about 145 MB. Subsequent runs reuse the cached file.
It preserves the notebook's median-intensity 20%
filter, four-component blind selection (`unique_profiles` parameter 0.5),
L2-normalized ALS, 500-update cap and 0.01% tolerance. A fixed seed makes the
selection repeatable; an earlier unseeded notebook run can produce a different
hull. It compares the original hull with that same hull plus up to 2,048
random pixels from the residual-RMS top 20% mask after radius-one opening.
Low-intensity filtering restricts selection; reporting includes every pixel.
Both fits restart identical blind spectra. There is one addition/refit round.

Use `--input PATH` to supply your own CZI or `--output-dir PATH` to choose the
results folder. By default, outputs go in the ignored
`Python/evaluations/results/real_residual_pooling/` folder. Each follow-up below
can run independently: it prepares the baseline and any intermediate stages
inside its own output directory. All accept `--input PATH` for a local CZI and
`--output-dir PATH` for a custom results tree (`--results-dir` remains an alias).
With no input argument they download/reuse the public CZI cache. To inspect an
existing completed results tree instead, pass `--reuse-prerequisites` and point
`--output-dir` at it. Keep that tree's input CZI available. Full-image projection
uses chunks and the shared reconstruction metrics to limit memory.

`comparison.png` displays RGB, threshold/opened masks, sampled pixels, shared
RMS scales and signed RMS reduction. Display limits use shared 99.5th
percentiles to expose structure; `residual_maps_full_range.png` provides
unclipped RMS and relative maps. `spectra_and_convergence.png` shows anonymous
spectra and pool-specific histories, which are not comparable full-image
scores. NPZ files retain full-resolution maps, abundances, spectra, histories
and indices; JSON and Markdown record settings, stopping and observed metrics.

Real detector intensities are not assumed to be photon counts. Raw residual
selection can favor brightness-dependent noise. A lower or flatter residual
on this same image is exploratory evidence about fit, not independent proof
of more accurate spectra or concentrations. Relative errors at faint pixels
also need care. No ground-truth audit is performed for the real image.

To generate fits and inspect their spectral residual structure in one command:

```bash
python -m evaluations.inspect_real_residual_structure
```

This saves six centred residual PCA directions paired with signed spatial
score maps, mean signed residual spectra and consecutive-channel correlation
maps under `results/real_residual_structure/residual_structure/`. Use
`--reuse-prerequisites --output-dir results/real_residual_pooling` to inspect
an existing completed baseline without rerunning ALS. Full-resolution
scores/loadings/covariance and correlation maps remain local in NPZ files.
Each reconstructed residual chunk is checked against its saved RMS map.
The paired PCA comparison uses the hull-only basis and mean for both fits,
with shared symmetric display scales. Independent PCA decompositions are also
shown, but their directions can rotate/change order between fits.

This resembles uniqueness-map visualization, but uses centred **signed
residuals**, not normalized original spectra or the hull's first-coordinate
ratios. PCA maps show unexplained pattern amplitudes, not purity or component
concentration. The mean is displayed separately because centred PCA removes
it. Regional means use fixed hull-derived masks, not biological segmentations.
Raw detector noise is not whitened. Fitting itself induces spectral residual
correlations, so autocorrelation is descriptive rather than a noise-null test.
The six displayed PCs are not six asserted missing components; detector
effects and errors in existing signatures can also produce coherent patterns.

To explore an additional component, including automatic baseline/diagnostic preparation:

```bash
python -m evaluations.evaluate_real_extra_component
```

This expands the **original hull**, using its original residual PC1/PC2 maps,
not the previous RMS-based expanded pool. Positive and negative high-score
regions each supply up to 512 opened-mask samples per PC. Two seeds provide a
bounded sampling sensitivity check. Original measured spectra enter ALS; signed
residual spectra only guide selection. Four and five components are fitted to
exactly the same pool within each seed, with the notebook's 500-update cap and
stopping settings. Four starts remain the original blind starts; five starts
are selected by a five-dimensional hull and `unique_profiles` within the new
pool, so the comparison includes initialization as well as component count.

Local `results/real_extra_component/extra_component/` contains residual maps on
the fixed original-hull PCA basis/centre, matched spectra, candidate extra
contribution maps, all five contribution maps, metrics and exact selections.
Rectangular spectral matching finds an unmatched fifth factor, but does not
establish that it is a pure extra fluorophore; splitting known signatures and
detector/model effects are alternative explanations. Two seeds and same-image
fit improvements are exploratory, without noise significance or independent
validation. No private real-image outputs are staged by these scripts.

To run the complete chain through a second addition round and six components:

```bash
python -m evaluations.continue_real_component_recovery
```

The fresh five-component residual PC1 supplies both positive and negative
high-score regions after opening, with up to 512 new samples per sign. Every
original hull and first-round pixel remains. Five/six fits share this new
pool; five starts use the preceding fitted spectra, while six starts use a
six-dimensional hull and `unique_profiles` within the new pool. The cap remains
500, with notebook normalization/tolerance. Two sampling seeds are retained.

Prerequisites are generated inside `results/real_component_recovery/`.
Its `component_recovery/` directory organizes results into `components_4`,
`components_5`, and `components_6`, further labelled by hull-only/round and seed.
Every fit has `fit.npz` and a fresh `residual_pca.png`. Prior results are copied
as snapshots so earlier scripts retain their inputs. Six-component folders
also show all spectral curves/contribution maps and a fixed pre-round residual
PC1 comparison. The root README records metrics and folder meanings.

The user's far-red actin-label interpretation is recorded as sample context.
Neither a coherent contribution nor a flatter residual uniquely establishes
identity. Raw noise is not whitened; fresh PCA directions change between fits,
and centred PCA removes mean bias. More components can capture detector/model
effects or split known signatures. This pilot does not automatically increase
rank until residuals appear flat.

## Residual-guided workflow building blocks

Selection and hull construction are explicitly separate:

```python
from lumic.utils import select_analysis_pixels
from evaluations._residual_workflow import convex_hull_pool

retained = select_analysis_pixels(mixData, median_quantile=0.20)
pool0 = convex_hull_pool(mixData, retained, npc=4, seed=2026)
```

`select_analysis_pixels` is public in `lumic.utils` and also available as
`lumic.select_analysis_pixels`. It returns `AnalysisPixelSelection` with original
indices, spatial mask, threshold/settings and disjoint exclusion reasons.
Default spectral axis is zero; use `spectral_axis=-1` for simulation images.
`median_quantile=None` disables optional intensity filtering; zero retains all
otherwise usable pixels. Nonfinite, negative and all-zero spectra are excluded
from candidates. Input data is never modified or repaired, so full-image
projection still requires valid inputs. Threshold ties remain; selecting a
percentile does not guarantee that exactly that fraction is excluded.

The legacy `remove_low_intensity` shares the same median-threshold calculation
and retains its copied-row, median-only behavior. It now emits a visible
`FutureWarning` at the caller suggesting the traceable function. The interfaces
are not identical for invalid/zero-signal data; no removal date is set.

`convex_hull_pool`, `fit_pool`, `inspect_fit_residuals` and `expand_pool` remain experimental
in `_residual_workflow.py`. Selected PCs use human-facing one-based numbers;
component count is independent. Both signs are sampled after opening, with
previous pixels retained and duplicate/stale-pool checks. Same-pool controls
are optional and do not influence the main fit or the next expansion.

Residual PCA calculation is now public in `lumic.evaluation`: `residual_pca`
accepts observed/reconstructed arrays, `residual_pca_from_residuals` accepts a
signed residual array, and `residual_pca_from_factors` streams factor products
for large images. Their common `ResidualPCAResult` contains mean bias,
loadings, variances and spatial scores. See the
[API guide](../lumic/evaluation/README.md#signed-residual-pca-spectral-patterns-and-spatial-maps).
The prototype `inspect_fit_residuals` only adapts a prototype fit and plots/saves
the public result. Evaluation scripts also use the shared public covariance
calculation; their regional mean/correlation analysis and plotting remain
prototype-specific. No API helper reads files or decides which PCs/rank to use.

Residual-PC sampling is also public: `lumic.evaluation.select_residual_pixels`
returns a `ResidualPixelSelection`. Call it with selected PCs, retained eligibility and existing pool indices. The
prototype `expand_pool` checks provenance/eligibility, combines indices and
saves previews. Existing evaluation tuple adapters delegate to the same public
selector; the RMS pilot shares its threshold/opening primitive. See the
[sampling guide](../lumic/evaluation/README.md#select-pixels-from-residual-pc-regions)
for an example, field meanings, quantile semantics and interpretation limits.

## Dim far-red line beside bright spokes: known-truth pilot

From `Python/`, run:

```bash
python -m evaluations.evaluate_dim_line_recovery
```

This first noiseless phase uses five true contributions: bright EBFP2, EGFP
and mCherry spokes; one dim mNeptune line away from the spokes; and a broad
500 nm background throughout the field. The 64x64 image uses 0.12 µm spacing,
a 0.12 µm line, 900 expected detected counts per bright spoke, 90 for the dim
line and 24 for background. Detector-sampled signatures are normalized to unit
sum so the input amplitudes are detected counts, despite incomplete red-tail
coverage by `980_32ch`. Overlapping spokes add their contributions.

Default blur FWHMs are 0, 0.24 and 0.60 µm. Geometry, amplitudes and seed stay
fixed; no noise is added. The no-blur line is aligned to a scan row so scan
sampling does not accidentally erase it. Broadening reduces both its peak
intensity and its spectral purity relative to background. Candidate intensity
quantile is zero to avoid separately hiding the line by low-intensity exclusion.

Blind analysis receives only the image/wavelengths. Start with four components,
retain all hull pixels, select both signs of residual PC1 with opening, and add
up to 128 distinct pixels per sign. Four-component original/expanded fits share
the same starts. Five-component expanded/original-pool controls share starts
inferred from the expanded observed pool: that original-pool control therefore
shares initialization information and is not an independent blind pipeline.
The cap is 500 with 0.01% tolerance; capped and failed runs remain explicit.

Truth is used afterward to audit spectra/contributions. Rectangular matching
documents the best one-to-one assignment of four fitted spectra to five truth
rows, with an unmatched row represented by zero contribution in audit figures.
The separate `initial_start_truth_audit.json` checks selected starts before ALS,
so missing-at-selection and missing-after-refinement cannot be conflated.

Outputs stay in ignored `results/dim_line_recovery/`: configuration/versions,
summary table, true-count/RGB scene, full-image residual comparison, shared-scale
mNeptune recovery/errors, fresh residual PCA per fit, factors and exact selection
provenance. Edit top-level scene/settings or choose `--output-dir PATH`.
For an explicitly labelled exploratory blur extension, preserve the first run:

```bash
python -m evaluations.evaluate_dim_line_recovery --blur-fwhm 1.20 \
    --output-dir evaluations/results/dim_line_recovery/extended_blur
```

The first three-condition run does not yet omit mNeptune from the starting
selection: background is unmatched there. At 0.60 µm, mNeptune is unmatched
after four-component ALS, and five components recover a localized line while
still blending its spectrum with background. Reconstruction becomes nearly
exact despite inaccurate pure factors. This is a limitation to preserve, not
a successful pure-signature-recovery claim or reason to tune starts using truth.
The separately recorded 1.20 µm exploratory extension leaves mNeptune unmatched
both in the selected four starts and after four-component ALS, meeting the
intended omission condition. The five-component fit still blends its spectrum
with background, and both five-component controls reach the 500-update cap;
neither pure recovery nor convergence is established by their small residuals.
Noise, repeated sampling and the excess sixth-component experiment follow only
after inspecting this bounded noiseless phase.


Residual correlation calculations are shared public APIs:
`lumic.evaluation.residual_neighbor_correlation` measures spatial persistence
of signed residual spectra, and `residual_spectral_correlation` maps
consecutive-channel correlation within pixels. The pooling pilot and real-data
inspection helpers delegate to these functions. Dim-line evaluations also save
`residual_correlations.npz` per fit and report spatial correlation in the summary;
they suppress correlations below the explicit 1e-10 relative residual floor.
Existing saved outputs predate this reporting addition until regenerated.
See the evaluation API README for definitions, examples and interpretation limits.


## Classical radial spokes with shot noise

From `Python/`:

```bash
python -m evaluations.evaluate_radial_spoke_noise
```

This separate evaluation reuses the original EBFP2/EGFP/mCherry spoke geometry
and broad 500 nm background. It adds independent Poisson measurements AFTER
blur and detector-space mixing. Background is an actual fourth contribution;
shot noise is a fluctuation of the total measured spectrum, not a fifth species.
There is no read-noise term or real-instrument gain/offset model in this phase.

Human inputs are grouped at the top of `evaluate_radial_spoke_noise.py`:

- Blur FWHMs: 0 and 0.65 µm; 64x64 pixels at 0.12 µm sampling.
- Count scales: 1, 0.25 and 0.0625, preserving all foreground/background ratios.
  Per-spoke amplitudes are 900, 225 and 56.25 expected detected counts; background
  is 24, 6 and 1.5 respectively. Overlapping objects add their contributions.
- Three independent noise seeds (2026–2028) and a separate fixed selection seed.
- Four components supplied by the researcher; all hull pixels are fitted, with
  no residual pool expansion and no median-intensity cutoff. Zero pixels are
  excluded from candidate selection, but the full measured image is projected.
- 500 ALS updates, 0.01% tolerance; rerun the first capped condition at 2000
  with identical data/starts. Earlier stopping is a fit-improvement criterion,
  not a guarantee of accurate spectra or global optimization.

There are 18 noisy cases and six matching noiseless controls, plus one longer
control if a cap is reached. Each noisy realization gets its own hull/starts;
thus the report measures the complete blind pipeline, including initialization
sensitivity. The older blur script uses a random fitting subset; compare noise
against the hull-only controls in THIS report instead of comparing between
scripts as though fitting-pool definitions were identical.

`results/radial_spoke_noise/` holds settings, per-case factors/indices/history,
residual PCA, spectral/contribution truth audits, CSV/JSON summaries and an
overview figure. Representative controls and the first noise seed get recovery
figures with true/starting/fitted spectra, true/recovered counts and error maps.
Use `--output-dir PATH` for a separate run. The report records failures explicitly,
plots successful-seed means/min–max ranges (not confidence intervals), and
leaves gaps where no seed succeeds. Partial availability is annotated.

Measured-image residuals, spatial/spectral correlations and signed means are
observable without truth. Clean-image prediction error, matched spectral errors,
contribution errors and realized oracle-noise RMS require simulation truth.
The ideal global theoretical noise RMS is sqrt(mean expected counts). Fitted
residuals use the same data as fitting and need not equal independent noise;
a ratio below one is not proof of correct recovery or a calibrated test.

At low counts, some runs fail the normalized hull's first-coordinate check;
others stop with poor factors. Preserve these outcomes rather than silently
filtering/replacing inputs to force success. Such a selection failure can reflect
implementation/conditioning limitations and is not a physical recovery bound.
Automatic count choices, noise-based selection thresholds and microscope
acquisition commands remain deferred; these figures support human decisions.


### Lowest-count spatial binning comparison

```bash
python -m evaluations.evaluate_radial_spoke_binning
```

This follow-up in the same PR compares 1×1, 2×2 and 4×4 blocks at count scale
0.0625, for both blur settings and all three noise seeds (18 attempts).
Sum the SAME observed pixels: never regenerate Poisson noise after binning.
2×2 and 4×4 pool 4 and 16 pixels and change sampling from 0.12 µm to 0.24 and
0.48 µm. Field of view and per-channel total counts are preserved. The new
`sum_bin` helper remains evaluation code; no new public API is introduced.

Learn spectra on each binned image using the same four-component blind
hull/ALS workflow, then hold spectra fixed and project ORIGINAL noisy pixels.
Audit coarse fits against equally summed expected counts separately from
original-resolution concentration errors; sums have different absolute units
per pixel and should not be compared as though their pixel sizes were equal.
`binning/discovery/truth_audit.*` lives under each case folder, while
`original_projection/truth_audit.*` describes the original-grid projection.

Outputs are in `results/radial_spoke_noise/binning/`: per-seed tables,
separate coarse/original audits, and two comparison figures showing matched
spectra and original-grid concentration errors with shared display limits.
The six unbinned cases reproduce the original sweep's outcomes/metrics.

All binned cases complete in this fixture; 4×4 improves spectral recovery more
than 2×2, but ambiguity and background mixing remain. All comparisons retain
failures, seed spread and stopping outcomes. Binning can make spectral discovery
more robust while mixing neighboring species; it does not add photons to the
original pixels or guarantee accurate signatures/concentrations.
