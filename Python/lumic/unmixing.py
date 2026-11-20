"""Nonnegative spectral unmixing and MCR-ALS refinement."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from ._nnls import MAX_BATCH_COMPONENTS as _MAX_BATCH_COMPONENTS
from ._nnls import prepare_small_batch as _prepare_small_batch
from ._nnls import solve_prepared_batch as _solve_prepared_batch


def _solve_batch_with_bvls(components, mixed_batch, first_observation):
    """Retry a failed NNLS batch one observation at a time with bounded least squares."""
    abundance = np.full(
        (mixed_batch.shape[0], components.shape[0]), np.nan, dtype=float
    )
    unresolved = []
    max_iter = max(1000, 10 * components.shape[0])

    for local_idx, spectrum in enumerate(mixed_batch):
        observation_idx = first_observation + local_idx
        try:
            result = lsq_linear(
                components.T,
                spectrum,
                bounds=(0, np.inf),
                method="bvls",
                max_iter=max_iter,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            unresolved.append(observation_idx)
            continue

        if result.success and np.all(np.isfinite(result.x)):
            abundance[local_idx] = result.x
        else:
            unresolved.append(observation_idx)

    return abundance, unresolved


def positive_least_squares(mixed_data, components, do_norm = False, batch_size = 1000, verbose=False, *, solver="nnls"):
    """Least square implementation of unmixing with non-negative constrain and L1 or L2 norm

    This funtion measures the abundace of components (single spect) in the query mixed data (spectral mix)
    mixed_data is requested first to make pooling easier later on.

    Parameters
    ----------
    mixed_data (numpy 2D array): Query data of mixed components. rows are observations, cols are spectral channels.
                This dataset is "flattened" meaning that each row belongs to a pixel in the image.
    components (numpy 2D array): Array containing the pure components to be used for unmixing of mixed_data and
                thus calculate the abundance. rows are component indexes and cols are spectral channels
    do_norm (bool False, or string "L1" "L2"): normalization strategy, bool False means no normalization
    batch_size (integer): number of oberveations to be unmixed "at the same time". For large datasets we cannot send all the
                data at once to sklearn linear regresion.
    verbose (bool): if true then we use tqdm to print out a progress bar.
    solver (str): "nnls" retains the original per-observation solver.
                "auto" uses batch solving for up to six well-conditioned
                components, and the original NNLS/BVLS path otherwise.
                Fixed-component decompositions are reused across batches
                within this call. Later calls prepare their own matrices;
                changing spectra in ALS cannot reuse a stale preparation.
                Single-batch calls stream subsets to limit memory use.

    If batched NNLS does not converge, the observations in that batch are retried
    individually with the BVLS solver. Observations that still do not converge
    are returned as NaN and identified by a RuntimeWarning with their flattened
    observation indices.

    Returns
    -------
    abundance (numpy 2D array): abuundance of each componen for each observation in the mixed_data. The diffing
                is done with non-negative constrains. rows are observations (same as mixed data), cols are
                components (same as rows in components)

    """
    if solver not in ("nnls", "auto"):
        raise ValueError("solver must be 'nnls' or 'auto'")
    # basic assertion
    assert mixed_data.shape[1] == components.shape[1], "number of spectral channels must be the same on both mixed_data and components"
    nObservations = mixed_data.shape[0]

    # TODO: this seems not to work, prone to bugs!!!!
    # normalization, we make here a switch case, and raise an error if normalization input is not understood
    if do_norm == "L1":
        normVal = np.abs(components).sum(axis=1)
    elif do_norm == "L2":
        normVal = np.sqrt(np.square(components).sum(axis=1))
    else:
        assert not(do_norm), "unexpected input"
        normVal = np.ones(components.shape[0])
    normVal = np.tile(normVal, (components.shape[1],1)).T
    components = components / normVal

    # linear regresion with non negative constrain
    reg_nnls = LinearRegression(
                                    positive=True,
                                    fit_intercept=False,
                                )
    # prepare an abundance array to fill during the fitting
    abundance = np.ndarray((mixed_data.shape[0],components.shape[0]), dtype=float)
    fallback_observations = 0
    fallback_batch_starts = []
    unresolved_observations = []

    batch_starts = range(0, nObservations, batch_size)
    prepared = None
    if solver == "auto" and nObservations:
        # Components stay fixed within this fitting call. Prepare their subset
        # decompositions once for all pixel batches. The next ALS fitting call
        # creates a new preparation, so updated factors never use stale matrices.
        try:
            prepared = _prepare_small_batch(
                components, cache_subsets=len(batch_starts) > 1,
            )
        except np.linalg.LinAlgError:
            pass  # Retain NNLS/BVLS when preparation is numerically unusable.
    if verbose:
        batch_starts = tqdm(batch_starts)

    for xi in batch_starts:
        xf = min(xi + batch_size, nObservations)
        if prepared is not None:
            try:
                batch_solution = _solve_prepared_batch(mixed_data[xi:xf], prepared)
            except np.linalg.LinAlgError:
                batch_solution = None
            if batch_solution is not None:
                abundance[xi:xf] = batch_solution
                continue
        try:
            abundance[xi:xf,:] = reg_nnls.fit(
                components.T, mixed_data[xi:xf,:].T
            ).coef_
        except RuntimeError:
            fallback_abundance, unresolved = _solve_batch_with_bvls(
                components, mixed_data[xi:xf,:], xi
            )
            abundance[xi:xf,:] = fallback_abundance
            fallback_observations += xf - xi
            fallback_batch_starts.append(xi)
            unresolved_observations.extend(unresolved)

    if fallback_observations:
        shown_batch_starts = fallback_batch_starts[:10]
        more_batches = len(fallback_batch_starts) - len(shown_batch_starts)
        additional_batches = f" and {more_batches} more" if more_batches else ""
        message = (
            f"NNLS did not converge for batch(es) starting at flattened "
            f"observation indices {shown_batch_starts}"
            f"{additional_batches}; "
            f"BVLS fallback was used for "
            f"{fallback_observations} observation(s)."
        )
        if unresolved_observations:
            message += (
                " BVLS also failed for flattened observation index/indices "
                f"{unresolved_observations}; their abundances are NaN."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return abundance


def normalize_components(components, do_norm=False):
    """Normalize component rows, keeping rows with no signal at zero."""
    if do_norm == "L1":
        normVal = np.abs(components).sum(axis=1)
    elif do_norm == "L2":
        normVal = np.sqrt(np.square(components).sum(axis=1))
    else:
        assert not(do_norm), "unexpected input"
        normVal = np.ones(components.shape[0])
    normVal = normVal[:, np.newaxis]
    # A component can disappear during MCR-ALS. Its zero spectrum has no norm
    # to divide by, so keep that row at zero instead of producing NaNs.
    components = np.divide(
        components,
        normVal,
        out=np.zeros_like(components, dtype=float),
        where=normVal != 0,
    )

    return components


@dataclass(frozen=True)
class MCRALSDiagnostics:
    """Why MCR-ALS stopped and where its best paired factors were found."""

    stop_reason: str
    iterations_completed: int
    best_iteration: int
    best_relative_l2: float


def mcr_als(mixData, in_estimates, nIter, normL, *, tol_percent=None,
            return_diagnostics=False, iteration_callback=None, solver="auto"):
    """Fit nonnegative concentration and spectral factors by alternating least squares.

    Parameters
    ----------
    mixData : numpy.ndarray
        Spectral data with pixels in rows and channels in columns.
    in_estimates : numpy.ndarray
        Initial spectra with components in rows and channels in columns.
        Initial concentration estimates are not yet supported.
    nIter : int
        Maximum number of alternating spectral and concentration updates.
    normL : bool or str
        False leaves spectra unscaled; "L1" or "L2" gives each spectrum unit norm.
        Concentrations are fitted against the spectra on that same scale.
    tol_percent : float, optional
        Stop when a complete update improves relative lack of fit by no more
        than this percentage. None runs all nIter updates. A fit
        at numerical precision also stops. A worsening step never counts as
        convergence.
    return_diagnostics : bool, optional
        If True, return an additional MCRALSDiagnostics value.
    iteration_callback : callable, optional
        If provided, called with the number of completed updates after each
        iteration. This can be used to report progress without changing the
        fitting results.
    solver : str, optional
        "auto" batches small, well-conditioned nonnegative least-squares
        problems, falling back to the original solver when needed. "nnls"
        uses the original per-observation solver for comparison. Both solve
        the same constrained least-squares objective. If "auto" falls back
        because the component count exceeds the batch limit, print an
        informational message once at the start of the run.

    Returns
    -------

    out_C : numpy.ndarray
        Fitted concentrations, with pixels in rows and components in columns.
    out_S : numpy.ndarray
        Fitted spectra on the scale selected by normL.
    R_vector : numpy.ndarray
        Fractional lack of fit before the first update and after each completed
        update. The returned factors are from the best iteration, which may
        precede the last history entry.
    diagnostics : MCRALSDiagnostics, optional
        Stopping reason, updates completed, best iteration, and best fit.
    """
    if (
        isinstance(nIter, bool)
        or not isinstance(nIter, (int, np.integer))
        or nIter < 0
    ):
        raise ValueError("nIter must be a non-negative integer")
    if tol_percent is not None and (
        isinstance(tol_percent, (bool, np.bool_))
        or not isinstance(tol_percent, (int, float, np.integer, np.floating))
        or not np.isfinite(tol_percent)
        or tol_percent <= 0
    ):
        raise ValueError("tol_percent must be a positive finite number or None")
    if iteration_callback is not None and not callable(iteration_callback):
        raise TypeError("iteration_callback must be callable or None")
    if mixData.ndim != 2 or in_estimates.ndim != 2:
        raise ValueError("mixData and in_estimates must be 2D arrays")
    # A matching channel count identifies the supported spectral start. Row
    # count alone can also match by coincidence, so check channels first.
    if in_estimates.shape[1] != mixData.shape[1]:
        if in_estimates.shape[0] == mixData.shape[0]:
            raise ValueError("Initial concentration estimates are not supported")
        raise ValueError("Initial spectra must have the same number of channels as mixData")
    # Lack of fit divides by the input signal magnitude, which is undefined
    # when every measured value is zero.
    if not np.any(mixData):
        raise ValueError("mixData must contain a nonzero signal")
    # A zero starting spectrum provides no usable initial direction. Zero
    # rows produced later by fitting are valid and remain zero instead.
    if np.any(np.all(in_estimates == 0, axis=1)):
        raise ValueError("Initial spectra must be nonzero")

    if solver == "auto" and in_estimates.shape[0] > _MAX_BATCH_COMPONENTS:
        # Report this choice once per ALS run, not once per batch or iteration.
        tqdm.write(
            f"MCR-ALS: using the original NNLS/BVLS solver for "
            f"{in_estimates.shape[0]} components. The batch solver currently "
            f"supports up to {_MAX_BATCH_COMPONENTS} components because the "
            f"number of subsets grows exponentially (2**n). This cutoff is "
            f"a conservative performance choice, not a mathematical limit; "
            f"it can be revisited with benchmarks."
        )

    # Keep the spectra used for fitting and reconstruction on the same scale.
    # Fitting against a normalized copy and returning the original spectra
    # would give concentrations in the wrong units for out_C @ out_S.
    out_S = normalize_components(in_estimates, do_norm=normL)
    out_C = positive_least_squares(
        mixed_data=mixData, components=out_S, do_norm=False, batch_size=1000,
        solver=solver,
    )
    iPred = np.dot(out_C, out_S)

    R_vector = np.empty(nIter+1)
    R_vector[0] = lof(mixData, iPred, method="sqr")
    best_fit = R_vector[0]
    best_iteration = 0
    best_C = out_C.copy()
    best_S = out_S.copy()
    stop_reason = "max_iterations"
    completed = 0
    precision_floor = 100 * np.finfo(float).eps

    for idx in range(nIter):
        # Estimate each spectral channel from the current concentrations.
        out_S = positive_least_squares(
            mixed_data=mixData.T, components=out_C.T, do_norm=False,
            batch_size=1000, solver=solver,
        ).T
        # Fix the arbitrary scale of each recovered spectrum. Refit C after
        # this change so its coefficients match the normalized spectra.
        out_S = normalize_components(out_S, do_norm=normL)
        out_C = positive_least_squares(
            mixed_data=mixData, components=out_S, do_norm=False, batch_size=1000,
            solver=solver,
        )
        # Record the fit of the paired factors after this complete update.
        prediction = np.dot(out_C, out_S)
        R_vector[idx+1] = lof(mixData, prediction, method="sqr")

        completed = idx + 1
        current_fit = R_vector[completed]
        previous_fit = R_vector[idx]
        if current_fit < best_fit or (
            current_fit <= precision_floor and best_fit <= precision_floor
        ):
            # When both fits are at floating point precision, prefer the
            # completed update. It can remove an unused component's spectrum
            # while changing the residual only by round-off.
            best_fit = current_fit
            best_iteration = completed
            best_C = out_C.copy()
            best_S = out_S.copy()

        # A fit indistinguishable from zero is already as good as floating
        # point arithmetic permits. Otherwise, compare two complete updates;
        # only a small nonnegative improvement signals a plateau.
        near_zero = current_fit <= precision_floor and (
            current_fit <= previous_fit or previous_fit <= precision_floor
        )
        small_improvement = (
            previous_fit > 0
            and 0 <= current_fit <= previous_fit
            and 100 * (previous_fit - current_fit) / previous_fit <= tol_percent
        ) if tol_percent is not None else False
        if iteration_callback is not None:
            iteration_callback(completed)
        if tol_percent is not None and (near_zero or small_improvement):
            stop_reason = "converged"
            break

    diagnostics = MCRALSDiagnostics(
        stop_reason=stop_reason,
        iterations_completed=completed,
        best_iteration=best_iteration,
        best_relative_l2=float(best_fit),
    )
    result = (best_C, best_S, R_vector[:completed + 1])
    return (*result, diagnostics) if return_diagnostics else result


def lof(y,y_pred, method="abs"):
    if method == "abs":
        sE = np.sum(np.abs(y-y_pred))
        sD = np.sum(np.abs(y))
        out = sE/sD
    elif method == "sqr":
        sE2 = np.sum(np.square(y-y_pred))
        sD2 = np.sum(np.square(y))
        out = np.sqrt(sE2/sD2)
    else:
        raise("problems, dont know this method")
    return out
