"""Spectral distinctness maps and selection of candidate component spectra.

The hull selector and pixel map share the normalized SVD projection below.
The unique-profile selector uses its own method to choose distinct profiles
from a candidate pool. These related tools live together for maintenance.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import ConvexHull


def _project_spectral_pixels(pixels, npc, *, random_state=None, components=None):
    """Share the hull's SVD projection and first-component normalization."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if components is None:
            model = TruncatedSVD(n_components=npc, random_state=random_state)
            projected = model.fit_transform(pixels)
            components = model.components_
        else:
            projected = pixels @ components.T
        # The hull helper normalizes U*S by its first column and drops that
        # constant coordinate. Small denominators cannot give reliable ratios.
        leading = projected[:, 0]
        tolerance = 100 * np.finfo(float).eps * np.linalg.norm(projected, axis=1)
        valid = np.isfinite(projected).all(axis=1) & (np.abs(leading) > tolerance)
        coordinates = np.full((len(pixels), projected.shape[1] - 1), np.nan)
        coordinates[valid] = projected[valid, 1:] / leading[valid, None]
    valid &= np.isfinite(coordinates).all(axis=1)
    return coordinates, components, valid


@dataclass(frozen=True)
class SpectralUniquenessResult:
    """A distance map and its fitted spectral coordinate system.

    ``scores``, ``valid_mask``, and ``total_intensity`` have the image's spatial
    shape. ``coordinates`` adds an axis of length npc-1. Excluded pixels have
    NaN scores and coordinates. ``center_coordinates`` is the mean position
    of retained pixels, unless a reference result supplies the center.
    ``projection_components`` stores the SVD basis for later comparisons.
    """

    scores: np.ndarray
    valid_mask: np.ndarray
    total_intensity: np.ndarray
    coordinates: np.ndarray
    center_coordinates: np.ndarray
    projection_components: np.ndarray


def spectral_uniqueness_map(image, *, npc=3, spectral_axis=0,
                            min_total_intensity=0.0, reference=None,
                            random_state=0):
    """Score distance from the center of the convex-hull spectral space.

    The mapping is the same as ``essential_spectral_pixels``: fit an
    uncentered TruncatedSVD to pixel spectra, divide each projected row by
    its first coordinate, and retain coordinates 2 through npc. The score
    is Euclidean distance from the mean point in that space. Computing a
    hull or choosing component signatures is not required to score pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Nonempty 3D spectral image, default layout (channel, y, x). Use
        ``spectral_axis=-1`` for simulation images (y, x, channel).
    npc : int, optional
        Number of SVD components, as in the hull helper. The resulting space
        has npc-1 coordinates. Must be at least two and no larger than the
        channel count or retained pixel count when fitting a new projection.
        Ignored when a reference supplies an existing projection.
    spectral_axis : int, optional
        Channel axis. The other axes keep their order in the returned maps.
    min_total_intensity : float, optional
        Retain pixels whose summed signal is at least this nonnegative value.
        Zero-signal pixels are always excluded. Set a threshold appropriate
        to the acquisition so very faint pixels do not amplify noise.
    reference : SpectralUniquenessResult, optional
        Reuse both the SVD basis and center from a previous result. Images
        must use the same channel definitions and order. This keeps the
        coordinate system fixed across images or brightness changes.
    random_state : int, optional
        Seed for fitting TruncatedSVD; default zero gives repeatable mapping.

    Returns
    -------
    SpectralUniquenessResult
        Distances, mask, intensity, projected coordinates, center, and basis.

    Notes
    -----
    This is spectral distinctness relative to the retained population, not a
    purity probability. Uniform pure spectra have zero distance from their
    own center. Noise, background, and artifacts can also lie far from it.

    First-coordinate normalization cancels a pixel's multiplicative intensity
    for a fixed basis. Refitting SVD after changing brightness or the retained
    population can change that basis and therefore the distances. Reuse a
    reference for controlled comparisons. Scores are raw distances, without
    per-image display rescaling, and depend on npc and the detector channels.

    Negative/nonfinite spectra, nonfinite totals, insufficient signal, and
    near-zero first SVD coordinates are excluded rather than clipped. At
    least one usable projected pixel is required.
    """
    if np.iscomplexobj(image):
        raise ValueError("image must contain real intensities")
    data = np.asarray(image, dtype=float)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("image must be a nonempty 3D spectral image")
    if (isinstance(spectral_axis, (bool, np.bool_))
            or not isinstance(spectral_axis, (int, np.integer))
            or not -data.ndim <= spectral_axis < data.ndim):
        raise ValueError("spectral_axis must identify one of the image's axes")
    if (isinstance(min_total_intensity, (bool, np.bool_))
            or not np.isscalar(min_total_intensity)):
        raise ValueError("min_total_intensity must be finite and nonnegative")
    try:
        threshold = float(min_total_intensity)
    except (TypeError, ValueError) as error:
        raise ValueError("min_total_intensity must be finite and nonnegative") from error
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("min_total_intensity must be finite and nonnegative")

    spectra = np.moveaxis(data, spectral_axis, -1)
    spatial_shape = spectra.shape[:-1]
    pixels = spectra.reshape(-1, spectra.shape[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        totals = pixels.sum(axis=1)
    valid = (np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1)
             & np.isfinite(totals) & (totals > 0) & (totals >= threshold))
    if not valid.any():
        raise ValueError("no valid pixels remain after intensity and data filtering")

    components = None
    if reference is None:
        if (isinstance(npc, (bool, np.bool_))
                or not isinstance(npc, (int, np.integer))
                or not 2 <= npc <= min(pixels.shape[1], int(valid.sum()))):
            raise ValueError("npc must be between 2 and the retained pixel/channel count")
    else:
        if not isinstance(reference, SpectralUniquenessResult):
            raise ValueError("reference must be a SpectralUniquenessResult")
        components = np.asarray(reference.projection_components)
        if (components.ndim != 2 or components.shape[0] < 2
                or components.shape[1] != pixels.shape[1]
                or not np.isfinite(components).all()):
            raise ValueError("reference projection must match the image's channels")

    projected, components, usable = _project_spectral_pixels(
        pixels[valid], npc, random_state=random_state, components=components
    )
    # Preserve image coordinates when projection adds further exclusions.
    retained_indices = np.flatnonzero(valid)
    valid[retained_indices[~usable]] = False
    if not valid.any():
        raise ValueError("no pixels have a usable first SVD coordinate")
    if reference is None:
        center = projected[usable].mean(axis=0)
    else:
        center = np.asarray(reference.center_coordinates)
        if center.shape != (components.shape[0] - 1,) or not np.isfinite(center).all():
            raise ValueError("reference center must match the projection")

    coordinates = np.full((len(pixels), components.shape[0] - 1), np.nan)
    coordinates[valid] = projected[usable]
    scores = np.full(len(pixels), np.nan)
    scores[valid] = np.linalg.norm(coordinates[valid] - center, axis=1)
    return SpectralUniquenessResult(
        scores=scores.reshape(spatial_shape), valid_mask=valid.reshape(spatial_shape),
        total_intensity=totals.reshape(spatial_shape),
        coordinates=coordinates.reshape(*spatial_shape, components.shape[0] - 1),
        center_coordinates=center.copy(), projection_components=components.copy(),
    )



def spectral_uniqueness_rgb(result, *, max_distance=None):
    """Color spectral direction by hue and full-space distance by brightness and saturation.

    Use the first two retained coordinates relative to the spectral center
    for the angle, regardless of npc. With npc=2, negative and positive
    positions use cyan and red respectively. HSV brightness and saturation both increase with distance.
    Pixels at the center are black; excluded pixels are dark gray.

    max_distance is the positive distance displayed at full brightness and saturation.
    By default use the largest retained score. Reuse an explicit scale along
    with a shared projection/center when comparing images. Larger distances
    are clipped for display only; numerical scores are unchanged.

    In higher dimensions, directions outside the displayed plane cannot be
    distinguished by hue. When both displayed coordinates are effectively
    zero, show black because the displayed direction is undefined, even if
    the full-dimensional score is nonzero. Colors are projection directions,
    not fluorophore identities or physical emission colors.
    """
    try:
        from matplotlib.colors import hsv_to_rgb
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise ImportError(
            "Spectral uniqueness RGB rendering requires Matplotlib. "
            "Install the visualization extra with pip install 'lumic[visualization]'."
        ) from error

    if not isinstance(result, SpectralUniquenessResult):
        raise ValueError("result must be a SpectralUniquenessResult")
    if max_distance is None:
        maximum = float(np.max(result.scores[result.valid_mask]))
        maximum = maximum if maximum > 0 else 1.0
    else:
        if isinstance(max_distance, (bool, np.bool_)) or not np.isscalar(max_distance):
            raise ValueError("max_distance must be finite and positive")
        try:
            maximum = float(max_distance)
        except (TypeError, ValueError) as error:
            raise ValueError("max_distance must be finite and positive") from error
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("max_distance must be finite and positive")

    rgb = np.full(result.scores.shape + (3,), 80 / 255)
    offsets = result.coordinates[result.valid_mask] - result.center_coordinates
    scores = result.scores[result.valid_mask]
    if offsets.shape[1] == 1:
        hue = np.where(offsets[:, 0] < 0, 0.5, 0.0)
        visible_radius = np.abs(offsets[:, 0])
    else:
        hue = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]) / (2 * np.pi), 1)
        visible_radius = np.linalg.norm(offsets[:, :2], axis=1)
    saturation = np.clip(scores / maximum, 0, 1)
    # Suppress arbitrary colors from round-off at the displayed center.
    saturation[visible_radius <= 100 * np.finfo(float).eps * np.maximum(1, scores)] = 0
    rgb[result.valid_mask] = hsv_to_rgb(np.column_stack(
        (hue, saturation, saturation)
    ))
    return rgb


# convex hull data reduction
def essential_spectral_pixels(mixed_data, npc=3):
    """Select candidate pixel spectra at the extremes of an SVD projection.

    Parameters
    ----------
    mixed_data : numpy.ndarray
        Spectral observations of shape (N, C), with pixels in rows and
        detector channels in columns. Each row must have a usable first
        SVD coordinate.
    npc : int, optional
        Number of SVD components, default 3. Dividing the remaining
        coordinates by the first leaves a projection of dimension npc - 1.

    Returns
    -------
    candidates : numpy.ndarray
        Original spectra of shape (number of candidates, C). In one
        projected dimension, the candidates are the two endpoints.
    indices : numpy.ndarray
        Row indices into mixed_data, in the same order as candidates.

    Notes
    -----
    Candidate rows are useful starting points for blind unmixing. Hull
    membership does not establish that a spectrum comes from one component.
    """

    coordinates, _, valid = _project_spectral_pixels(mixed_data, npc)
    if not valid.all():
        raise ValueError("convex-hull projection requires a usable first SVD coordinate for every pixel")

    # to convex hull we give ndarray of floats (points) with shape (npoints, ndim)
    print((len(mixed_data), coordinates.shape[1] + 1))
    if coordinates.shape[1] > 1:
        hull = ConvexHull(coordinates)
    #idx = hull.vertices
    #hull_points = X_norm[hull.vertices,:]
        hull_vert = hull.vertices
        #important_spec = mixed_data[hull_vert,:]
    else:
        print('special case for the algorithm as nc < 3')
        points_1d = coordinates
        v1 = np.argmin(points_1d)
        v2 = np.argmax(points_1d)
        hull_vert = np.array([v1,v2])

    important_spec = mixed_data[hull_vert,:]
    return important_spec, hull_vert


# Preserve the original spelling for existing callers, including the napari plugin.
essential_spectral_pixles = essential_spectral_pixels


## detection of most unique spectra
def unique_profiles(d, nr, f):
    """Select distinct profiles from the columns of a data matrix.

    Parameters
    ----------
    d : numpy.ndarray
        Matrix with one candidate profile per column. To select spectra,
        pass the transpose of the candidates returned by
        `essential_spectral_pixels()`, so rows are detector channels and
        columns are candidate spectra.
    nr : int
        Number of profiles to select.
    f : float
        Noise offset as a percentage of the largest column mean. For
        example, f=1.0 adds an offset of 1 percent of that mean.

    Returns
    -------
    profiles : numpy.ndarray
        Selected profiles as rows, with shape (nr, number of input rows).
    indices : numpy.ndarray
        Selected column indices into d, in the same order as profiles.
    """


    (nrow, ncol) = d.shape

    # calculation of the purity spectrum
    f = f/100
    sVec  = np.std(d, axis=0, ddof=1)
    mVec  = np.mean(d, axis=0)

    llVec = np.square(sVec) + np.square(mVec)
    f  = np.max(mVec)*f
    pVec   = sVec / (mVec+f)

    imp = np.array([np.argmax(pVec)])
    mp = pVec[imp]


    lVec = np.sqrt( np.square(sVec) + np.square(mVec+f) )

    # normalizing d by value of lVec, this is the same as taking each
    # col of dl and divided by the element of lVec, the newaxis is used
    # because lVec is a vector
    # dl is a matrix with same chape as d
    dl = np.divide(d, lVec[np.newaxis,:])

    # related to covariance
    # c is a matrix with shape ncol x ncol
    # eucledian distance
    c = np.divide(np.dot(dl.T, dl), nrow)


    # calculating weights
    wMat = np.zeros((nr,ncol))
    wMat[0,:] = np.divide(llVec,np.square(lVec))

    pMat = np.zeros((nr,ncol))
    pMat[0,:] = np.multiply(wMat[0,:],pVec)

    sMat = np.zeros((nr,ncol))
    sMat[0,:] = np.multiply(wMat[0,:],sVec)

    for i_idx in range(1,nr): # check if I need nr+1

        for j_idx in range(ncol): # check if I need ncol+1

            dm = wmat(c, imp, i_idx, j_idx)
            wMat[i_idx,j_idx] = np.linalg.det(dm)
            pMat[i_idx,j_idx] = np.multiply(pMat[0,j_idx], wMat[i_idx,j_idx])
            sMat[i_idx,j_idx] =np.multiply(sMat[0,j_idx], wMat[i_idx,j_idx])

        imp_i = np.array([np.argmax(pMat[i_idx,:])])
        mp_i = pMat[i_idx,imp_i]

        mp  = np.concatenate((mp, mp_i))
        imp = np.concatenate((imp, imp_i))

    sp = np.zeros((nrow,nr))

    for i_idx in range(nr):
        # sp is a rearranged copy of d based on imp
        sp[:,i_idx] = d[:,imp[i_idx]]

    sp = normv2(sp.T)

    return sp, imp

def wmat(c, imp, irank, jvar):
    """ similarity among columns of covariance

    this could be a nested function of unique_profiles"""

    dm = np.zeros((irank+1,irank+1))
    dm[0,0] = c[jvar, jvar]

    for k in range(1,irank+1):

        kvar = imp[k-1]
        dm[0,k] = c[jvar, kvar]
        dm[k,0] = c[kvar, jvar]
        for kk in range(1,irank+1):
            kkvar = imp[kk-1]
            dm[k,kk] = c[kvar,kkvar]


    return dm

def normv2(sMatrix):
    """Simple norm
    """

    # this could be a nested function of unique_profiles
    #nrows = sMatrix.shape[0]

    sr = np.sqrt(np.sum(np.square(sMatrix), axis=1))
    sn = np.divide(sMatrix, sr[:,np.newaxis])
    return sn
