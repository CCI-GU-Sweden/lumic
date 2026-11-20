"""Public Lumic entry points and legacy array helpers."""

import numpy as np
from scipy import ndimage

from .image import SpectralImage, UnfoldedImage, RGBRendering
from .spectra import Spectrum
from .unmixing import (
    MCRALSDiagnostics,
    lof,
    mcr_als,
    normalize_components,
    positive_least_squares,
)
from .utils import AnalysisPixelSelection, select_analysis_pixels
# Keep existing lumic.* calls available while grouping spectral selection in
# one module alongside the pixel distinctness map.
from .uniqueness import (
    SpectralUniquenessResult,
    spectral_uniqueness_map,
    spectral_uniqueness_rgb,
    essential_spectral_pixles,
    unique_profiles,
    wmat,
    normv2,
)

## generic functions used to work with matrices
def unfold_spec_data(spec_data, spec_dim = 0):
    """Unfolds a spectral dataset from a 3D [C, Y, X] to a 2D matrix [pixel-idx, C]

    Flattening of the iamge is a a very common step in linear unmixing. This is meant to
    be a helper function to change from the iamge representation to the matrix one together
    with fold_flat_spec_data

    Parameters
    ----------
    spec_data : numpyArray
        spectral dataset 3D [C, Y, X]

    spec_dim : int
        For the moment unused but can be used to specify the dim of the channels

    Returns
    -------
    flattened_data : numpyArray 
        Flattened representation of the spectral dataset 2D [pixel-idx, C]

    See Also
    --------
    fold_flat_spec_data : folds back a flattened dataset
    """
    
    assert spec_dim ==0, "I will modify this function later, for the moment we asume that the spec is dim 0"
    assert spec_data.shape[0] > 1,"a single pix spectra is not expected"
    assert spec_data.ndim == 3,"the data should be 3-dim dim0: spec dim1: ypix, dim2: xpix"
    
    flattened_data = np.reshape(spec_data,[spec_data.shape[0],spec_data.shape[1]*spec_data.shape[2]]).T
    
    return flattened_data

def fold_flat_spec_data(flat_data, spec_data_shape):
    """Folds back a flattened_data into a spec_data

    Parameters
    ----------
    flat_data : numpyArray
        flattened data 2D matrix [pixel-idx, C]

    spec_data_shape : tuple, int
        shape of the dataset once is folded back into iamges

    Returns
    -------
    folded_data : numpyArray 
        spectral dataset 3D [C, Y, X]

    See Also
    --------
    unfold_spec_data : flattens a spectral dataset for unmixing purposes
    """
    
    #data should enter with dim0: spec, dim1: pix
    assert isinstance(spec_data_shape, tuple), 'spec_data_shape must be a tuple'
    assert len(spec_data_shape) == 3, 'we expect the spec_data_shape to be in [C, Y, X]'

    assert flat_data.ndim == 2,"the data should be 2-dim dim0: unfolded pix pos dim1: spec wavel"
    assert flat_data.shape[1]==spec_data_shape[0]

    # transpose it for consistency
    flat_data = flat_data.T

    folded_data = np.reshape(flat_data,[spec_data_shape[0], spec_data_shape[1],spec_data_shape[2]])

    return folded_data

def fold_abundance_matrix(abundance, spec_data_shape):
    """Folds abbundance matrix into iamges after linear unmixing

    Parameters
    ----------
    abundance : numpyArray 2D
        flattened data 2D matrix [pixel-idx, C], here C is given by the number of components used during unmixing

    spec_data_shape : tuple, int
        shape of the dataset once is folded back into iamges, we will ignore the dim0 as it can be different than the 
        desired number of components after unmixing

    Returns
    -------
    folded_data : numpyArray 
        spectral dataset 3D [C, Y, X]
        
    See Also
    --------
    unfold_spec_data : flattens a spectral dataset for unmixing purposes
    """
    
    #abundance should enter with dim0: pix, dim1: n_components
    assert abundance.ndim == 2
    assert spec_data_shape[1]*spec_data_shape[2] == abundance.shape[0] 

    # transpose it for consistency
    abundance = abundance.T
    folded_data = np.reshape(abundance,[abundance.shape[0], spec_data_shape[1],spec_data_shape[2]])

    return folded_data

## data augmentation functions
def augmentation_convolution(raw_data,kernels, norm_ks = True):
    """Spectral data augmentation using kernels
    
    this is very usefull when some components are very rare, works because biological structures
    have a shape, so spatial information and correlations carry a lot of info
    
    Parameters
    ----------
    raw_data : int
        numpy array of ints (image) in the format of spectral imaging [C, Y, X]

    kernels : tuple
        tuple that contains kernels to be used
    
    norm_ks : boolean
        to force the normalization of kernels, if not this might give more weights to 
        some convolutions than others
    """
    assert isinstance(kernels, tuple), 'kernels must be given as a tuple'
    nChannels = raw_data.shape[0]
    aug_data = np.ndarray([nChannels*len(kernels), raw_data.shape[1], raw_data.shape[2]], dtype=raw_data.dtype)
    
    for k_idx in range(len(kernels)):
        k = kernels[k_idx]
        if norm_ks:
            k = np.divide(k, np.sum(k))
        
        for im_idx, im in enumerate(raw_data):
            aug_data[im_idx+k_idx*nChannels,:,:] = ndimage.convolve(im, k, mode='constant', cval=0.0)

    return aug_data
