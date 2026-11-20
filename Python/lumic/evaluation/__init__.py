"""Reconstruction, spectral stability, and ground-truth recovery diagnostics."""

from .metrics import (AlignedFactors, FactorEvaluation, SpectralComparison,
                      align_factors, compare_spectra, evaluate_factors, match_components)
from .maps import (ConcentrationErrorMaps, ReconstructionErrorMaps,
                   concentration_error_maps, reconstruction_error_maps)
from .residuals import (ResidualPCAResult, residual_pca,
                       residual_pca_from_residuals, residual_pca_from_factors)
from .correlation import residual_neighbor_correlation, residual_spectral_correlation
from .sampling import ResidualPixelSelection, select_residual_pixels

__all__ = ["residual_neighbor_correlation", "residual_spectral_correlation",
           "FactorEvaluation", "evaluate_factors", "match_components",
           "AlignedFactors", "SpectralComparison", "align_factors", "compare_spectra",
           "ConcentrationErrorMaps", "ReconstructionErrorMaps",
           "concentration_error_maps", "reconstruction_error_maps",
           "ResidualPCAResult", "residual_pca", "residual_pca_from_residuals",
           "residual_pca_from_factors", "ResidualPixelSelection", "select_residual_pixels"]
