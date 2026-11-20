"""Offline tools for generating synthetic spectral images."""

from .detector import SpectralDetector
from .mixing import simulate_linear_mixture
from .noise import add_confocal_noise
from .psf import GaussianPSF
from .render import SamplingGrid, SpatialRender, render_scene
from .result import NoisySimulationResult, SyntheticDataset
from .scene import Ellipse, Line, Rectangle, Scene2D
from .simulate import simulate_noisy_image
from .spatial import make_abundance_maps, make_mixture_gradient
from .spectra import load_offline_emission_spectrum, resample_spectrum

__all__ = [
    "SyntheticDataset",
    "NoisySimulationResult",
    "SpectralDetector",
    "Scene2D",
    "Ellipse",
    "Rectangle",
    "Line",
    "SamplingGrid",
    "GaussianPSF",
    "SpatialRender",
    "render_scene",
    "add_confocal_noise",
    "make_abundance_maps",
    "make_mixture_gradient",
    "resample_spectrum",
    "load_offline_emission_spectrum",
    "simulate_linear_mixture",
    "simulate_noisy_image",
]
