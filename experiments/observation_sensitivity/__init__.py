"""Пространственные ядра и ковариационные модели эксперимента E06."""

from .kernels import CompactKernel, GaussianKernel, spatial_weights
from .covariance import (
    CovarianceGuards,
    ExponentialMixture,
    covariance_diagnostics,
    matched_lag_mixture,
    mixture_covariance,
    population_derivative,
    population_objective,
)

__all__ = [
    "CompactKernel", "GaussianKernel", "spatial_weights",
    "CovarianceGuards", "ExponentialMixture", "covariance_diagnostics",
    "matched_lag_mixture", "mixture_covariance", "population_derivative",
    "population_objective",
]
