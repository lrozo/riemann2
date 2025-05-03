"""
Implements a WrappedGaussianDistribution on top
of the MultivariateNormal implementation in GPyTorch.

Wrapped Gaussian Distributions can be thought of as
the "pushforward" of the usual Gaussian Distribution
onto a manifold via the exponential map. For more details,
check [Mallasto18].

[Mallasto18]: Wrapped Gaussian Process Regression on Riemannian Manifolds.
"""
from typing import Union

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class WrappedGaussianDistribution(MultitaskMultivariateNormal):
    """
    A wrapper around the MultitaskMultivariateNormal distribution
    of gpytorch, which sends samples and the mean from the tangent
    space to the manifold using the exponential map.

    Attributes
    ----------
    - manifold (type: geomstats.geometry.manifold.{Manifold or RiemannianMetric}):
      An object which implements the exponential map. Check geomstats.geometry for
      examples.

    - basepoints (type: torch.Tensor): the base points w.r.t. which we compute
      the exponential map.

    - tangent_mean (type: torch.Tensor): the Euclidean mean on the tangent space.

    - tangent_covariance (type: torch.Tensor): the Euclidean covariance matrix in
      tangent space.

    Methods
    -------
    - rsample(sample_shape=...): A re-implementation of the rsample from the original
      MultitaskMultivariateGaussian that pushes the Euclidean samples in tangent space
      forward using the exponential map.

    - mean() (a @property): A re-implementation of the mean property that sends it
      to the manifold using the exponential map.
    """

    def __init__(
        self,
        manifold: Union[Manifold, RiemannianMetric],
        basepoints: torch.Tensor,
        tangent_mean: torch.Tensor,
        tangent_covariance: torch.Tensor,
        validate_args=False,
        interleaved=True,
    ):
        self.manifold = manifold
        self.basepoints = basepoints

        self.tangent_mean = tangent_mean
        self.tangent_covariance = tangent_covariance

        # Under the hood, we'll still have a Euclidean Gaussian.
        super().__init__(tangent_mean, tangent_covariance, validate_args, interleaved)

    def rsample(self, sample_shape=..., base_samples=None):
        """
        Returns samples on the manifold (w.r.t which we could backpropagate).

        Parameters
        ----------
        - sample_shape (type: torch.Size, optional)
        """
        if isinstance(self.manifold, Manifold):
            metric = self.manifold.metric
        else:
            metric = self.manifold

        tangent_samples = super().rsample(sample_shape, base_samples)
        samples_in_manifold = metric.exp(tangent_samples, base_point=self.basepoints)
        return samples_in_manifold

    @property
    def mean(self):
        """
        Returns the pushforward of the Euclidean mean in tangent space.
        """
        if isinstance(self.manifold, Manifold):
            metric = self.manifold.metric
        else:
            metric = self.manifold

        manifold_mean = metric.exp(self.tangent_mean, base_point=self.basepoints)
        return manifold_mean
