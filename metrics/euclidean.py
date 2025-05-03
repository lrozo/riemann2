"""
This module implements a geomstats Euclidean metric augmented with a logdet function
"""

import torch

from geomstats.geometry.euclidean import CanonicalEuclideanMetric


class AugmentedCanonicalEuclideanMetric(CanonicalEuclideanMetric):
    """Class for the canonical Euclidean metric.

    Notes
    -----
    Metric matrix is identity (NB: `EuclideanMetric` allows
    to use a different metric matrix).
    """

    def __init__(self, space):
        super().__init__(space)

    def logdet(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log determinant of the Jacobian derivative of tangent projection with respect to the operand u. 
        In the Euclidean case, this is equal to 0.

        Parameters
        ----------
        - u (torch.Tensor): Data on the Euclidean manifold of dimension batch x n

        Returns
        -------
        Tensor
            logdet(df/dx) for all data

        Note
        ----
        An example application of this log determinant is to compute the change of variable term from the normal
        distribution to the wrapped normal distribution, i.e., log(p(y)) = log(p(x)) - logdet(df/dx) with y = f(x).
        Here, f is proj_{\mu}(u) and denotes the tangent projection

        """
        logdet_partial = torch.zeros_like(u).sum(dim=-1)

        return logdet_partial.sum()


