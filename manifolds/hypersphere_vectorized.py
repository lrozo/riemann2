"""
This module implements a manifold (akin to a geomstats manifold)
that uses the vectorized versions of the metrics instead of the
ambient space ones. It implements a HypersphereVectorized that
shares methods and attributes with the usual Hypersphere, except
for the metric.

It can only be defined for S2 and S3 for now.
"""

import torch

from geomstats.geometry.hypersphere import Hypersphere

from metrics.sphere_metric_vectorized import SphereMetricVectorized
from metrics.quaternion_metric_vectorized import QuaternionMetricVectorized


class HypersphereVectorized(Hypersphere):
    """
    A vectorized version of the Hypersphere manifold.
    Besides all the usual methods of the Hypersphere,
    this manifold uses a vectorized version of the
    metrics for S2 and S3.

    Vectorized versions for other dimensions have
    not been implemented yet.
    """
    def __init__(
        self,
        dim,
        default_coords_type="extrinsic",
        transport_basis_from: torch.Tensor = None,
    ):
        """
        Initializes the vectorized Hypersphere.

        Parameters
        ----------

        - dim (int): either 2 or 3.

        - default_coords_type (str): the default coordinate type
          which geomstats uses internally.

        - transport_basis_from (torch.Tensor, optional): a point
          on the manifold from which bases are transported smoothly
          in the case of S2. Since S3 is parallelizable (i.e. there
          a smooth frame over all the sphere), we don't need to
          transport it smoothly from anywhere.
        """
        if dim not in [2, 3]:
            raise ValueError("We only support S2 and S3 at the moment.")

        super().__init__(dim, default_coords_type)

        if dim == 2:
            self.equip_with_metric(SphereMetricVectorized, transport_from=transport_basis_from)
        else:
            self.equip_with_metric(QuaternionMetricVectorized)

        # if dim == 2:
        #     self.metric = SphereMetricVectorized(transport_from=transport_basis_from)
        # else:
        #     self.metric = QuaternionMetricVectorized()

