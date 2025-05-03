"""
Implements a manifold wrapper around our ProductMetricVectorized.
This is required by the pymanopt optimization, which we are not
currently using. Most of the use-cases of this product manifold
are already covered by the ProductMetricVectorized itself.
"""

from typing import List
import geomstats

from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_manifold import ProductManifold
from metrics.product_metric_vectorized import ProductMetricVectorized


class ProductManifoldVectorized(ProductManifold):
    """
    Implements a Product Manifold wrapper around geomstats',
    but using our vectorized Product Metric instead. This allows
    us to wrap these manifolds for pymanopt optimizations (which,
    at the moment, we are not using).
    """

    def __init__(
        self,
        manifolds: List[Manifold],
        manifold_dims: List[int] = None,
        metrics=None,
        **kwargs
    ):
        """
        Initializes the product manifold.

        Parameters
        ----------

        - manifolds (List[Manifold]), order matters.

        - manifold_dims (List[int], optional): a list with
          the actual dimensions of the different manifolds,
          since geomstats sometimes confuses tangent_space_dim
          with ambient_space_dim.

        - metrics (List[RiemannianMetric], optional): a list
          with the metrics, if they are not to be inferred from
          the provided manifolds.
        """
        # This implementation directly copies that of geomstats,
        # but modifies it to use our ProductMetricVectorized as
        # the default metric type.

        # Hyperparameters required by geomstats.
        default_point_type = "vector"
        n_jobs = 1
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.dims = [manifold.dim for manifold in manifolds]
        if metrics is None:
            metrics = [manifold.metric for manifold in manifolds]

        kwargs.setdefault(
            "metric",
            ProductMetricVectorized(metrics, manifold_dims=manifold_dims),
        )
        dim = sum(self.dims)
        shape = (sum([m.shape[0] for m in manifolds]),)

        # super(ProductManifold, self).__init__(
        #     dim=dim,
        #     shape=shape,
        #     default_point_type=default_point_type,
        #     **kwargs,
        # )
        # self.manifolds = manifolds
        # self.n_jobs = n_jobs
        super().__init__(factors=manifolds, equip=False)
        # super().__init__(factors=manifolds, default_point_type=default_point_type)

        self.metric = ProductMetricVectorized(metrics, manifold_dims=manifold_dims)
        # TODO: the clean way to initialize the metric would be below, needs to be debugged
        # self.equip_with_metric(ProductMetricVectorized, metrics=metrics, manifold_dims=manifold_dims)

