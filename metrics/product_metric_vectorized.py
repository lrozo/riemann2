"""
Implements a version of the product RiemannianMetric
that behaves well with vectorization and with a combination of
intrinsic & extrinsic manifolds at the same time.

More motivation: the current implementation of geomstats'
product manifold doesn't combine well when considering
e.g. a product like R2xS2, since it expects all points
to be defined either intrinsically or extrinsically. In other
words, geomstats assumes that we want to deal with tangent vectors
in S2 as "extrinsic" vectors lying in R3, instead of "intrinsically"
in the actual tangent space TpM which, in this example, would be R2.

In our implementation we would like to be able to deal with
products of manifolds that are defined using our vectorization.

Moreover, we deal with tangent vectors in an abstract way:
instead of e.g. considering literal tangent vectors to the
manifolds, we think about them as coordinates instead. By abstract,
we mean tangent vectors that are given by the coordinates w.r.t
a basis of the tangent space. This stands in comparison to how
geomstats thinks about tangent spaces as embedded vectors in ambient
space.
"""

from typing import List, Tuple

import torch

from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.product_manifold import ProductRiemannianMetric

# TODO: in an ideal world, we would inherit from ProductRiemannianMetric
# and we would only need to reimplement is_intrinsic and _iterate_over_metrics.
# For our experiments, we'll only need exp and log. We could easily
# implement the rest of the methods once we need them.
class ProductMetricVectorized(ProductRiemannianMetric):
    def __init__(
        self,
        metrics: List[RiemannianMetric],
        manifold_dims: List[int] = None,
        default_point_type="vector",
    ) -> None:
        """
        Initializes the Product metric, given a list of Riemannian metrics
        and (optionally) the dimensions of the manifold points in ambient space.

        Parameters
        ----------
        - metrics (List[RiemannianMetric]): a list of metrics that compose the product.
          The order of the metrics matters, of course.

        - manifold_dims (Optional, List[int]): a list of the ambient space dimensions for
          each manifold. This is used to separate provided points into points for each
          one of the manifolds.
        """
        self.metrics = metrics
        self.tangent_space_dims = [metric.signature[0] for metric in metrics]
        # self.tangent_space_dims = [metric.dim for metric in metrics]
        self.tangent_space_dim = sum(self.tangent_space_dims)

        if manifold_dims is not None:
            self.manifold_dims = manifold_dims
        else:
            manifold_dims = []
            for metric in metrics:
                if isinstance(metric, EuclideanMetric):
                    # Euclidean metrics are in intrinsic coords...
                    manifold_dims.append(metric.signature[0])
                else:
                    # ...the rest are in extrinsic coords.
                    # TODO: this won't work for most manifolds, right?
                    manifold_dims.append(metric.signature[0] + 1)

            self.manifold_dims = manifold_dims

        self.dim = sum(self.manifold_dims)
        self.ambient_space_dim = sum(self.manifold_dims)
        self.dims = manifold_dims

    def exp(self, tangent_vec: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        The exponential map on the product manifold.

        Since we're dealing with the Levi-Civita connection, it suffices
        to compute it independently on both manifolds and then concatenate.

        Parameters
        ----------
        - tangent_vec (torch.Tensor of shape (sum_of_tangent_space_dims,) or
          (b, sum_of_tangent_space_dims)): the tangent vectors in which to
          evaluate the exponential map. These are split into separate tangent
          vectors for each one of the manifolds according to self.tangent_space_dims,
          and the exponentials are computed on a per-metric basis.

        - base_point (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): the base points w.r.t. which we
          compute the exponential map. These are split according to self.manifold_dims
          as with the tangent vectors.

        Returns
        -------
        - exponentials (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): The concatenation of the exponentials
          computed individually per manifold.
        """

        tangent_vecs = self._split_tangent_vec(tangent_vec)
        base_points = self._split_point_on_manifolds(base_point)

        exponentials = [
            metric.exp(t_vec, b_point).flatten(start_dim=1)
            for metric, t_vec, b_point in zip(self.metrics, tangent_vecs, base_points)
        ]

        return torch.hstack(exponentials)

    def log(self, point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        The logarithm map on the product manifold.

        Using the identification T_(p,q) (MxN) == T_pM x T_qN, we consider
        each log of the product to return the concatenation of the tangent
        vectors to each.

        Parameters
        ----------
        - point (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): the points in which to evaluate the logarithm map.
          These are split into separate tangent vectors for each one of the manifolds
          according to self.tangent_space_dims, and the logarithms are computed on a
          per-metric basis.

        - base_point (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): the base points w.r.t. which we
          compute the exponential map. These are split according to self.manifold_dims
          as with the tangent vectors.

        Returns
        -------
        - logarithms (torch.Tensor of shape (sum_of_tangent_space_dims,) or
          (b, sum_of_tangent_space_dims)): The concatenation of the exponentials
          computed individually per manifold.
        """
        points = self._split_point_on_manifolds(point)
        base_points = self._split_point_on_manifolds(base_point)

        logarithms = [
            metric.log(pt, b_point)
            for metric, pt, b_point in zip(self.metrics, points, base_points)
        ]

        return torch.hstack(logarithms)

    def _split_tensor_according_to_dims(
        self, tensor: torch.Tensor, dims: List[int]
    ) -> Tuple[torch.Tensor]:
        if len(tensor.shape) == 1:
            batched = False
            tensor = tensor.unsqueeze(0)
        else:
            batched = True

        tensor_dim = tensor.shape[-1]

        if tensor_dim != sum(dims):
            raise ValueError(
                f"Can't split a vector of shape {tensor.shape} into a "
                f"tuple of tensors with last dimensions {dims}"
            )

        split_tensors = torch.split(tensor, dims, dim=-1)

        if not batched:
            split_tensors = tuple([t_vec.squeeze(0) for t_vec in split_tensors])

        return split_tensors

    def _split_tangent_vec(self, tangent_vec: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Takes a tangent vector and splits it according
        to the dimensions of the metrics.

        Parameters
        ----------
        - tangent_vec (torch.Tensor of shape (sum_of_tangent_space_dims,) or
          (b, sum_of_tangent_space_dims)): the tangent vectors we want to
          split s.t. the output is a tuple of tangent vectors for each of
          the manifolds in the product.

        Returns
        -------
        - splitted_tangent_vecs (Tuple[torch.Tensor] where each tangent vector
          has dim == the tangent space dimension for the respective manifold).
        """
        if len(tangent_vec.shape) == 1:
            batched = False
            tangent_vec = tangent_vec.unsqueeze(0)
        elif len(tangent_vec.shape) == 2:
            batched = True
        else:
            raise ValueError(
                "We are expecting vectorized tangent vectors. "
                "We don't support matrix-valued tangent vectors, "
                "are you using the correct (vectorized) metric?"
            )

        _, tangent_dim = tangent_vec.shape

        if tangent_dim != sum(self.tangent_space_dims):
            raise ValueError(
                "The 1-dimension of the tangent vector should match the sum of metric dimensions"
            )

        tangent_vecs = torch.split(tangent_vec, self.tangent_space_dims, dim=1)

        if not batched:
            tangent_vecs = tuple([t_vec.squeeze(0) for t_vec in tangent_vecs])

        return self._split_tensor_according_to_dims(
            tangent_vec, self.tangent_space_dims
        )

    def inner_product(
        self,
        tangent_vec_a: torch.Tensor,
        tangent_vec_b: torch.Tensor,
        base_point: torch.Tensor,
    ):
        """
        Computes the inner product of two tangent vectors with respect
        to a metric evaluated at a given base point. This inner product
        is computed using the usual identification between T_(p,q)MxN
        and the direct sum T_pM \otimes T_qM

        Parameters
        ----------

        - tangent_vec_a (torch.Tensor of shape (sum_of_tangent_space_dims,) or
          (b, sum_of_tangent_space_dims)).

        - tangent_vec_b (torch.Tensor of shape (sum_of_tangent_space_dims,) or
          (b, sum_of_tangent_space_dims)).

        - base_point (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): The base point(s) at which to evaluate
          the inner product.

        Returns
        -------

        - inner_product (torch.Tensor): the sum of the individual inner products,
          each computed w.r.t. an individual manifold. (since T_(p,q)MxN == T_pM x T_qN)
        """
        if len(tangent_vec_a.shape) == 1:
            tangent_vec_a = tangent_vec_a.unsqueeze(0)
            batched_a = False
        else:
            batched_a = True

        if len(tangent_vec_b.shape) == 1:
            tangent_vec_b = tangent_vec_b.unsqueeze(0)
            batched_b = False
        else:
            batched_b = True

        if len(base_point.shape) == 1:
            base_point = base_point.unsqueeze(0)
            batched_base_point = False
        else:
            batched_base_point = True

        inner_products = [
            metric.inner_product(
                tangent_vec_a[..., i, :],
                tangent_vec_b[..., i, :],
                base_point[..., i, :],
            )
            for i, metric in enumerate(self.metrics)
        ]

        if not all((batched_a, batched_b, batched_base_point)):
            return sum(inner_products).squeeze(0)
        else:
            return sum(inner_products)

    def dist(self, point_a: torch.Tensor, point_b: torch.Tensor, **kwargs):
        """
        Computes the distance between two points on the product manifold
        using the property that the norm of the Logarithm is the distance.
        This function does not handle the case where (one of) the arguments'
        shape is [..., dim] (at least when using sphere_metric_vectorized).
        In this case, use square_dist below.

        Parameters
        ----------

        - point_a (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)).

        - point_b (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)).

        Returns
        -------

        - distance(s) between point_a and point_b.
        """
        log_ = self.log(point_a, point_b)
        squared_distance = self.inner_product(log_, log_, point_a)

        return torch.sqrt(squared_distance)
    
    def squared_dist(self, point_a, point_b):
        """
        Computes the distance between two points on the product manifold
        using the property that the squared distance is the sum of the 
        squared distances for each manifold.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        points_a = self._split_point_on_manifolds(point_a)
        points_b = self._split_point_on_manifolds(point_b)

        squared_dists = [
            metric.squared_dist(pt_a, pt_b) for metric, pt_a, pt_b in zip(self.metrics, points_a, points_b)
        ]

        return torch.stack(squared_dists).sum(dim=0)

    def _split_point_on_manifolds(self, point: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Splits the point according to the dimensions of the metrics.
        """
        manifolds_dim = point.shape[-1]

        if manifolds_dim != sum(self.manifold_dims):
            raise ValueError(
                "The dimension of the point doesn't match the sum of manifold dimensions."
                " Did you forget to add custom manifold dimensions?"
            )

        return self._split_tensor_according_to_dims(point, self.manifold_dims)

    def metric_matrix(self, base_point: torch.Tensor):
        """
        Returns a matrix by blocks, where each block {n} is just the
        matrix representation of the metric {n} in the product.

        Parameters
        ----------

        - base_point (torch.Tensor of shape (sum_of_manifold_dims,) or
          (b, sum_of_manifold_dims)): the base point(s) where we evaluate
          the metric and get its matrix representation.

        Returns
        -------

        - metric_matrix (torch.Tensor of shape (sum_of_tangent_space_dims,
          sum_of_tangent_space_dims) or (b, sum_of_tangent_space_dims,
          sum_of_tangent_space_dims)): a matrix (or batch of matrices)
          representation of the metric evaluated at the given basepoint(s).
        """
        if len(base_point.shape) == 1:
            batched = False
            base_point = base_point.unsqueeze(0)
        elif len(base_point.shape) == 2:
            batched = True

        metric_matrix_of_product = torch.zeros(
            (base_point.shape[0], self.tangent_space_dim, self.tangent_space_dim)
        )

        current_tangent_dim = 0
        current_manifold_dim = 0
        for metric, tangent_space_dim, manifold_dim in zip(
            self.metrics, self.tangent_space_dims, self.manifold_dims
        ):
            current_batch = metric.metric_matrix(
                base_point=base_point[
                    :, current_manifold_dim : current_manifold_dim + manifold_dim
                ]
            )
            metric_matrix_of_product[
                :,
                current_tangent_dim : (current_tangent_dim + tangent_space_dim),
                current_tangent_dim : (current_tangent_dim + tangent_space_dim),
            ] = current_batch

            current_tangent_dim += tangent_space_dim
            current_manifold_dim += manifold_dim

        if not batched:
            metric_matrix_of_product = metric_matrix_of_product.squeeze(0)

        return metric_matrix_of_product

    def logdet(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the log determinant of the Jacobian derivative of tangent projection with respect to the operand u. 
        An example application of this log determinant is to compute the change of variable term from the normal
        distribution to the wrapped normal distribution, i.e., log(p(y)) = log(p(x)) - logdet(df/dx) with y = f(x).
        Here, f is proj_{\mu}(u) and denotes the tangent projection

        """
        tangent_vecs = self._split_tangent_vec(u)

        logdet_partial = [
            metric.logdet(tangent_vec) for metric, tangent_vec in zip(self.metrics, tangent_vecs)
        ]

        return torch.stack(logdet_partial).sum(dim=0)