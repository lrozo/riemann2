"""
Implements a 'vectorized' version of the S3 metric. By this,
we mean a metric that expects to receive 3-dimensional
tangent vectors instead of 4-dimensional ones.

We need this vectorized input to be able to define a Euclidean
Gaussian Process on only the tangent space (instead of the entire
ambient space R4).
"""

from typing import List
import torch
import numpy as np

from geomstats.geometry.hypersphere import HypersphereMetric, Hypersphere

torch.set_default_dtype(torch.float64)


class QuaternionMetricVectorized(HypersphereMetric):
    """
    A metric for the quaternions that expects tangent vectors
    in R3 instead of R4.

    The exponential and logarithm maps are re-implemented by
    considering the tangent vector inputs as the coordinates of
    the following global frame in S3:

    X1(w, x, y, z) = (-x, w, -z, y)
    X2(w, x, y, z) = (-y, z, w, -x)
    X3(w, x, y, z) = (-z, -y, x, w)

    Notice how this defines a basis at any basepoint (w, x, y, z) of S3.

    In other words, exp expects a 3-dimensional vector (a, b, c) which
    represents the coordinates of X1, X2 and X3. The tangent vector associated
    with (a, b, c) in R3 is a*X1 + b*X2 + c*X3, now living tangential to the
    sphere in R4.
    """

    def __init__(self, space):
        """
        This quaternion metric is a wrapper around the Hyperspherical
        metric S3. The exponential and logarithm maps are modified s.t.
        tangent vectors live in R3 instead of R4.
        """
        # dim = 3
        super().__init__(space)

    def exp(self, coords: torch.Tensor, base_point: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The exponential map, taking the coordinates of the tangent vectors
        with respect to the basis

        {(-x, w, -z, y), (-y, z, w, -x), (-z, -y, x, w)}

        where (w, x, y, z) is the basepoint. This coordinates get mapped to
        tangent vectors to the hypersphere in R4.

        Parameters:
            - coordinates (torch.Tensor of shape (3,) or (b, 3)).
            - base_point (torch.Tensor of shape (4,) or (b, 4)).
        """
        if not Hypersphere(3).belongs(base_point.type(torch.float64)).all():
            raise ValueError("Basepoints should be on the Hypersphere S3")

        # Coordinates are assumed to be living in R3. We take the linear combination
        # w.r.t. the global frame.
        tangent_vec = self._from_coordinates_to_tangent_vec(coords, base_point)

        # The way pymanopt computes it:
        norm_of_vector = torch.linalg.norm(tangent_vec, dim=1, keepdims=True)

        return base_point * torch.cos(norm_of_vector) + tangent_vec * torch.sinc(
            norm_of_vector / np.pi
        )

        # The way geomstats computes it is, literally, a weird approximation
        # using a Taylor expansion of the exp.
        # return super().exp(tangent_vec, base_point, **kwargs)

    def log(
        self, point: torch.Tensor, base_point: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        The logarithm map, returning the coordinates of the tangent vector w.r.t
        the global frame given by {(-x, w, -z, y), (-y, z, w, -x), (-z, -y, x, w)}
        where (w, x, y, z) is the basepoint.

        Parameters:
            - point (torch.Tensor of shape (4,) or (b, 4)): target point.
            - base_point (torch.Tensor of shape (4,) or (b, 4)): basepoint from which
              we "shoot" the tangent vector.
        """
        if not Hypersphere(3).belongs(point.type(torch.float64)).all():
            raise ValueError("The point should belong to the hypersphere S3")

        if not Hypersphere(3).belongs(base_point.type(torch.float64)).all():
            raise ValueError("Basepoints should be on the Hypersphere S3")

        # The original log returns a tangent vector in R4...
        tangent_vec = super().log(point, base_point, **kwargs).type(torch.float64)

        # ...so we compute its coordinates and return it.
        return self._from_tangent_vec_to_coordinates(tangent_vec, base_point)

    def _from_coordinates_to_tangent_vec(
        self, coords: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns a tangent vector with coordinates {coords} according
        to the basis

        {(-x, w, -z, y), (-y, z, w, -x), (-z, -y, x, w)}

        which spans the tangent space at the basepoint (w, x, y, z).

        Parameters:
            - coords (torch.Tensor of shape (3,) or (b, 3)): the coordinates
              of the tangent vector w.r.t. the global frame.
            - base_point (torch.Tensor of shape (4,) or (b, 4)): the basepoint
              that generates the basis.
        Returns:
            - tangent_vec (torch.Tensor of shape (4,) or (b, 4)): a tangent
              vector given by the linear combination of the coordinates and
              the basis.

        TODO: there must be a way to vectorize this.
        """
        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)
            batched = False
        elif len(coords.shape) == 2:
            batched = True
        else:
            raise ValueError(
                "We expect coordinates to be of shape (b, n) where n is the dimension of the manifold"
            )

        basis = self._basis_from_base_point(base_point)

        tangent_vec = (
            coords[:, 0].unsqueeze(1) * basis[0]
            + coords[:, 1].unsqueeze(1) * basis[1]
            + coords[:, 2].unsqueeze(1) * basis[2]
        )

        if not batched:
            tangent_vec = tangent_vec.squeeze(0)

        return tangent_vec

    def _from_tangent_vec_to_coordinates(
        self, tangent_vec: torch.Tensor, base_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the coordinates of the tangent vector w.r.t.
        the basis spanned by the base point.

        Parameters:
            - tangent_vec (torch.Tensor of shape (4,) or (b, 4)).
            - base_point (torch.Tensor of shape (4,) or (b, 4)).
        """
        # TODO: Deal when the base_point is only 1, and
        # the user means to consider it as a constant function.

        if len(base_point.shape) == 1 and len(tangent_vec.shape) > 1:
            base_point = base_point.unsqueeze(0)

        if base_point.shape[0] == 1 and tangent_vec.shape[0] > 1:
            base_point = np.repeat(base_point, tangent_vec.shape[0], axis=0)

        basis = self._basis_from_base_point(base_point)
        basis_matrix = torch.cat([b_vector.unsqueeze(-1) for b_vector in basis], dim=-1)

        # Cast to the correct dtype
        if tangent_vec.dtype != basis_matrix.dtype:
            basis_matrix = basis_matrix.type(tangent_vec.dtype)

        # Since the system actually has a solution, I expect this
        # least squares (i.e. multiplying by the pseudoinverse of the
        # basis matrix) to actually converge pretty close to the original.

        # Tests with random coordinates say yes, this works (with a presicion
        # of 1e-8 with double precision). I wonder if there's a way to implement this solution better,
        # knowing that the tangent vecs actually live in the span of the basis.
        coords = torch.linalg.lstsq(basis_matrix, tangent_vec).solution

        return coords

    def _basis_from_base_point(self, base_point: torch.Tensor) -> List[torch.Tensor]:
        """
        Given a basepoint (w, x, y, z), returns the basis

        [(-x, w, -z, y), (-y, z, w, -x), (-z, -y, x, w)]

        as a list of torch.Tensors.

        Parameters:
            - base_point (torch.Tensor of shape (4,) or (b, 4)).

        Returns:
            - basis (List[torch.Tensor], each Tensor of shape (4,) or (b, 4)).

        TODO: there must be a prettier way to do this.
        """
        if len(base_point.shape) == 1:
            batched = False
            base_point = base_point.unsqueeze(0)
        else:
            batched = True

        w = base_point[:, 0].unsqueeze(0)
        x = base_point[:, 1].unsqueeze(0)
        y = base_point[:, 2].unsqueeze(0)
        z = base_point[:, 3].unsqueeze(0)

        basis = [
            torch.vstack((-x, w, -z, y)).T,
            torch.vstack((-y, z, w, -x)).T,
            torch.vstack((-z, -y, x, w)).T,
        ]

        return basis

    def metric_matrix(self, base_point):
        """
        Returns the metric matrix by embedding the tangent vectors
        into an ambient space, computing the corresponding components of the
        matrix, and returning a 3x3 matrix representation.

        Parameters
        ----------
        - base_point (torch.Tensor of shape (4,) or (b, 4)): the point
          in the manifold w.r.t. which we want to compute the metric.

        Returns
        -------
        - metric_matrix (torch.Tensor of shape (3, 3) or (b, 3, 3)): the
          matrix representation of the metric at the given basepoints.

        TODO: are we doing the right thing? This essentially embeds the
        coordinates into an actual tangent vector to the sphere, and then
        computes the Euclidean dot product between them. In theory, this is
        exactly how the metric of the sphere is defined: the pullback of the
        immersion of the sphere into R3. Yet, we don't take into account
        the derivative of this immersion and focus on computing the Euclidean
        dot products of the "immersed" tangent vectors directly...
        """
        if len(base_point.shape) == 1:
            batched = False
            base_point = base_point.unsqueeze(0)
        else:
            batched = True

        # Getting the transformations of the basis vectors
        basis_in_ambient_space = self._basis_from_base_point(base_point)
        first_basis_vector = basis_in_ambient_space[0].unsqueeze(-1)
        second_basis_vector = basis_in_ambient_space[1].unsqueeze(-1)
        third_basis_vector = basis_in_ambient_space[2].unsqueeze(-1)

        # Computing the symmetric part of the matrix
        metric_matrix = torch.zeros((base_point.shape[0], 3, 3))
        metric_matrix[:, 0, 0] = torch.bmm(
            first_basis_vector.permute(0, 2, 1), first_basis_vector
        ).flatten()
        metric_matrix[:, 0, 1] = torch.bmm(
            first_basis_vector.permute(0, 2, 1), second_basis_vector
        ).flatten()
        metric_matrix[:, 0, 2] = torch.bmm(
            first_basis_vector.permute(0, 2, 1), third_basis_vector
        ).flatten()
        metric_matrix[:, 1, 1] = torch.bmm(
            second_basis_vector.permute(0, 2, 1), second_basis_vector
        ).flatten()
        metric_matrix[:, 1, 2] = torch.bmm(
            second_basis_vector.permute(0, 2, 1), third_basis_vector
        ).flatten()
        metric_matrix[:, 2, 2] = torch.bmm(
            third_basis_vector.permute(0, 2, 1), second_basis_vector
        ).flatten()

        # and symmetrizing the rest.
        metric_matrix[:, 1, 0] = metric_matrix[:, 0, 1]
        metric_matrix[:, 2, 0] = metric_matrix[:, 0, 2]
        metric_matrix[:, 2, 1] = metric_matrix[:, 1, 2]

        # Dealing with batches.
        if not batched:
            metric_matrix = metric_matrix.squeeze(0)

        return metric_matrix
    
    def logdet(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log determinant of the Jacobian derivative of tangent projection with respect to the operands
        x and u. This method was adapted from https://github.com/oskopek/mvae/blob/master/mt/mvae/ops/spherical.py

        .. math::

            \text{log det}(\partial f / \partial u) = (n-1)\log (\frac{R \: | \sin
            \bigl( \frac{\|\boldsymbol{u}\|_2}{R} \bigl) |}{\|\boldsymbol{u}\|_2} )

        Parameters
        ----------
        - u (torch.Tensor): Data on the sphere manifold of dimension batch x n

        Returns
        -------
        Tensor
            logdet(df/dx) for all data

        Note
        ----
        (1) An example application of this log determinant is to compute the change of variable term from the normal
        distribution to the wrapped normal distribution, i.e., log(p(y)) = log(p(x)) - logdet(df/dx) with y = f(x).
        Here, f is proj_{\mu}(u) and denotes the tangent projection
        (2) This log determinant is the same as the log determinant of the exponential map, as the log
        determinant of the derivative of parallel transport is zero.

        """
        # Geoopt version
        # norm_u = u.norm(dim=-1)
        # val = torch.abs(sindiv(norm_u)).log()
        # logdet_partial = (u.shape[-1] - 2) * val

        assert torch.isfinite(u).all()
        # det [(\partial / \partial v) proj_{\mu}(v)] = (|sin(r)| / r)^(n-1)
        r = torch.norm(u, dim=-1, p=2)  
        n = u.shape[-1] - 1
        # We clamp due to the numerical instability of logs for values close to zero
        logdet_partial = (n - 1) * (torch.log(torch.abs(torch.sin(r)).clamp(min=1e-5)) - torch.log(r.clamp(min=1e-5)))

        assert torch.isfinite(logdet_partial).all()
        return logdet_partial.sum()


if __name__ == "__main__":
    # TODO: move this to a unit test.
    metric = QuaternionMetricVectorized()

    coords = torch.randn((100, 3))
    base_point = np.repeat(torch.Tensor([[0, 0, 0, 1]]), 100, axis=0)

    tangent_vec = metric._from_coordinates_to_tangent_vec(coords, base_point)
    new_coords = metric._from_tangent_vec_to_coordinates(tangent_vec, base_point)
    assert torch.isclose(coords, new_coords, atol=1e-5).all()
