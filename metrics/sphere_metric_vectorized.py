"""
Implements a vectorized version of the Spherical metric
by choosing, locally, the basis given by the orthogonal
points to the basepoint.
"""

from typing import List
import torch
import numpy as np

from geomstats.geometry.hypersphere import HypersphereMetric, Hypersphere


class LeastSquares:
    def __init__(self):
        pass

    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: b x m x n
        :param Y: b x n x 1
        """
        # print(A.shape)
        if len(A.shape) == 2:
            # Assuming A is m x n
            A = A.unsqueeze(0)

        if len(Y.shape) in [1, 2]:
            if len(Y.shape) == 1:
                Y = Y.unsqueeze(0).unsqueeze(-1)
            elif len(Y.shape) == 2:
                # Assmuing Y is b x n
                Y = Y.unsqueeze(-1)

        # Assuming A to be full column rank
        cols = A.shape[2]
        if (cols == torch.linalg.matrix_rank(A)).all():
            q, r = torch.linalg.qr(A)
            x = torch.bmm(torch.bmm(torch.inverse(r), q.permute(0, 2, 1)), Y)
        else:
            if (Y == torch.zeros_like(Y)).all():
                return torch.zeros_like(Y).squeeze(-1)
            else:
                raise ValueError(
                    "Non-invertible matrix, or the vector provided is different from null"
                )

        return x.squeeze(-1)


class SphereMetricVectorized(HypersphereMetric):
    """
    A metric for the sphere that expects tangent vectors
    in R2 instead of R3.

    The exponential and logarithm maps are re-implemented by
    considering the tangent vector inputs as the coordinates of
    a smooth local frame that is given by the orthogoal space
    to the given basepoint. In other words, if (x, y, z) in S2 is
    a basepoint, we consider the orthonormal basis spawned by:

    X1(x, y, z) = (-z, 0, x)
    X2(x, y, z) = (0, z, -y)

    This defines a basis the tangent space to (x, y, z) only when
    x and y are different from +-1. In those cases, other bases
    could be defined. The process for defining this basis was considering
    the tangent vectors (-z, 0, x) and (0, z, -y) and orthonormalizing them.

    In other words, exp expects a 2-dimensional vector (a, b) which
    represents the coordinates of X1 and X2. The tangent vector associated
    with (a, b) in R3 is a*X1 + b*X2, now living tangential to the
    sphere in R3.
    """

    def __init__(self, space, transport_from: torch.Tensor = None):
        """
        This quaternion metric is a wrapper around the Hyperspherical
        metric S2. The exponential and logarithm maps are modified s.t.
        tangent vectors live in R2 instead of R3.

        Parameters
        ----------
        - transport_from (torch.Tensor, optional): a point on the sphere
          from which we will transport the basis smoothly along geodesics.
          This is a way to ensure smoothness of basis choice, at least for
          most points on a given trajectory (if the transport_from isn't
          antipodal to any of them).
        """
        # dim = 2
        super().__init__(space)

        self.transport_from = transport_from
        self.ambient_space_dim = self._space.dim + 1
        self.tangent_space_dim = self._space.dim
        # if self.transport_from is not None:
        #     self._basis = self._basis_from_base_point(base_point=transport_from)

    def exp(
        self, coords: torch.Tensor, base_point: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        The exponential map, taking the coordinates of the tangent vectors
        with respect to the orthogonal basis spanned by

        X1(x, y, z) = (-z, 0, x)
        X2(x, y, z) = (0, z, -y)

        where (x, y, z) is the basepoint. This coordinates get mapped to
        tangent vectors to the hypersphere in R3.

        Parameters:
            - coords (torch.Tensor of shape (2,) or (b, 2)).
            - base_point (torch.Tensor of shape (3,) or (b, 3)).
        """
        if isinstance(base_point, np.ndarray):
            base_point = torch.from_numpy(base_point)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        if not Hypersphere(2).belongs(base_point).all():
            if torch.isclose(
                torch.linalg.norm(base_point), torch.tensor(1.0).type(base_point.dtype)
            ):
                print("The basepoint didn't belong to the sphere, projecting")
                base_point = Hypersphere(2).projection(base_point)
            else:
                raise ValueError("Basepoints should be on the Sphere S2")

        # Coordinates are assumed to be living in R3. We take the linear combination
        # w.r.t. the global frame.
        # TODO: fix the transport_from.
        tangent_vec = self._from_coordinates_to_tangent_vec(
            coords, base_point, transport_from=self.transport_from
        )

        return super().exp(tangent_vec, base_point, **kwargs)

    def log(
        self, point: torch.Tensor, base_point: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        The logarithm map, returning the coordinates of the tangent vector w.r.t
        the global frame given by {(-x, w, -z, y), (-y, z, w, -x), (-z, -y, x, w)}
        where (w, x, y, z) is the basepoint.

        Parameters:
            - point (torch.Tensor of shape (3,) or (b, 3)): target point.
            - base_point (torch.Tensor of shape (3,) or (b, 3)): basepoint from which
              we "shoot" the tangent vector.
        """
        # if not Hypersphere(2).belongs(point.type(torch.float)).all():
        if not Hypersphere(2).belongs(point).all():
            raise ValueError("The point should belong to the sphere S2")

        # if not Hypersphere(2).belongs(base_point.type(torch.float)).all():
        if not Hypersphere(2).belongs(base_point).all():
            if torch.isclose(
                torch.linalg.norm(base_point, dim=1),
                torch.ones((len(base_point),), dtype=torch.float32),
            ).all():
                print(
                    "Points were not in the manifold, but they were close to 1.0 in norm. Projecting"
                )
                base_point = Hypersphere(2).projection(base_point)
            else:
                raise ValueError("Basepoints should be on the sphere S2")

        # The original log returns a tangent vector in R4...
        tangent_vec = super().log(point, base_point, **kwargs).type(torch.float64)

        # ...so we compute its coordinates and return it.
        return self._from_tangent_vec_to_coordinates(
            tangent_vec, base_point, transport_from=self.transport_from
        )

    def _from_coordinates_to_tangent_vec(
        self,
        coords: torch.Tensor,
        base_point: torch.Tensor,
        transport_from: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns a tangent vector with coordinates {coords} according
        to the orthonormal basis associated with.

        X1(x, y, z) = (-z, 0, x)
        X2(x, y, z) = (0, z, -y)

        which spans the tangent space at the basepoint (x, y, z).

        Parameters:
            - coords (torch.Tensor of shape (3,) or (b, 3)): the coordinates
              of the tangent vector w.r.t. the global frame.
            - base_point (torch.Tensor of shape (4,) or (b, 4)): the basepoint
              that generates the basis.
        Returns:
            - tangent_vec (torch.Tensor of shape (4,) or (b, 4)): a tangent
              vector given by the linear combination of the coordinates and
              the basis.
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

        basis = self._basis_from_base_point(base_point, transport_from=transport_from)

        tangent_vec = (
            coords[:, 0].unsqueeze(1) * basis[..., 0]
            + coords[:, 1].unsqueeze(1) * basis[..., 1]
        )

        if not batched:
            tangent_vec = tangent_vec.squeeze(0)

        return tangent_vec

    def _from_tangent_vec_to_coordinates(
        self,
        tangent_vec: torch.Tensor,
        base_point: torch.Tensor,
        transport_from: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the coordinates of the tangent vector w.r.t.
        the basis spanned by the base point.

        Parameters:
            - tangent_vec (torch.Tensor of shape (3,) or (b, 3)).
            - base_point (torch.Tensor of shape (3,) or (b, 3)).
        """
        if len(base_point.shape) == 1 and len(tangent_vec.shape) > 1:
            base_point = base_point.unsqueeze(0)
            base_point = base_point.repeat([len(tangent_vec), 1])

        basis = self._basis_from_base_point(base_point, transport_from=transport_from)
        # Since the system actually has a solution, I expect this
        # least squares (i.e. multiplying by the pseudoinverse of the
        # basis matrix) to actually converge pretty close to the original.
        # The problem should have a unique minima, and it should be pretty
        # convex.

        # Tests with random coordinates say yes, this works (with a presicion
        # of 1e-5). I wonder if there's a way to implement this solution better,
        # knowing that the tangent vecs actually live in the span of the basis.
        lstsq = LeastSquares()
        coords = lstsq.lstq(basis.type(tangent_vec.dtype), tangent_vec)

        return coords

    def _basis_from_base_point(
        self, base_point: torch.Tensor, transport_from: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Given a basepoint (x, y, z), returns a basis of the tangent space spanned
        by either the QR decomposition of the matrix

        [
            [-z, 0],
            [0, z],
            [x, -y],
        ]

        as a list of torch.Tensors, or the result of parallely-transporting said
        basis from {transport_from}.

        Parameters:
            - base_point (torch.Tensor of shape (3,) or (b, 3)).
            - transport_from (torch.Tensor of shape (3,) or (b, 3)): a
              priviledged point from which we transport the orthonormal basis

        Returns:
            - basis (List[torch.Tensor], each Tensor of shape (3,) or (b, 3)).
        """
        if len(base_point.shape) == 1:
            batched = False
            base_point = base_point.unsqueeze(0)
        else:
            batched = True

        if transport_from is not None:
            # Computes the basis there
            basis_there = self._basis_from_base_point(transport_from)

            # Computes the direction in which we need to move
            # (using the super methods).
            direction = super().log(base_point, transport_from)

            # Transports it
            basis_here = [
                [
                    self.parallel_transport(
                        basis_vec,
                        base_point=transport_from,
                        direction=dir_,
                        return_tangent_vec=True,
                    )
                    for basis_vec in basis_there.T
                ]
                for dir_ in direction
            ]

            basis = torch.cat(
                [torch.vstack(basis_at).T.unsqueeze(0) for basis_at in basis_here]
            )

            assert basis.shape[1] == 3 and basis.shape[2] == 2

            return basis

        x = base_point[:, 0].unsqueeze(0)
        y = base_point[:, 1].unsqueeze(0)
        z = base_point[:, 2].unsqueeze(0)
        zeros = torch.zeros_like(z)

        matrix = [torch.vstack((-z, zeros, x)).T, torch.vstack((zeros, z, -y)).T]
        matrix = torch.cat([v.unsqueeze(-1) for v in matrix], dim=2)

        basis, _ = torch.linalg.qr(matrix)

        assert basis.shape[1] == 3 and basis.shape[2] == 2
        if basis.shape[1] == 2 and basis.shape[2] == 2:
            print("Hmm.")
            print("Stopping here.")

        if not batched:
            basis = basis.squeeze(0)

        return basis

    def parallel_transport(
        self,
        coordinates: torch.Tensor,
        base_point: torch.Tensor,
        end_point: torch.Tensor = None,
        direction: torch.Tensor = None,
        transport_basis_from: torch.Tensor = None,
        return_tangent_vec: bool = False,
    ) -> torch.Tensor:
        # Resolve both end point and direction
        # Assuming that the direction is a vector in R3
        if end_point is None:
            assert (
                direction is not None
            ), "At least one between end point and direction shouldn't be None"
            end_point = super().exp(direction, base_point)

        if direction is None:
            assert (
                end_point is not None
            ), "At least one between end point and direction shouldn't be None"
            direction = super().log(end_point, base_point)

        # Add basis transportation.
        tangent_vec = self._from_coordinates_to_tangent_vec(coordinates, base_point)

        # TODO: why doesn't super() work?
        other_metric = HypersphereMetric(dim=self.dim)
        vector_after_transporting = other_metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        if return_tangent_vec:
            return vector_after_transporting

        coords_in_new_basis = self._from_tangent_vec_to_coordinates(
            vector_after_transporting,
            base_point=end_point,
            transport_from=transport_basis_from,
        )
        return coords_in_new_basis

    def metric_matrix(self, base_point):
        """
        Returns the metric matrix by embedding the tangent vector's basis
        into an ambient space, computing the corresponding components of the
        matrix, and returning a 2x2 matrix representation.

        Parameters
        ----------
        - base_point (torch.Tensor of shape (3,) or (b, 3)): the point
          in the manifold w.r.t. which we want to compute the metric.

        Returns
        -------
        - metric_matrix (torch.Tensor of shape (2, 2) or (b, 2, 2)): the
          matrix representation of the metric at the given basepoints.

        """
        if len(base_point.shape) == 1:
            batched = False
            base_point = base_point.unsqueeze(0)
        else:
            batched = True

        # Getting the transformations of the basis vectors
        basis_in_ambient_space = self._basis_from_base_point(base_point)
        first_basis_vector = basis_in_ambient_space[..., 0].unsqueeze(-1)
        second_basis_vector = basis_in_ambient_space[..., 1].unsqueeze(-1)

        # Computing the symmetric part of the matrix
        metric_matrix = torch.zeros((base_point.shape[0], 2, 2))
        metric_matrix[:, 0, 0] = torch.bmm(
            first_basis_vector.permute(0, 2, 1), first_basis_vector
        ).flatten()
        metric_matrix[:, 0, 1] = torch.bmm(
            first_basis_vector.permute(0, 2, 1), second_basis_vector
        ).flatten()
        metric_matrix[:, 1, 1] = torch.bmm(
            second_basis_vector.permute(0, 2, 1), second_basis_vector
        ).flatten()

        # and symmetrizing the rest.
        metric_matrix[:, 1, 0] = metric_matrix[:, 0, 1]

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
    metric = SphereMetricVectorized()

    coords = torch.randn((100, 2))
    base_point = np.repeat(
        torch.Tensor([[0, np.sqrt(2) / 2, np.sqrt(2) / 2]]), 100, axis=0
    )

    tangent_vec = metric._from_coordinates_to_tangent_vec(coords, base_point)
    new_coords = metric._from_tangent_vec_to_coordinates(tangent_vec, base_point)
    assert torch.isclose(coords, new_coords, atol=1e-5).all()

    # TODO: test in non-batched.
    metric_matrix = metric.metric_matrix(base_point)
    print(metric_matrix)
