import torch

from geomstats.geometry.spd_matrices import SPDAffineMetric
from geomstats.geometry.hermitian_matrices import expmh


class SPDMetricAffineVectorized(SPDAffineMetric):
    """
    A Wrapper around the affine metric for the SPD Manifold
    provided by geomstats. In our framework, we are expecting
    tangent vectors to be 6-dimensional, but geomstats expects
    a 3x3 symmetric matrix. This wrapper handles that disconnect
    by re-implementing log and exp.

    This vectorization comes with considering the tangent space
    at SPD (i.e. the symmetric matrices) as R6 after taking the
    canonical basis. This connection between symmetric matrices
    and tangent vectors is taken care of by the local methods
    _matrix_from_tangent_vector and _tangent_vec_from_matrix
    """

    def __init__(self, space):
        super().__init__(space) 
        self.n = space.n
        self.tangent_space_dim = space.dim
        self.ambient_space_dim = space.n**2

    def exp(self, tangent_vec: torch.Tensor, base_point: torch.Tensor, **kwargs):
        """
        Computes the exponential map of the SPD manifold, taking tangent vectors
        as vectors of coordinates k=n(n+1)/2 coordinates. Returns an nxn SPD matrix.

        Parameters:
            - tangent_vec (torch.Tensor of shape (b, k)): the coordinates
              of a tangent vector.
            - base_point (torch.Tensor of shape (b, n, n) or (b, n**2)): the
              basepoints on the manifold from which the tangent vectors are
              "shot" using the exponential.

        Returns:
            - exp (torch.Tensor of shape (b, n, n)): the exponential of
              the tangent vectors at the given basepoints.
        """
        if len(base_point.shape) == 2 and base_point.shape[1] == self.n**2:
            base_point = base_point.view(-1, self.n, self.n)
        elif len(base_point.shape) == 1 and base_point.shape[0] == self.n**2:
            base_point = base_point.view(self.n, self.n)

        # At this point, we are sure that base_point is a matrix or
        # batch of matrices.

        tangent_matrix = self._matrix_from_tangent_vec(tangent_vec)

        return super().exp(tangent_matrix, base_point, **kwargs)

    def log(self, point: torch.Tensor, base_point: torch.Tensor, **kwargs):
        """
        Computes the logarithm map for the SPD manifold, returning a
        tangent vector of shape k = n(n+1)/2 + n. This is the tangent vector
        that is needed to get from the basepoints to the points using
        the exponential map.

        Parameters:
            - point (torch.Tensor of shape (b, n, n) or (b, n**2)): the points
              we want to "lift" to the tangent space.
            - base_point (torch.Tensor of shape (b, n, n) or (b, n**2)): the
              basepoints that are used to define the tangent spaces we want to
              lift to.

        Returns:
            - tangent_vectors (torch.Tensor of shape (b, k)): the tangent vectors
              that "shoot" to the points starting at the base_points.
        """
        if base_point.shape[-1] == self.n**2:
            base_point = base_point.view(*base_point.shape[:-1], self.n, self.n)

        if point.shape[-1] == self.n**2:
            point = point.view(*point.shape[:-1], self.n, self.n)

        tangent_matrix = super().log(point, base_point, **kwargs)

        return self._tangent_vec_from_matrix(tangent_matrix)
    
    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points.

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
        log = self.log(point=point_b, base_point=point_a)

        if point_a.shape[-1] == self.n**2:
            point_a = point_a.view(*point_a.shape[:-1], self.n, self.n)

        log = self._matrix_from_tangent_vec(log)

        return self.squared_norm(vector=log, base_point=point_a)

    def _matrix_from_tangent_vec(self, tangent_vec: torch.Tensor) -> torch.Tensor:
        """
        Grabs an k = n(n+1)/2 tangent vector and returns a symmetric matrix
        by filling the diagonal and upper-triangular part with the vector itself,
        and then forces the lower-triangular part to be symmetric.

        Parameters:
            - tangent_vec (torch.Tensor of shape (b, k))

        Returns:
            - tangent_matrix (torch.Tensor of shape (b, n, n))
        """
        if len(tangent_vec.shape) == 1:
            batched = False
            tangent_vec = tangent_vec.unsqueeze(0)
        else:
            batched = True

        i, j = torch.triu_indices(self.n, self.n)
        tangent_matrices = torch.zeros((*tangent_vec.shape[:-1], self.n, self.n), dtype=tangent_vec.dtype)
        tangent_matrices[..., i, j] = tangent_vec
        tangent_matrices.transpose(-2, -1)[..., i, j] = tangent_vec

        if batched:
            return tangent_matrices
        else:
            return tangent_matrices.squeeze(0)

    def _tangent_vec_from_matrix(self, tangent_matrix: torch.Tensor) -> torch.Tensor:
        """
        Returns the n(n+1)/2-sized vector from a symmetric matrix by considering
        only the diagonal and upper-triangular part.

        Parameters:
            - tangent_matrix (torch.Tensor of shape (b, n, n))

        Returns:
            - tangent_vector (torch.Tensor of shape (b, k) where k = n(n+1)/2).
        """
        if len(tangent_matrix.shape) == 2:
            batched = False
            tangent_matrix = tangent_matrix.unsqueeze(0)
        else:
            batched = True

        i, j = torch.triu_indices(self.n, self.n)
        tangent_vec = tangent_matrix[..., i, j]

        if batched:
            return tangent_vec
        else:
            return tangent_vec.squeeze(0)

    def tangent_space_basis_as_matrices(self) -> torch.Tensor:
        """
        SPDs are parallelizable, and the basis is given
        by the coordinates of the diagonal and upper-triangular
        part of the symmetric matrix.
        """
        i_s, j_s = torch.triu_indices(self.n, self.n)
        basis_matrices = [
            torch.zeros(self.n, self.n) for _ in range(self.tangent_space_dim)
        ]

        # triu indices
        for k, i, j in zip(range(self.tangent_space_dim), i_s, j_s):
            basis_matrices[k][i, j] = 1.0
            basis_matrices[k][j, i] = 1.0

        return torch.cat([bm.unsqueeze(0) for bm in basis_matrices])

    def metric_matrix(self, base_point: torch.Tensor) -> torch.Tensor:
        """
        Computes the metric matrix by evaluating the inner product
        of the basis tangent vectors at the given basepoint,
        assuming g_p_ij = g_p(e_i, e_j).

        Parameters
        ----------

        - base_point (torch.Tensor of shape (n, n), (n**2,) (b, n, n) or
          (b, n**2)) the matrix basepoint in which to evaluate the
          inner product.
        """
        # Making sure that base_point is batched
        if len(base_point.shape) == 1:
            # Making sure it's a single vector of size n**2
            assert base_point.shape[0] == self.n**2
            batched = False
            base_point = base_point.view(1, self.n, self.n)
        elif len(base_point.shape) == 2:
            # We have two cases, it's either (n, n) or (b, n**2)
            if base_point.shape[0] == base_point.shape[1] == self.n:
                print("Warning: assuming what was given is a single matrix.")
                batched = False
                base_point = base_point.unsqueeze(0)
            elif base_point.shape[1] == self.n**2:
                batched = True
                base_point = base_point.view(-1, self.n, self.n)
            else:
                raise ValueError(
                    "base_point's shape is of length 2 but it isn't "
                    f"(b, {self.n **2}) nor ({self.n}, {self.n})"
                )
        else:
            # Can only be (b, n, n)
            assert base_point.shape[-1] == base_point.shape[-2] == self.n
            batched = True

        # At this point, base_point is batched.
        metric_matrix = torch.zeros(
            (len(base_point), self.tangent_space_dim, self.tangent_space_dim)
        )

        basis = self.tangent_space_basis_as_matrices()

        i_s, j_s = torch.triu_indices(self.n, self.n)
        for i, j in zip(i_s, j_s):
            metric_matrix[:, i, j] = self.inner_product(basis[i], basis[j], base_point)
            if i != j:
                metric_matrix[:, j, i] = self.inner_product(
                    basis[i], basis[j], base_point
                )

        if not batched:
            metric_matrix = metric_matrix.squeeze(0)

        return metric_matrix
    
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
        # TODO add comment that we consider that the basepoint is I and check beforehand.
        # TODO add that we do not consider parallel transport.

        u_mat = self._matrix_from_tangent_vec(u)
        exp_from_id = expmh(u_mat)
        logdet_partial =torch.log(torch.det(exp_from_id))

        return logdet_partial.sum()


if __name__ == "__main__":
    spd_3 = SPDMetricAffineVectorized(3)
    metric_matrix = spd_3.metric_matrix(
        torch.eye(3).unsqueeze(0).repeat_interleave(100, dim=0)
    )
    print(metric_matrix)
