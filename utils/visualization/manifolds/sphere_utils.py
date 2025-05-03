import numpy as np
from typing import Union, Tuple
from mayavi import mlab


def expmap(u, x0):
    """
    This function maps a vector u lying on the tangent space of x0 into the manifold.

    Parameters
    ----------
    :param u: vector in the tangent space
    :param x0: basis point of the tangent space

    Returns
    -------
    :return: x: point on the manifold
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(u) < 2:
        u = u[:, None]

    norm_u = np.sqrt(np.sum(u*u, axis=0))
    x = x0 * np.cos(norm_u) + u * np.sin(norm_u)/norm_u

    x[:, norm_u < 1e-16] = x0

    return x


def logmap(x, x0):
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    theta = np.arccos(np.maximum(np.minimum(np.dot(x0.T, x), 1.), -1.))
    u = (x - x0 * np.cos(theta)) * theta/np.sin(theta)

    u[:, theta[0] < 1e-16] = np.zeros((u.shape[0], 1))

    return u


def sphere_distance(x, y):
    """
    This function computes the Riemannian distance between two points on the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param y: point on the manifold

    Returns
    -------
    :return: distance: manifold distance between x and y
    """
    if np.ndim(x) < 2:
        x = x[:, None]

    if np.ndim(y) < 2:
        y = y[:, None]

    # Compute the inner product (should be [-1,1])
    inner_product = np.dot(x.T, y)
    inner_product = np.max(np.min(inner_product, 1), -1)
    return np.arccos(inner_product)


def parallel_transport_operator(x1, x2):
    """
    This function computes the parallel transport operator from x1 to x2.
    Transported vectors can be computed as u.dot(v).

    Parameters
    ----------
    :param x1: point on the manifold
    :param x2: point on the manifold

    Returns
    -------
    :return: operator: parallel transport operator
    """
    if np.sum(x1-x2) == 0.:
        return np.eye(x1.shape[0])
    else:
        if np.ndim(x1) < 2:
            x1 = x1[:, None]

        if np.ndim(x2) < 2:
            x2 = x2[:, None]

        x_dir = logmap(x2, x1)
        norm_x_dir = np.sqrt(np.sum(x_dir*x_dir, axis=0))
        normalized_x_dir = x_dir / norm_x_dir
        u = np.dot(-x1 * np.sin(norm_x_dir), normalized_x_dir.T) + \
            np.dot(normalized_x_dir * np.cos(norm_x_dir), normalized_x_dir.T) + np.eye(x_dir.shape[0]) - \
            np.dot(normalized_x_dir, normalized_x_dir.T)

        return u


def karcher_mean_sphere(data, nb_iter=10):
    """
    This function computes the mean of points lying on the manifold (Fréchet/Karcher mean).

    Parameters
    ----------
    :param data: data points lying on the manifold

    Optional parameters
    -------------------
    :param nb_iter: number of iterations

    Returns
    -------
    :return: m: mean of the datapoints
    """
    # Initialize the mean as equal to the first datapoint
    m = data[:, 0]
    for i in range(nb_iter):
        data_tgt = logmap(data, m)
        m_tgt = np.mean(data_tgt, axis=1)
        m = expmap(m_tgt, m)

    return m


def get_axisangle(d):
    """
    Gets axis-angle representation of a point lying on a unit sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param d: point on the sphere

    Returns
    -------
    :return: axis, angle: corresponding axis and angle representation
    """
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return np.array([0, 0, 1]), 0
    else:
        vec = np.array([-d[1], d[0], 0])
        return vec/norm, np.arccos(d[2])


def rotation_from_sphere_points(x, y):
    """
    Gets the rotation matrix that moves x to y in the geodesic path on the sphere.
    Based on the equations of "Analysis of principal nested spheres", Sung et al. 2012 (appendix)

    Parameters
    ----------
    :param x: point on a sphere
    :param y: point on a sphere

    Returns
    -------
    :return: rotation matrix
    """
    if np.ndim(x) < 2:
        x = x[:, None]
    if np.ndim(y) < 2:
        y = y[:, None]

    dim = x.shape[0]

    in_prod = np.dot(x.T, y)
    in_prod = np.max(np.min(in_prod, 1), -1)
    c_vec = x - y * in_prod
    c_vec = c_vec / np.linalg.norm(c_vec)

    R = np.eye(dim) + np.sin(np.arccos(in_prod)) * (np.dot(y, c_vec.T) - np.dot(c_vec, y.T)) + (in_prod - 1.) * (np.dot(y, y.T) + np.dot(c_vec, c_vec.T))

    return R


def rotation_matrix_to_unit_sphere(R: np.ndarray) -> Union[np.ndarray, int]:
    """
    This function transforms a rotation matrix to a point lying on a sphere (i.e., unit vector).
    This function is valid for rotation matrices of dimension 2 (to S1) and 3 (to S3).

    Parameters
    ----------
    :param R: rotation matrix

    Returns
    -------
    :return: a unit vector on S1 or S3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if R.shape[0] == 3:
        return rotation_matrix_to_quaternion(R)
    elif R.shape[0] == 2:
        return R[:, 0]
    else:
        raise ValueError('Input rotation matrix must be 2x2 or 3x3!')


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    This function transforms a 3x3 rotation matrix into a quaternion.
    This function was implemented based on Peter Corke's robotics toolbox.

    Parameters
    ----------
    :param R: 3x3 rotation matrix

    Returns
    -------
    :return: a quaternion [scalar term, vector term]
    """

    qs = min(np.sqrt(np.trace(R) + 1)/2.0, 1.0)
    kx = R[2, 1] - R[1, 2]   # Oz - Ay
    ky = R[0, 2] - R[2, 0]   # Ax - Nz
    kz = R[1, 0] - R[0, 1]   # Ny - Ox

    if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]) :
        kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1 # Nx - Oy - Az + 1
        ky1 = R[1, 0] + R[0, 1]               # Ny + Ox
        kz1 = R[2, 0] + R[0, 2]               # Nz + Ax
        add = (kx >= 0)
    elif R[1, 1] >= R[2, 2]:
        kx1 = R[1, 0] + R[0, 1]               # Ny + Ox
        ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1 # Oy - Nx - Az + 1
        kz1 = R[2, 1] + R[1, 2]               # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2, 0] + R[0, 2]               # Nz + Ax
        ky1 = R[2, 1] + R[1, 2]               # Oz + Ay
        kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1 # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.zeros(4)
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s*np.array([kx, ky, kz])
        q = np.hstack((qs, qv))

    return q


def unit_sphere_to_rotation_matrix(unit_vector):
    """
    This function transforms a point lying on a sphere (i.e., unit vector) to a rotation matrix.
    This function is valid for rotation matrices of dimension 2 (from S1) and 3 (from S3).

    Parameters
    ----------
    :param unit_vector: a unit vector on S1 or S3

    Returns
    -------
    :return: a rotation matrix 2x2 or 3x3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if np.ndim(unit_vector) < 2:
        unit_vector = unit_vector[:, None]

    if unit_vector.shape[0] == 4:
        return quaternion_to_rotation_matrix(unit_vector)
    elif unit_vector.shape[0] == 2:
        R = np.zeros((2, 2))
        R[:, 0] = unit_vector[:, 0]
        R[0, 1] = unit_vector[1]
        R[1, 1] = -unit_vector[0]
        # Ensure determinant 1
        if np.linalg.det(R) < 0:
            R = np.array([R[:, 1], R[:, 0]])
        return R
    else:
        raise NotImplementedError


def quaternion_to_rotation_matrix(q):
    """
    This function transforms a quaternion into a 3x3 rotation matrix.

    Parameters
    ----------
    :param quaternion: a quaternion or a batch of quaternion    N x [scalar term, vector term]

    Returns
    -------
    :return: Nxx3x3 rotation matrices
    """

    n = q.shape[0]

    R = np.zeros((n, 3, 3))

    for i in range(n):
        w, x, y, z = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
        R[i] = np.array([[2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1]])
    return R


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Gets rotation matrix from axis angle representation using Rodriguez formula.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param axis: unit axis defining the axis of rotation
    :param angle: angle of rotation

    Returns
    -------
    :return: R(ax, angle) = I + sin(angle) x ax + (1 - cos(angle) ) x ax^2 with x the cross product.
    """
    utilde = vector_to_skew_matrix(axis)
    return np.eye(3) + np.sin(angle)*utilde + (1 - np.cos(angle))*utilde.dot(utilde)


def vector_to_skew_matrix(q: np.ndarray) -> np.ndarray:
    """
    Transform a vector into a skew-symmetric matrix

    Parameters
    ----------
    :param q: vector

    Returns
    -------
    :return: corresponding skew-symmetric matrix
    """
    return np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def plot_sphere_mayavi(radius, offset=np.array([0.0, 0.0, 0.0]), n_elems=100, color: Tuple = (0.2, 0.2, 0.2),
                       opacity=0.7, figure=None):
    """
    Plot the sphere with given radius and center (offset) via mayavi package
    """
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    U, V = np.meshgrid(u, v)
    x = radius*np.cos(U) * np.sin(V) + offset[0]
    y = radius*np.sin(U) * np.sin(V) + offset[1]
    z = radius*np.cos(V) + offset[2]

    if figure is None:
        figure = mlab.gcf()

    s = mlab.mesh(x, y, z, color=color, opacity=opacity, figure=figure)
    return s

# def sphere_brownian_motion_generator(x0, total_time, dtime, number_realizations):
#     """
#     TODO comment
#     Parameters
#     ----------
#     x0
#     total_time
#     dtime
#     number_realizations
#
#     Returns
#     -------
#
#     """
#     if np.ndim(x0) < 2:
#         x0 = x0[None]
#
#     dimension = x0.shape[1]
#     # sphere_manifold = pyman_man.Sphere(dimension)
#
#     # Generate brownian steps in the tangent space of the origin
#     nb_steps = int(total_time/dtime)
#     origin = np.zeros(dimension)
#     origin[0] = 1.
#     brownian_steps_origin = np.zeros((number_realizations, nb_steps, dimension))
#     brownian_steps_origin[:, :, 1:] = euclidean_brownian_motion_steps(dimension-1, total_time, dtime,
#                                                                       number_realizations)
#     # brownian_steps_origin = euclidean_brownian_motion_steps(dimension, total_time, dtime, number_realizations)
#     # Brownian motions obtained recursively
#     brownian_motions = np.zeros((number_realizations, nb_steps, dimension))
#     brownian_motions[:, 0, :] = x0[None]
#     for r in range(number_realizations):
#         for n in range(1, nb_steps):
#             # Project current step to the tangent space of the current location
#             # brownian_step = sphere_manifold.proj(brownian_motions[r, n-1, :], brownian_steps_origin[r, n, :])
#             # brownian_step = sphere_manifold.transp(origin, brownian_motions[r, n-1, :], brownian_steps_origin[r, n, :])  # This is not parallel tranport, but an approximation
#             transport_operator = parallel_transport_operator(origin, brownian_motions[r, n-1, :])
#             brownian_step = np.dot(transport_operator, brownian_steps_origin[r, n, :])
#             # Project the Brownian step to the manifold
#             brownian_motions[r, n, :] = expmap(brownian_step, brownian_motions[r, n-1, :])[:, 0]
#             # brownian_motions[r, n, :] = sphere_manifold.exp(brownian_motions[r, n-1, :], brownian_step)
#
#     return brownian_motions
