import os
import math
import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive

from kernels.gegenbauer_polynomials import gegenbauer_polynomial
from kernels.jacobi_theta_functions import jacobi_theta_function3

dirname = os.path.dirname(os.path.realpath(__file__))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'
device = 'cpu'


def sphere_distance_torch(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a sphere manifold.

    Parameters
    ----------
    :param x1: points on the sphere                                             N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the sphere                                             N2 x dim or b1 x ... x bk x N2 x dim

    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    if diag is False:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)

        # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
        x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
        x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

        # Expand dimension to perform inner product
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-1)

        # Compute the inner product (should be [-1,1])
        inner_product = torch.bmm(x1.view(-1, 1, x1.shape[-1]), x2.view(-1, x2.shape[-2], 1)).view(x1.shape[:-2])

    else:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-1).transpose(-1, -2)
        x2 = x2.unsqueeze(-1)
        inner_product = torch.bmm(x1, x2).squeeze(-1)

    # Clamp in case any value is not in the interval [-1,1]
    # A small number is added/substracted to the bounds to avoid NaNs during backward computation.
    inner_product = inner_product.clamp(-1.+1e-15, 1.-1e-15)

    return torch.acos(inner_product)


class SphereRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the sphere manifold.

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel
    self.cst_nd, precomputed constant term of the summation formula of the kernel (term n, dimension d)
    self.zero_gpolynomials, precompute Gegenbauer polynomial for 0-distance (for the normalizing term of the kernel)

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, nu=None, serie_nb_terms=10, nu_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param nu_prior: prior function on the smoothness parameter
        :param kwargs: additional arguments
        """

        self.has_lengthscale = True
        super(SphereRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Register smoothness parameter
        self.register_parameter(name="raw_nu", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if nu_prior is not None:
            self.register_prior("nu_prior", nu_prior, lambda module: module.nu,
                                lambda module, value: module._set_nu(value))

        # A Positive constraint is defined on the smoothness parameter.
        self.register_constraint("raw_nu", Positive())

        # If the smoothness parameter is given, set it and deactivate its optimization by setting requires_grad false
        if nu is not None:
            self.nu = nu
            self.raw_nu.requires_grad = False

        # Dimension of sphere data
        self.dim = dim

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

        # Precompute constant terms in the sum of the serie defining the kernel
        self.cst_nd = [compute_riemannian_matern_kernel_constant(n, self.dim).to(device) for n in range(self.serie_nb_terms)]

        # Precompute Gegenbauer polynomial for 0-distance (used to compute the normalizing term of the kernel)
        self.zero_gpolynomials = [gegenbauer_polynomial(n, (self.dim-1.)/2., torch.ones(1)).to(device) for n in range(self.serie_nb_terms)]

    @property
    def nu(self):
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value):
        self._set_nu(value)

    def _set_nu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute cos of distance
        cos_distance = torch.cos(sphere_distance_torch(x1, x2, diag=diag))

        # # Compute serie and normalization factor
        # kernel = torch.zeros_like(cos_distance)
        # norm_factor = torch.zeros((1, 1))
        # for n in range(self.serie_nb_terms):
        #     # Compute exponential term
        #     exp_term = torch.pow(2*self.nu/self.lengthscale**2 + n*(n+self.dim-1), -(self.nu + self.dim/2))
        #     # Compute Gegenbauer polynomial
        #     gpolynomial = gegenbauer_polynomial(n, (self.dim-1.)/2., cos_distance)
        #
        #     # Kernel serie's n-th term
        #     kernel += exp_term * self.cst_nd[n] * gpolynomial
        #     # Normalization factor serie's n-th term
        #     norm_factor += exp_term * self.cst_nd[n] * self.zero_gpolynomials[n]

        # Compute serie and normalization factor
        kernel = torch.zeros_like(cos_distance)
        norm_factor = torch.zeros((1, 1)).to(device)
        exp_term0 = torch.pow(2*self.nu/self.lengthscale**2, -(self.nu + self.dim/2))
        for n in range(self.serie_nb_terms):
            # Compute exponential term normalized by exp_term0 to avoid too small values and numerical errors
            exp_term = torch.pow(2*self.nu/self.lengthscale**2 + n*(n+self.dim-1), -(self.nu + self.dim/2)) / exp_term0
            # Compute Gegenbauer polynomial
            gpolynomial = gegenbauer_polynomial(n, (self.dim - 1.) / 2., cos_distance)

            # Kernel serie's n-th term
            kernel += exp_term * self.cst_nd[n] * gpolynomial
            # Normalization factor serie's n-th term
            norm_factor += exp_term * self.cst_nd[n] * self.zero_gpolynomials[n]

        # Kernel
        return kernel / norm_factor


class SphereRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.

    Attributes
    ----------
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel
    self.cst_nd, precomputed constant term of the summation formula of the kernel (term n, dimension d)
    self.zero_gpolynomials, precompute Gegenbauer polynomial for 0-distance (for the normalizing term of the kernel)

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, serie_nb_terms=10,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SphereRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Dimension of sphere data
        self.dim = dim

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

        # Precompute constant terms in the sum of the serie defining the kernel
        self.cst_nd = [compute_riemannian_matern_kernel_constant(n, self.dim).to(device) for n in range(self.serie_nb_terms)]

        # Precompute Gegenbauer polynomial for 0-distance (used to compute the normalizing term of the kernel)
        self.zero_gpolynomials = [gegenbauer_polynomial(n, (self.dim-1.)/2., torch.ones(1)).to(device) for n in range(self.serie_nb_terms)]

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute cos of distance
        cos_distance = torch.cos(sphere_distance_torch(x1, x2, diag=diag))

        # Compute serie and normalization factor
        kernel = torch.zeros_like(cos_distance)
        norm_factor = torch.zeros((1, 1)).to(device)
        for n in range(self.serie_nb_terms):
            # Compute exponential term
            exp_term = torch.exp(-self.lengthscale**2/2. * n * (n+self.dim-1))
            # Compute Gegenbauer polynomial
            gpolynomial = gegenbauer_polynomial(n, (self.dim-1.)/2., cos_distance)

            # Kernel serie's n-th term
            kernel += exp_term * self.cst_nd[n] * gpolynomial
            # Normalization factor serie's n-th term
            norm_factor += exp_term * self.cst_nd[n] * self.zero_gpolynomials[n]

        # Kernel
        return kernel / norm_factor


def compute_riemannian_matern_kernel_constant(n, d):
    """
    This function computes the constant terms of the summation formula for the Matérn kernel on the sphere.

    Parameters
    ----------
    :param n: term n of the summation serie
    :param d: dimension of sphere S^d

    Returns
    -------
    :return: constant term c_nd

    """
    dn = (2*n+d-1) * math.gamma(n+d-1) / math.gamma(d) / math.gamma(n+1)
    # Compute Gegenbauer polynomial
    gpolynomial = gegenbauer_polynomial(n, (d-1.)/2., torch.ones(1))
    # Constant value
    cnd = dn * math.gamma((d-1.)/2.) / (2*math.pow(math.pi, (d-1.)/2.) * gpolynomial)
    return cnd


class CircleRiemannianMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the circle, i.e.,
    sphere manifold S¹.

    Attributes
    ----------
    self.nu, smoothness parameter

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, nu,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        self.nu, smoothness parameter

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        if nu not in {0.5, 1.5, 2.5}:  # TODO: we can remove this once the implementation for nu > 2.5 works.
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")

        self.has_lengthscale = True
        super(CircleRiemannianMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Smoothness parameter
        self.nu = nu

    def compute_constant_parameters(self):
        # TODO: this is not working.
        # Compute a_ss
        s = int(self.nu - 0.5)
        a_ss = 1. / (torch.pow(-self.nu / (np.pi**2 * self.lengthscale ** 2), s) * math.factorial(s))

        # Compute H matrix
        h_matrix = torch.zeros((s, s+1))
        for r in range(s):
            for k in range(s+1):
                for j in range(2*r+1+1):  # +1 to include 2*r+1 in the sum
                    power_term = torch.pow(math.sqrt(2*self.nu)/(2*self.lengthscale), k-j)

                    if j == 0:
                        falling_factorial_term = 1.
                    elif j > k:
                        falling_factorial_term = 0.
                    else:
                        falling_factorial_term = np.prod([k - i for i in range(j)])

                    binomial_coeff_term = math.factorial(2*r+1) / math.factorial(2*r+1-j) / math.factorial(j)
                        # binomial_coeff_term = math.comb(2*r+1, j)
                    if (k-j+1) % 2 == 0:
                        hyperbolic_term = torch.pow(torch.sinh(math.sqrt(2*self.nu)/(2*self.lengthscale)), k-j+1)
                    else:
                        hyperbolic_term = torch.pow(torch.cosh(math.sqrt(2*self.nu)/(2*self.lengthscale)), k-j+1)
                    h_matrix[r, k] += binomial_coeff_term * falling_factorial_term * power_term[0, 0] * hyperbolic_term[0, 0]

        # Compute a terms
        a = - a_ss * torch.matmul(torch.inverse(h_matrix[:, :-1]), h_matrix[:, -1])
        return torch.hstack((a, a_ss))

    def coth(self, z):
        # expz = torch.exp(z)
        # expmz = torch.exp(-z)
        # return (expz + expmz) / (expz - expmz)
        expmz = torch.exp(-2*z)
        return (1 + expmz) / (1 - expmz)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a circle / sphere manifold S^1.

        Parameters
        ----------
        :param x1: input points on the circle
        :param x2: input points on the circle

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag) / (2 * np.pi)
        
        if self.nu == 0.5:
            # Compute kernel
            kernel = torch.cosh((scaled_distance - 0.5) / self.lengthscale)

            # Compute normalization factor
            norm_factor = torch.cosh(-0.5 / self.lengthscale)
            
        elif self.nu == 1.5:
            # Compute kernel
            u = np.sqrt(3.) * (scaled_distance - 0.5) / self.lengthscale
            sinh_term = 2. * self.lengthscale * u * torch.sinh(u)
            cosh_cst = 2. * self.lengthscale + np.sqrt(3.) * self.coth(np.sqrt(3.) / (2 * self.lengthscale))
            cosh_term = cosh_cst * torch.cosh(u)
            kernel = np.pi**2 * self.lengthscale / 3. * (cosh_term - sinh_term)

            # Compute normalization factor
            u_norm = -0.5 * np.sqrt(3.)/self.lengthscale
            sinh_term_norm_factor = 2. * self.lengthscale * u_norm * torch.sinh(u_norm)
            cosh_term_norm_factor = cosh_cst * torch.cosh(u_norm)
            norm_factor = np.pi**2 * self.lengthscale / 3. * (cosh_term_norm_factor - sinh_term_norm_factor)

        elif self.nu == 2.5:
            # TODO: this is not working.
            # Compute constants
            coth_term = self.coth(np.sqrt(5.) / (2 * self.lengthscale))
            # a20 = - np.pi ** 4 * self.lengthscale ** 2 / 50. * \
            #       (-5. + 12. * self.lengthscale ** 2 + 6. * np.sqrt(5.) * self.lengthscale * coth_term
            #        + 10. * coth_term ** 2)
            # a21 = 2 * np.pi ** 4 * self.lengthscale ** 3 / 25. * (3. * self.lengthscale + np.sqrt(5.) * coth_term)
            # a22 = - 2. * np.pi ** 4 * self.lengthscale ** 4 / 25.
            a20 = - 1 / 50. * \
                  (-5. + 12. * self.lengthscale ** 2 + 6. * np.sqrt(5.) * self.lengthscale * coth_term
                   + 10. * self.coth(5. / (4 * self.lengthscale**2)))
                   # + 10. * coth_term ** 2)
            a21 = 2 * self.lengthscale / 25. * (3. * self.lengthscale + np.sqrt(5.) * coth_term)
            a22 = - 2. * self.lengthscale ** 2 / 25.

            # Compute kernel
            u = np.sqrt(5.) * (scaled_distance - 0.5) / self.lengthscale
            cosh_u = torch.cosh(u)
            sinh_u = torch.sinh(u)
            kernel = a20 * cosh_u + a21 * u * sinh_u * a22 * u**2 * cosh_u

            # Compute normalization factor
            u_norm = - 0.5 * np.sqrt(5.) / self.lengthscale
            cosh_u_norm = torch.cosh(u_norm)
            sinh_u_norm = torch.sinh(u_norm)
            norm_factor = a20 * cosh_u_norm + a21 * u_norm * sinh_u_norm * a22 * u_norm**2 * cosh_u_norm

        else:
            # TODO: this is not working.
            # Compute constant parameters
            const_terms = self.compute_constant_parameters()
    
            # Compute serie and normalization factor
            s = int(self.nu - 0.5)
            kernel = torch.zeros_like(scaled_distance)
            for k in range(s+1):  # +1 to include s in the sum
                scaled_distance_term = math.sqrt(2 * self.nu) * (scaled_distance - 0.5) / self.lengthscale
                power_term = torch.pow(scaled_distance_term, k)
                if k % 2 == 0:
                    hyperbolic_term = torch.pow(torch.sinh(scaled_distance_term), k)
                else:
                    hyperbolic_term = torch.pow(torch.cosh(scaled_distance_term), k)
                kernel += const_terms[:, k] * power_term * hyperbolic_term
    
            # Compute normalization factor
            norm_factor = torch.zeros((1, 1))
            for k in range(s+1):
                scaled_distance = math.sqrt(2 * self.nu) * (- 0.5) / self.lengthscale
                power_term = torch.pow(scaled_distance, k)
                if k % 2 == 0:
                    hyperbolic_term = torch.pow(torch.sinh(scaled_distance), k)
                else:
                    hyperbolic_term = torch.pow(torch.cosh(scaled_distance), k)
                norm_factor += const_terms[:, k] * power_term * hyperbolic_term

        # Kernel
        return kernel / norm_factor


class CircleRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the circle, i.e.,
    sphere manifold S¹.

    Attributes
    ----------
    self.serie_nb_terms, number of terms used to compute the Jacobi theta function of the kernel

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, serie_nb_terms=100,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------

        Optional parameters
        -------------------
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(CircleRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Number of term used to compute the jacobi theta function
        self.serie_nb_terms = serie_nb_terms

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a circle / sphere manifold S^1.

        Parameters
        ----------
        :param x1: input points on the circle
        :param x2: input points on the circle

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag)/(2*np.pi)

        # Compute kernel equal to jacobi theta function
        q_param = torch.exp(-2 * np.pi**2 * self.lengthscale**2).to(device)
        kernel = jacobi_theta_function3(np.pi * scaled_distance, q_param).to(device)

        # Normalizing term
        norm_factor = jacobi_theta_function3(torch.zeros((1, 1)).to(device), q_param).to(device)

        # Kernel
        return kernel / norm_factor


class CircleRiemannianIntegratedMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on the circle manifold obtained
    by integrating over the heat kernel

    Attributes
    ----------
    self.nu, smoothness parameter
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.nb_points_integral, number of points used to compute the integral over the heat kernel
    self.serie_nb_terms, number of terms used to compute the summation formula of the kernel

    Methods
    -------
    link_function(distance, precomputed_Gegenbauer_polynomial, t)
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, nu=None, nu_prior=None, serie_nb_terms=20, nb_points_integral=50, **kwargs):
        """
        Initialisation.

        Parameters
        ----------

        Optional parameters
        -------------------
        :param nu: smoothness parameter, it will be selected automatically (optimized) if it is not given.
        :param nu_prior: prior function on the smoothness parameter
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param nb_points_integral: number of points used to compute the integral over the heat kernel
        :param kwargs: additional arguments
        """

        self.has_lengthscale = True
        super(CircleRiemannianIntegratedMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Register smoothness parameter
        self.register_parameter(name="raw_nu", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if nu_prior is not None:
            self.register_prior("nu_prior", nu_prior, lambda module: module.nu,
                                lambda module, value: module._set_nu(value))

        # A Positive constraint is defined on the smoothness parameter.
        self.register_constraint("raw_nu", Positive())

        # Dimension of the sphere
        self.dim = 1

        # If the smoothness parameter is given, set it and deactivate its optimization by setting requires_grad false
        if nu is not None:
            self.nu = nu
            self.raw_nu.requires_grad = False

        # Number of points for the integral computation
        self.nb_points_integral = nb_points_integral

        # Number of term used to approximate the infinite serie approximating the kernel
        self.serie_nb_terms = serie_nb_terms

    @property
    def nu(self):
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value):
        self._set_nu(value)

    def _set_nu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    def link_function(self, scaled_distance, t):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.

        Parameters
        ----------
        :param distance: precomputed scaled distance between the inputs
        :param t: heat kernel lengthscale

        Returns
        -------
        :return: link function between the heat and Matérn kernels

        """
        # Compute unnormalized heat kernel
        q_param = torch.exp(-4 * np.pi ** 2 * t)
        heat_kernel = jacobi_theta_function3(np.pi * scaled_distance, q_param).to(device)

        result = torch.pow(t, self.nu + self.dim / 2 - 1.0) \
                 * torch.exp(- 2.0 * self.nu / self.lengthscale ** 2 * t) \
                 * heat_kernel

        return result

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the integrated Matérn kernel matrix between inputs x1 and x2 belonging to a circle manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag) / (2 * np.pi)

        # Evaluate integral
        shift = torch.log10(self.lengthscale).item()
        t_vals = torch.logspace(-3 + shift, 1 + shift, self.nb_points_integral).to(device)
        integral_vals = torch.zeros([self.nb_points_integral] + list(scaled_distance.shape)).to(device)
        for i in range(self.nb_points_integral):
            integral_vals[i] = self.link_function(scaled_distance, t_vals[i])
        # Kernel
        kernel = torch.trapz(integral_vals, t_vals, dim=0)

        # Evaluate the integral for the normalizing constant
        integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral).to(device)
        for i in range(self.nb_points_integral):
            integral_vals_normalizing_cst[i] = self.link_function(torch.zeros(1, 1).to(device), t_vals[i])
        # Normalizing constant
        normalizating_cst = torch.trapz(integral_vals_normalizing_cst, t_vals, dim=0)

        # Kernel
        return kernel / normalizating_cst


class SphereApproximatedGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.
    This covariance matrix is an approximation of the SphereRiemannianGaussianKernel, where the Euclidean distance is
    replaced by the geodesic distance in a Euclidean-like RBF kernel.

    Attributes
    ----------
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.beta_min, minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, beta_min=None, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the sphere S^d on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param beta_min: minimum value of the inverse square lengthscale parameter beta.
                         If None, it is determined automatically.
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(SphereApproximatedGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)

        # Define beta_min
        if beta_min is None:
            if dim == 2:
                beta_min = 6.5
            elif dim == 3:
                beta_min = 2.
            elif dim == 4:
                beta_min = 1.2
            elif dim == 5:
                beta_min = 1.0
            elif dim <= 10:
                beta_min = 0.6
            elif dim <= 15:
                beta_min = 0.42
            elif dim <= 50:
                beta_min = 0.35
            elif 50 <= dim <= 160:
                beta_min = 0.21

        self.dim = dim
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda module: module.beta,
                                lambda module, value: module._set_beta(value))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee positive-definiteness.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_beta(value)

    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        distance2 = torch.mul(distance, distance)
        # Kernel
        exp_component = torch.exp(- distance2.mul(self.beta.double()))
        return exp_component


class SphereRiemannianLaplaceKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Laplace covariance matrix between input points on the sphere manifold.
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SphereRiemannianLaplaceKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Laplace kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        # Kernel
        exp_component = torch.exp(- distance.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))
        return exp_component
