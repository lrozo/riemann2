import torch
import torch.nn.functional

from gpytorch.models.gplvm.latent_variable import LatentVariable
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.priors import Prior


class BackConstraintsLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models to estimate the latent variables using back constraints.
    The latent variable x is computed as a function of the corresponding observations y.
    Namely, given the training targets (or observations) y_n, each dimension j of x is computed as
    x_j = \sum_n w_jn k(y, y_n)
    with k a kernel function on the observation space.

    Attributes
    ----------
    self.n (int): size of the latent space.
    self.latent_dim (int): dimensionality of latent space.
    self.train_targets (torch.Tensor): training observations  [n x dimension]
    self.targets_kernel (gpytorch.Kernel): kernel defining the relationship between the observations for the BC function

    Methods
    -------
    self.forward(self)
    self.back_constraint_function(self, targets)
    """

    def __init__(self, n, latent_dim, train_targets, targets_kernel: Kernel = None, weight_prior: Prior = None,
                 weight_init: torch.Tensor = None, prior_x: Prior = None):
        """
        Initialization.

        Parameters
        ----------
        :param n: size of the latent space
        :param latent_dim: dimension of the latent space
        :param train_targets: training observations  [n x dimension]

        Optional parameters
        -------------------
        :param targets_kernel: kernel defining the relationship between the observations for the back constraints
        :param weight_prior: prior for the weight parameter of the back constraints
        :param weight_init: initial weight parameter
        """
        super().__init__(n, latent_dim)

        # Data
        self.train_targets = train_targets

        # Define kernels
        if targets_kernel:
            self.targets_kernel = targets_kernel
        else:
            self.targets_kernel = ScaleKernel(RBFKernel())

        # Ensure that kernel parameters are not trainable (necessary for back constraints)
        for parameter in self.targets_kernel.parameters():
            parameter.requires_grad = False

        # Define back constraints weights
        if weight_init is None:
            weight_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        else:
            weight_init = torch.nn.Parameter(weight_init)
        self.register_parameter("weights", weight_init)
        if weight_prior:
            self.register_prior("prior_w", weight_prior, "weights")
        if prior_x:
            kernel_matrix = self.targets_kernel(self.train_targets, self.train_targets)
            self.register_prior("prior_x", prior_x, lambda module: torch.matmul(kernel_matrix.evaluate(), module.weights))

    def back_constraint_function(self, targets):
        """
        Computes the back constraint function x_j = \sum_n w_jn k(y, y_n).

        Parameters
        ----------
        :param targets: observations [nb_data x dimension]

        Returns
        -------
        :return x: latent variables [nb_data x latent dimension]
        """
        kernel_matrix = self.targets_kernel(targets, self.train_targets)
        x = torch.matmul(kernel_matrix.evaluate(), self.weights)
        return x

    def forward(self):
        """
        Computes the back constraint function for the training targets.

        Returns
        -------
        :return x: latent variables corresponding to the training targets [nb_data x latent dimension]
        """
        x = self.back_constraint_function(self.train_targets)
        return x
